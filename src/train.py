import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score

from preprocessing.getDataset import get_IEMOCAP_loaders
from model import Transformer_Based_Model

class MaskedKLDivLoss(nn.Module):
    def __init__(self, tau=1.0):
        super(MaskedKLDivLoss, self).__init__()
        self.tau = tau

    def forward(self, student_logits, teacher_logits, mask):
        mask = mask.view(-1, 1).float()
        teacher_soft = F.softmax(teacher_logits / self.tau, dim=-1)
        student_log_soft = F.log_softmax(student_logits / self.tau, dim=-1)
        kl = F.kl_div(student_log_soft, teacher_soft, reduction='none').sum(dim=1)
        loss = (kl * mask.squeeze()).sum() / mask.sum()
        return loss * (self.tau ** 2)

class MaskedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight

    def forward(self, log_probs, target, mask):
        nll = F.nll_loss(log_probs, target, weight=self.weight, reduction='none')
        loss = (nll * mask).sum() / mask.sum()
        return loss

def train_or_eval_model(model, loss_fun, kl_loss, dataloader, epoch, optimizer=None,
                        train=False, gamma_1=1.0, gamma_2=1.0, gamma_3=1.0):
    preds, losses, labels, masks = [], [], [], []

    model.train() if train else model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        text_feature, audio_feature, qmask, umask, label, vid = data
        text_feature = text_feature.permute(0, 1, 3, 2).squeeze(-1)
        audio_feature = audio_feature.permute(0, 1, 3, 2).squeeze(-1)
        label = label.squeeze(-1)
        umask = umask.permute(1, 0)
        qmask = qmask.permute(0, 1, 3, 2)[:, :, 0, 0]

        t_log_prob, a_log_prob, all_log_prob, all_prob, t_kl_log_prob, a_kl_log_prob, all_kl_prob = model(
            text_feature, audio_feature, qmask, umask
        )

        t_lp = t_log_prob.view(-1, t_log_prob.size(2))
        a_lp = a_log_prob.view(-1, a_log_prob.size(2))
        all_lp = all_log_prob.view(-1, all_log_prob.size(2))
        labels_ = label.view(-1)
        mask_ = umask.contiguous().view(-1).float()

        t_kl_lp = t_kl_log_prob.view(-1, t_kl_log_prob.size(2))
        a_kl_lp = a_kl_log_prob.view(-1, a_kl_log_prob.size(2))
        all_kl_p = all_kl_prob.view(-1, all_kl_prob.size(2))

        loss = (
            gamma_1 * loss_fun(all_lp, labels_, mask_) +
            gamma_2 * (loss_fun(t_lp, labels_, mask_) + loss_fun(a_lp, labels_, mask_)) +
            gamma_3 * (kl_loss(t_kl_lp, all_kl_p, mask_) + kl_loss(a_kl_lp, all_kl_p, mask_))
        )

        lp_ = all_prob.view(-1, all_prob.size(2))
        pred = torch.argmax(lp_, 1)

        preds.append(pred.cpu().numpy())
        labels.append(labels_.cpu().numpy())
        masks.append(mask_.cpu().numpy())
        losses.append(loss.item() * mask_.sum().item())

        if train:
            loss.backward()
            optimizer.step()

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    masks = np.concatenate(masks)

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_acc = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)

    return avg_loss, avg_acc, labels, preds, masks, avg_fscore

if __name__ == "__main__":
    input_dim = {'text': 768, 'audio': 88, 'speaker': 2}
    train_loader, val_loader, test_loader = get_IEMOCAP_loaders(batch_size=16, validRatio=0.2)

    model = Transformer_Based_Model(
        dataset=train_loader,
        input_dimension=input_dim,
        model_dimension=128,
        temp=1,
        n_head=8,
        n_classes=11,
        dropout=0.1
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_weight = torch.rand(11)
    loss_fun = MaskedNLLLoss(loss_weight)
    kl_loss = MaskedKLDivLoss(tau=1.0)

    for e in range(1, 151):
        avg_loss, avg_acc, labels, preds, masks, avg_fscore = train_or_eval_model(
            model=model,
            loss_fun=loss_fun,
            kl_loss=kl_loss,
            dataloader=train_loader,
            epoch=e,
            optimizer=optimizer,
            train=True
        )
        print(f"[Epoch {e}] Loss: {avg_loss} | Acc: {avg_acc}% | F1: {avg_fscore}%")
