from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
import torch.optim as optim
<<<<<<< HEAD
import torch.nn.functional as F
=======
>>>>>>> 4be672c (New losses)
from sklearn.metrics import f1_score, accuracy_score

from dataLoader.getDataset import get_IEMOCAP_loaders, lossWeights
from model import Transformer_Based_Model
from src.Plot.lossPlot import plotLoss


class MaskedKLDivLoss(nn.Module):
    def __init__(self, tau=1.0):
        super(MaskedKLDivLoss, self).__init__()
        self.tau = tau

    def forward(self, student_logits, teacher_logits, mask):
<<<<<<< HEAD
        mask = mask.contiguous().view(-1, 1).float()
=======
        mask = mask.view(-1, 1).float()
>>>>>>> 4be672c (New losses)
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
<<<<<<< HEAD
        mask = mask.contiguous().view(-1).float()
=======
>>>>>>> 4be672c (New losses)
        nll = F.nll_loss(log_probs, target, weight=self.weight, reduction='none')
        loss = (nll * mask).sum() / mask.sum()
        return loss

<<<<<<< HEAD

def train_or_eval_model(model, loss_fun, kl_loss, dataloader, epoch, optimizer=None, train=False, writer = None, gamma_1=1.0, gamma_2=1.0, gamma_3=1.0):
<<<<<<< HEAD
=======
def train_or_eval_model(model, loss_fun, kl_loss, dataloader, epoch, optimizer=None,
                        train=False, gamma_1=1.0, gamma_2=1.0, gamma_3=1.0):
>>>>>>> 4be672c (New losses)
=======
>>>>>>> 1cae55d (Grid search and model selection)
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

<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> 1cae55d (Grid search and model selection)
        text_feature = text_feature.permute(0, 1, 3, 2).squeeze(-1)  # (seq_len_text, batch, dim)
        audio_feature = audio_feature.permute(0, 1, 3, 2).squeeze(-1)  # (seq_len_audio, batch, dim)
        label = label.squeeze(-1)  # (batch, seq_len)
        umask = umask.permute(1, 0)  # (seq_len, batch)
        qmask = qmask.permute(0, 1, 3, 2)[:, :, 0, 0]  # (seq_len_spk, batch, dim)

        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))] # Compute the real length of a sequence

        t_log_prob, a_log_prob, all_log_prob, all_prob, t_kl_log_prob, a_kl_log_prob, all_kl_prob = model(text_feature, audio_feature, qmask, umask)
        #print(f"t_log_prob: {t_log_prob}\n a_log_prob: {a_log_prob} \n all_log_prob: {all_log_prob} \n all_prob: {all_prob}")

        t_lp = t_log_prob.view(-1, t_log_prob.size()[2])
        a_lp = a_log_prob.view(-1, a_log_prob.size()[2])
        all_lp = all_log_prob.view(-1, all_log_prob.size()[2])
=======
        t_log_prob, a_log_prob, all_log_prob, all_prob, t_kl_log_prob, a_kl_log_prob, all_kl_prob = model(
            text_feature, audio_feature, qmask, umask
        )

        t_lp = t_log_prob.view(-1, t_log_prob.size(2))
        a_lp = a_log_prob.view(-1, a_log_prob.size(2))
        all_lp = all_log_prob.view(-1, all_log_prob.size(2))
>>>>>>> 4be672c (New losses)
        labels_ = label.view(-1)
        mask_ = umask.contiguous().view(-1).float()

        t_kl_lp = t_kl_log_prob.view(-1, t_kl_log_prob.size(2))
        a_kl_lp = a_kl_log_prob.view(-1, a_kl_log_prob.size(2))
        all_kl_p = all_kl_prob.view(-1, all_kl_prob.size(2))

<<<<<<< HEAD
        loss_task = loss_fun(all_lp, labels_, umask) 
        loss_ce_t = loss_fun(t_lp, labels_, umask)
        loss_ce_a = loss_fun(a_lp, labels_, umask)
        loss_kl_t = kl_loss(t_kl_lp, all_kl_p, umask)
        loss_kl_a = kl_loss(a_kl_lp, all_kl_p, umask)

        loss = gamma_1 * loss_task + gamma_2 * (loss_ce_t + loss_ce_a) + gamma_3 * (loss_kl_t + loss_kl_a)

        lp_ = all_prob.view(-1, all_prob.size()[2])
=======
        loss = (
            gamma_1 * loss_fun(all_lp, labels_, mask_) +
            gamma_2 * (loss_fun(t_lp, labels_, mask_) + loss_fun(a_lp, labels_, mask_)) +
            gamma_3 * (kl_loss(t_kl_lp, all_kl_p, mask_) + kl_loss(a_kl_lp, all_kl_p, mask_))
        )
>>>>>>> 4be672c (New losses)

        lp_ = all_prob.view(-1, all_prob.size(2))
        pred = torch.argmax(lp_, 1)

        preds.append(pred.cpu().numpy())
        labels.append(labels_.cpu().numpy())
        masks.append(mask_.cpu().numpy())
        losses.append(loss.item() * mask_.sum().item())

        if train:
            loss.backward()
<<<<<<< HEAD
            # tensorboard
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(name, param.grad, epoch)
            optimizer.step()
        
    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan')

<<<<<<< HEAD

=======
    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_acc = round(accuracy_score(labels, preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels,preds, sample_weight=masks, average='weighted', zero_division=0) * 100, 2)

    return avg_loss, avg_acc, labels, preds, masks, avg_fscore, loss_task, loss_ce_t, loss_ce_a, loss_kl_t, loss_kl_a, loss


>>>>>>> 1cae55d (Grid search and model selection)
def TrainSDT(temp=1.0, gamma_1=0.1, gamma_2=0.1, gamma_3=0.1, run_name="exp1", return_val_score=False, **kwargs):
    torch.manual_seed(42)

    input_dim = {'text': 768, 'audio': 88, 'speaker':2}
    train_loader, val_loader, test_loader = get_IEMOCAP_loaders(batch_size=16, validRatio=0.2)

    model_dimension = 1024
    n_head = 8
    n_classes = 6

    n_epochs = kwargs.get("n_epochs", 3)
    dropout = kwargs.get("dropout", 0.5)
    lr = kwargs.get("lr", 0.0001)
    weight_decay = kwargs.get("weight_decay", 0.00001)

    tensorboard = kwargs.get("tensorboard", True)
<<<<<<< HEAD
    if run_name != None:
        writer = SummaryWriter(log_dir=f"runs/{run_name}") if tensorboard else None
=======

    writer = SummaryWriter(log_dir=f"runs/{run_name}") if tensorboard else None
>>>>>>> 1cae55d (Grid search and model selection)

    model = Transformer_Based_Model(
        dataset=train_loader,
        input_dimension=input_dim,
        model_dimension=model_dimension,
        temp=temp,
        n_head=n_head,
        n_classes=n_classes,
        dropout=dropout
    )

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_weight = lossWeights()
    loss_fun = MaskedNLLLoss(loss_weight)
    kl_loss = MaskedKLDivLoss(tau=temp)

    logs = {
        'loss_task': [],
        'loss_ce_t': [],
        'loss_ce_a': [],
        'loss_kl_t': [],
        'loss_kl_a': [],
        'loss': [],
        'val_fscore': []
    }

    best_val_fscore = 0

    for e in range(n_epochs):
        # Training
        train_loss, train_acc, train_labels, train_preds, train_masks, train_fscore, loss_task, loss_ce_t, loss_ce_a, loss_kl_t, loss_kl_a, loss  = train_or_eval_model(
            model=model, loss_fun=loss_fun, kl_loss=kl_loss, dataloader=train_loader, epoch=e, writer=writer,
            optimizer=optimizer, train=True, gamma_1=gamma_1, gamma_2=gamma_2, gamma_3=gamma_3
        )

        # Validation
        val_loss, val_acc, val_labels, val_preds, val_masks, val_fscore, *_ = train_or_eval_model(
            model=model, loss_fun=loss_fun, kl_loss=kl_loss, dataloader=val_loader, epoch=e, writer=writer,
            optimizer=None, train=False, gamma_1=gamma_1, gamma_2=gamma_2, gamma_3=gamma_3
        )

        if tensorboard:
            writer.add_scalar('train: accuracy', train_acc, e)
            writer.add_scalar('train: fscore', train_fscore, e)
            writer.add_scalar('val: accuracy', val_acc, e)
            writer.add_scalar('val: fscore', val_fscore, e)

<<<<<<< HEAD
        print(f"Epoch: {e}")
=======
        print(f"Epoch: {e}/{n_epochs}")
>>>>>>> 1cae55d (Grid search and model selection)
        print(f"Train -> loss: {train_loss}, acc: {train_acc}, fscore: {train_fscore}")
        print(f"Val   -> loss: {val_loss}, acc: {val_acc}, fscore: {val_fscore}")

        logs['loss_task'].append(loss_task.detach())
        logs['loss_ce_t'].append(loss_ce_t.detach())
        logs['loss_ce_a'].append(loss_ce_a.detach())
        logs['loss_kl_t'].append(loss_kl_t.detach())
        logs['loss_kl_a'].append(loss_kl_a.detach())
        logs['loss'].append(loss.detach())
        logs['val_fscore'].append(val_fscore)

        if val_fscore > best_val_fscore:
            best_val_fscore = val_fscore

    plotLoss(logs, n_epochs)

    if tensorboard:
        writer.close()

    if return_val_score:
        return best_val_fscore

def grid_search(param_grid, fixed_params=None):
    """
    param_grid: dict of hyperparameters to grid search (values must be lists)
    fixed_params: dict of additional fixed parameters passed to TrainSDT
    """
    if fixed_params is None:
        fixed_params = {}

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    best_score = -1
    best_config = None
    all_results = []
    num_config = len(list(product(*values)))

    for i, combo in enumerate(product(*values)):
        config = dict(zip(keys, combo))
        full_config = {**fixed_params, **config}

        run_name = "grid_" + "_".join([f"{k}{v}" for k, v in config.items()])
        print(f"\n🔍 Running config {i+1}/{num_config}: {config}")

<<<<<<< HEAD
        score = TrainSDT(**full_config, run_name=None, return_val_score=True)
=======
        score = TrainSDT(**full_config, run_name=run_name, return_val_score=True)
>>>>>>> 1cae55d (Grid search and model selection)
        print(f"Validation F1-score: {score}")

        all_results.append((config, score))

        if score > best_score:
            best_score = score
            best_config = config

    print(f"\nBest Config: {best_config}")
    print(f"Best Validation F1: {best_score}")
    return best_config, best_score, all_results


<<<<<<< HEAD
=======
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
>>>>>>> 4be672c (New losses)
=======
>>>>>>> 1cae55d (Grid search and model selection)
