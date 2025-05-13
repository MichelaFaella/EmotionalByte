import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report

from preprocessing.getDataset import get_IEMOCAP_loaders
from model import Transformer_Based_Model

class MaskedKLDivLoss(nn.Module):
    def __init__(self):
        super(MaskedKLDivLoss, self).__init__()
        self.loss = nn.KLDivLoss(reduction='sum')

    def forward(self, log_pred, target, mask):
        mask_ = mask.contiguous().view(-1, 1)
        loss = self.loss(log_pred * mask_, target * mask_) / torch.sum(mask)   
        return loss

class MaskedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target, mask):
        mask_ = mask.contiguous().view(-1, 1)
        if type(self.weight) == type(None):
            print(np.shape(pred * mask_))
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            print(np.shape(pred * mask_))
            loss = self.loss(pred * mask_, target) \
                   / torch.sum(self.weight[target] * mask_.squeeze())  
        return loss

def train_or_eval_model(model, loss_fun, kl_loss, dataloader, epoch, optimizer=None, train=False, gamma_1=1.0, gamma_2=1.0, gamma_3=1.0):
    preds, losses, labels, masks = [], [], [], []
    #assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        text_feature, audio_feature, qmask, umask, label, vid = data

        print(f"Unique labels: {torch.unique(label)}")

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
        labels_ = label.view(-1)

        t_kl_lp = t_kl_log_prob.view(-1, t_kl_log_prob.size()[2])
        a_kl_lp = a_kl_log_prob.view(-1, a_kl_log_prob.size()[2])
        all_kl_p = all_kl_prob.view(-1, all_kl_prob.size()[2])

        print(f"Unique labels: {torch.unique(labels_)}")

        loss = gamma_1 * loss_fun(all_lp, labels_, umask) 
        + gamma_2 * (loss_fun(t_lp, labels_, umask) + loss_fun(a_lp, labels_, umask))
        + gamma_3 * (kl_loss(t_kl_lp, all_kl_p, umask) + kl_loss(a_kl_lp, all_kl_p, umask))

        lp_ = all_prob.view(-1, all_prob.size()[2])

        pred = torch.argmax(lp_, 1)
        preds.append(pred.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.contiguous().view(-1).cpu().numpy())

        losses.append(loss.item()*masks[-1].sum())

        if train:
            loss.backward()
            # tensorboard
            #for param in model.named_parameters():
                #writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        
        if preds != []:
            preds = np.concatenate(preds)
            labels = np.concatenate(labels)
            masks = np.concatenate(masks)
        else:
            return float('nan'), float('nan'), [], [], [], float('nan')
        
        avg_loss = round(np.sum(losses) / np.sum(masks), 4)
        avg_acc = round(accuracy_score(labels, preds, sample_weight=masks)*100, 2)   
        avg_fscore = round(f1_score(labels,preds, sample_weight=masks, average='weighted')*100, 2)  
    
        return avg_loss, avg_acc, labels, preds, masks, avg_fscore     

if __name__ == "__main__":

    input_dim = {'text': 768, 'audio': 88, 'speaker':2}
    train_loader, val_loader, test_loader = get_IEMOCAP_loaders(batch_size=16, validRatio=0.2)


    model = Transformer_Based_Model(dataset=train_loader, input_dimension=input_dim, model_dimension=128, temp=1, n_head=8, n_classes=11, dropout=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
    loss_weight = torch.rand(11)
    loss_fun = MaskedNLLLoss(loss_weight)
    kl_loss = MaskedKLDivLoss()
    
    for e in range(150):
        avg_loss, avg_acc, labels, preds, masks, avg_fscore = train_or_eval_model(model=model, loss_fun=loss_fun, kl_loss=kl_loss, dataloader=train_loader, epoch=1, optimizer=optimizer, train=True)

        print(f"Epoch: {e}\navg_loss: {avg_loss}, avg_acc: {avg_acc}, labels: {labels}, preds: {preds}, masks: {masks}, avg_fscore: {avg_fscore}")
