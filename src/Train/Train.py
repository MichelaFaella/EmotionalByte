from itertools import product

import torch
import numpy as np
from tensorboardX import SummaryWriter
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score

from dataLoader.getDataset import get_IEMOCAP_loaders, lossWeightsNormalized, getDataName, getDimension, changeDimension
from components.model import Transformer_Based_Model
from Plot.Plot import log_confusion_matrix
from Train.Losses import *


def train_or_eval_model(model, loss_fun, kl_loss, dataloader, epoch, optimizer=None, train=False, writer=None,
                        gamma_1=1.0, gamma_2=1.0, gamma_3=1.0):
    preds, losses, labels, masks = [], [], [], []
    # assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        text_feature, audio_feature, qmask, umask, label, vid = data

        text_feature, audio_feature, qmask, umask, label = changeDimension(text_feature, audio_feature, label, umask, qmask)
                                                                                                                                                                                                                                                                    
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in
                   range(len(umask))]  # Compute the real length of a sequence

        if model.modality=='text':
            t_log_prob = model(qmask, umask, text_feats=text_feature)
        elif model.modality=='audio':
            a_log_prob = model(qmask, umask, audio_feats=audio_feature)
        elif model.modality=='multi':
            t_log_prob, a_log_prob, all_log_prob, all_prob, t_kl_log_prob, a_kl_log_prob, all_kl_prob = model(qmask,
                                                                                                              umask,
                                                                                                              text_feats=text_feature,
                                                                                                              audio_feats=audio_feature)

        labels_ = label.view(-1)
        if model.modality in ['multi', 'text']:
            t_lp = t_log_prob.view(-1, t_log_prob.size()[2])
            loss_ce_t = loss_fun(t_lp, labels_, umask)
            if model.modality=='text':
                loss = gamma_1 * loss_ce_t
                lp_ = t_lp      # Get probabilities for each class
                # Define losses to return
                losses_ = {
                    'loss_ce_t': loss_ce_t,
                    'loss': loss
                }
        if model.modality in ['multi', 'audio']:
            a_lp = a_log_prob.view(-1, a_log_prob.size()[2])
            loss_ce_a = loss_fun(a_lp, labels_, umask)
            if model.modality == 'audio':
                loss = gamma_1 * loss_ce_a
                lp_ = a_lp      # Get probabilities for each class
                # Define losses to return
                losses_ = {
                    'loss_ce_a': loss_ce_a,
                    'loss': loss
                }
        if model.modality == 'multi':
            all_lp = all_log_prob.view(-1, all_log_prob.size()[2])               
            all_kl_p = all_kl_prob.view(-1, all_kl_prob.size()[2])
            loss_task = loss_fun(all_lp, labels_, umask)
            t_kl_lp = t_kl_log_prob.view(-1, t_kl_log_prob.size()[2])
            loss_kl_t = kl_loss(t_kl_lp, all_kl_p, umask)
            a_kl_lp = a_kl_log_prob.view(-1, a_kl_log_prob.size()[2])
            loss_kl_a = kl_loss(a_kl_lp, all_kl_p, umask)
            loss = gamma_1 * loss_task + gamma_2 * (loss_ce_t + loss_ce_a) + gamma_3 * (loss_kl_t + loss_kl_a)
            lp_ = all_prob.view(-1, all_prob.size()[2])      # Get probabilities for each class
            # Define losses to return
            losses_ = {
                'loss_task': loss_task,
                'loss_ce_t': loss_ce_t,
                'loss_kl_t': loss_kl_t,
                'loss_ce_a': loss_ce_a,
                'loss_kl_a': loss_kl_a,
                'loss': loss
            }

        pred = torch.argmax(lp_, 1)
        preds.append(pred.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.contiguous().view(-1).cpu().numpy())

        losses.append(loss.item() * masks[-1].sum())

        if train:
            loss.backward()
            # tensorboard
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(name + '_grad', param.grad, epoch)
                writer.add_histogram(name + '_weights', param, epoch)
            optimizer.step()

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan')

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_acc = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted', zero_division=0) * 100, 2)

    return avg_loss, avg_acc, labels, preds, masks, avg_fscore, losses_

def run_phase(model, loss_fun, kl_loss, dataloader, epoch, writer, optimizer, train, gamma_1, gamma_2, gamma_3):
    return train_or_eval_model(
        model=model,
        loss_fun=loss_fun,
        kl_loss=kl_loss,
        dataloader=dataloader,
        epoch=epoch,
        writer=writer,
        optimizer=optimizer if train else None,
        train=train,
        gamma_1=gamma_1,
        gamma_2=gamma_2,
        gamma_3=gamma_3,
    )

def log_tensorboard(writer, phase, acc, fscore, loss, losses, epoch):
    if writer:
        writer.add_scalar(f'{phase}: accuracy', acc, epoch)
        writer.add_scalar(f'{phase}: fscore', fscore, epoch)
        writer.add_scalar(f'{phase}: loss', loss, epoch)
        for key in losses:
            writer.add_scalar(f'{phase}: {key}', losses[key].item(), epoch)

def TrainSDT(temp=1.0, gamma_1=0.1, gamma_2=0.1, gamma_3=0.1, run_name="exp1", return_val_score=False, **kwargs):
    torch.manual_seed(42)

    model_dimension = kwargs.get("model_dimension", 32)
    n_head = kwargs.get("n_head", 8)
    n_epochs = kwargs.get("n_epochs", 70)
    dropout = kwargs.get("dropout", 0.01)
    lr = kwargs.get("lr", 0.0001)
    weight_decay = kwargs.get("weight_decay", 0.0001)
    batch_size = kwargs.get("batch_size", 16)
    modality = kwargs.get("modality", 'multi')

    train_loader, val_loader, test_loader, design_loader = get_IEMOCAP_loaders(batch_size=batch_size, validRatio=0.2)

    # Get a single batch from the training loader to determine input dimensions dynamically
    sample_batch = next(iter(train_loader))
    # Unpack only the necessary components from the batch (ignore others with underscores)
    text_feature, audio_feature, _, _, _, _ = sample_batch
    # Compute text and audio feature dimensions using a helper function
    text_dim, audio_dim = getDimension(text_feature, audio_feature)
    input_dim = {'text': text_dim, 'audio': audio_dim, 'speaker': 2}
    n_classes = 6

    print(f"Model Dimension:{model_dimension}, Epochs:{n_epochs}, Learning Rate:{lr}, Weight Decay:{weight_decay}, Modality: {modality}")
    tensorboard = kwargs.get("tensorboard", True)
    writer = SummaryWriter(log_dir=f"runs/{run_name}") if tensorboard else None
    if writer:
        hparams_text = (
            f"Model Dim: {model_dimension}, Dropout: {dropout}, LR: {lr}, "
            f"Weight Decay: {weight_decay}, Batch Size: {batch_size}, "
            f"Gamma: ({gamma_1}, {gamma_2}, {gamma_3}), Modality: {modality}"
        )
        writer.add_text("Hyperparameters", hparams_text, 0)

    model = Transformer_Based_Model(
        dataset=train_loader,
        input_dimension=input_dim,
        model_dimension=model_dimension,
        temp=temp,
        n_head=n_head,
        n_classes=n_classes,
        dropout=dropout,
        modality=modality
    )

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fun = MaskedNLLLoss(lossWeightsNormalized())
    kl_loss = MaskedKLDivLoss(tau=temp)

    logs_train = {key: [] for key in ['loss_task', 'loss_ce_t', 'loss_ce_a', 'loss_kl_t', 'loss_kl_a', 'loss']}
    logs_eval = {'loss': [], 'acc': [], 'fscore': []}

    train_dl, eval_dl = (train_loader, val_loader) if return_val_score else (design_loader, test_loader)

    file_name = getDataName()
    best_val_fscore = 0

    for e in range(n_epochs):
        train_metrics = run_phase(model, loss_fun, kl_loss, train_dl, e, writer, optimizer, True, gamma_1, gamma_2, gamma_3)
        eval_metrics = run_phase(model, loss_fun, kl_loss, eval_dl, e, writer, None, False, gamma_1, gamma_2, gamma_3)

        train_loss, train_acc, train_labels, train_preds, _, train_fscore, losses = train_metrics
        eval_loss, eval_acc, eval_labels, eval_preds, _, eval_fscore, *_ = eval_metrics

        log_tensorboard(writer, "train", train_acc, train_fscore, train_loss, losses, e)
        log_tensorboard(writer, "val" if return_val_score else "test", eval_acc, eval_fscore, eval_loss, {}, e)

        logs_eval['loss'].append(eval_loss)
        logs_eval['acc'].append(eval_acc)
        logs_eval['fscore'].append(eval_fscore)

        for key in logs_train:
            if key in losses:
                logs_train[key].append(losses[key].item())
            else:
                logs_train[key].append(0.0)  # Handle missing keys appropriately

        if not return_val_score:
            print(f"Epoch: {e + 1}/{n_epochs}")
            print(f"Train -> loss: {train_loss}, acc: {train_acc}, fscore: {train_fscore}")
            print(f"Test  -> loss: {eval_loss}, acc: {eval_acc}, fscore: {eval_fscore}")

        if return_val_score and eval_fscore > best_val_fscore:
            best_val_fscore = eval_fscore

        if writer:
            # Log confusion matrix for eval phase
            log_confusion_matrix(writer, eval_labels, eval_preds, e, "val" if return_val_score else "test")

    if writer:
        # Log confusion matrix
        log_confusion_matrix(writer, eval_labels, eval_preds, e, "val" if return_val_score else "test")
        writer.close()

    if return_val_score:
        return best_val_fscore
