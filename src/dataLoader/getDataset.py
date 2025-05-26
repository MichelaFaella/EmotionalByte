import math

import torch
from torch.utils.data import DataLoader, random_split
from dataLoader import DataLoader as dl
from collections import Counter
from pathlib import Path

data = "data/IEMOCAP/iemocap_multimodal_features_6_labels_roberta_fine_tuned_eGeMAPSv02.pkl"

def splitDataset(ds, vaildRatio):
    
    size = len(ds)
    val_size = int(vaildRatio * size)
    train_size = size - val_size
    train_dataset, val_dataset = random_split(ds, [train_size, val_size])
    
    return train_dataset, val_dataset

def get_IEMOCAP_loaders(batch_size, validRatio, bios=False):
    # Create dataset for training
    dataset, test_set = getIEMOCAP(bios)
    train_set, val_set = splitDataset(dataset, validRatio)

    tr_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    design_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    return tr_loader, val_loader, test_loader,design_loader

def getIEMOCAP(bios=False):
    # Create dataset for training 
    if bios == False: 
        dataset = dl.IEMOCAPDataset(data, train=True)
        testset = dl.IEMOCAPDataset(data, train=False)
    # Create dataset for training with biosERC
    else: 
        dataset = dl.IEMOCAPDataset(data, speaker_bios_path="src/biosERC/speaker_bios.pt", train=True)
        testset = dl.IEMOCAPDataset(data, speaker_bios_path="src/biosERC/speaker_bios.pt", train=False)

    return dataset, testset

def lossWeights():
    dataset, _ = getIEMOCAP()

    # Initialize a counter
    label_counter = Counter()
    for vid in dataset.trainVid:
        labels = dataset.videoLabels[vid]
        
        # Flatten if labels are list of lists
        for label in labels:
            if isinstance(label, list):
                label_counter.update(label)
            else:
                label_counter.update([label])

    # Optionally compute normalized class weights
    total = sum(label_counter.values())
    #loss_weights = torch.FloatTensor([1.0 / (label_counter[i] / total) for i in range(6)])
    loss_weights = torch.FloatTensor([1.0 / math.sqrt(label_counter[i] / total) for i in range(6)])

    return loss_weights

import torch
from collections import Counter
from math import isclose

def lossWeightsNormalized():
    dataset, _ = getIEMOCAP()

    # Conta le etichette nel training set
    label_counter = Counter()
    for vid in dataset.trainVid:
        labels = dataset.videoLabels[vid]
        for label in labels:
            if isinstance(label, list):
                label_counter.update(label)
            else:
                label_counter.update([label])

    total = sum(label_counter.values())

    # Calcola i pesi come inverso della frequenza relativa
    raw_weights = [1.0 / (label_counter[i] / total) for i in range(6)]

    # Normalizza i pesi in modo che la somma sia 1
    sum_weights = sum(raw_weights)
    normalized_weights = [w / sum_weights for w in raw_weights]

    # Verifica che la somma sia davvero 1 (debug)
    assert isclose(sum(normalized_weights), 1.0, rel_tol=1e-3), "I pesi non sono normalizzati correttamente"

    return torch.FloatTensor(normalized_weights)



def getDataName():
    data_name = Path(data).stem
    return data_name


def changeDimension(text_feature, audio_feature, label, umask, qmask):
    
    text_feature = text_feature.permute(0, 1, 3, 2).squeeze(-1)  # (seq_len_text, batch, dim)
    audio_feature = audio_feature.permute(0, 1, 3, 2).squeeze(-1)  # (seq_len_audio, batch, dim)
    label = label.squeeze(-1)  # (batch, seq_len)
    umask = umask.permute(1, 0)  # (seq_len, batch)
    qmask = qmask.permute(0, 1, 3, 2)[:, :, 0, 0]  # (seq_len_spk, batch, dim)

    return text_feature, audio_feature, qmask, umask, label

def getDimension(text_feature, audio_feature):
    
    text_dim = text_feature.shape[-1]
    audio_dim = audio_feature.shape[-1]

    return text_dim, audio_dim