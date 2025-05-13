import torch
from torch.utils.data import DataLoader, random_split
from dataLoader import DataLoader as dl

data = "./data/iemocap_multimodal_features.pkl"

def splitDataset(ds, vaildRatio):
    
    size = len(ds)
    val_size = int(vaildRatio * size)
    train_size = size - val_size
    train_dataset, val_dataset = random_split(ds, [train_size, val_size])
    
    return train_dataset, val_dataset

def get_IEMOCAP_loaders(batch_size, validRatio):
    # Create dataset for training
    dataset = dl.IEMOCAPDataset(data, train=True)
    train_set, val_set = splitDataset(dataset, validRatio)
    
    # Create DataLoader with batch_size=2 to visualize the output
    tr_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    
    test_set = dl.IEMOCAPDataset(data, train=False)
    
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    return tr_loader, val_loader, test_loader
