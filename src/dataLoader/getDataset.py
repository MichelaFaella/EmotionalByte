import torch
from torch.utils.data import DataLoader, random_split
from dataLoader import DataLoader as dl
from collections import Counter

data = "data/iemocap_multimodal_features_6labels.pkl"

def splitDataset(ds, vaildRatio):
    
    size = len(ds)
    val_size = int(vaildRatio * size)
    train_size = size - val_size
    train_dataset, val_dataset = random_split(ds, [train_size, val_size])
    
    return train_dataset, val_dataset

def get_IEMOCAP_loaders(batch_size, validRatio):
    # Create dataset for training
    dataset, test_set = getIEMOCAP()
    train_set, val_set = splitDataset(dataset, validRatio)
    
    # Create DataLoader with batch_size=2 to visualize the output
    tr_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    return tr_loader, val_loader, test_loader

def getIEMOCAP():
    # Create dataset for training
    dataset = dl.IEMOCAPDataset(data, train=True)
    testset = dl.IEMOCAPDataset(data, train=False)

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

    # Show the results
    print("Label counts:", dict(label_counter))

    # Optionally compute normalized class weights
    total = sum(label_counter.values())
    print("total: ", total)
    loss_weights = torch.FloatTensor([1.0 / (label_counter[i] / total) for i in range(6)])
    print("Loss weights:", loss_weights)

    return loss_weights

