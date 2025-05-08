import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd


class IEMOCAPDataset(Dataset):
    def __init__(self, pickle_path, train=True):
        """
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.roberta2, self.roberta3, self.roberta4, \
        self.videoAudio, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open(pickle_path, 'rb'), encoding='latin1')
        """
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.roberta2, self.roberta3, self.roberta4, \
        self.videoAudio, self.trainVid, \
        self.testVid = pickle.load(open(pickle_path, 'rb'), encoding='latin1')


        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<3 else pad_sequence(dat[i], True) if i<5 else dat[i].tolist() for i in dat]

