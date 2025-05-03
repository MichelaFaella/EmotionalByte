import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from abc import ABC, abstractmethod


class MultimodalDataset(Dataset, ABC):
    """
    Abstract base class for multimodal datasets.
    """

    def __init__(self, pickle_path, train=True):
        self.keys = None
        self.train = train
        self.load_pickle(pickle_path)
        self.set_split_keys()

    @abstractmethod
    def load_pickle(self, path):
        """Loads data structures from the pickle file."""
        pass

    @abstractmethod
    def set_split_keys(self):
        """Sets the IDs of training or testing samples."""
        pass

    @abstractmethod
    def __getitem__(self, index):
        """Retrieves a single sample from the dataset."""
        pass

    def __len__(self):
        return len(self.keys)

    def collate_fn(self, batch):
        dat = pd.DataFrame(batch)
        return [
            pad_sequence(dat[i].tolist(), batch_first=True) if i < 5 else dat[i].tolist()
            for i in dat
        ]


class IEMOCAPDataset(MultimodalDataset):
    def load_pickle(self, path):
        (
            self.videoIDs,
            self.videoSpeakers,
            self.videoLabels,
            self.videoText,
            self.roberta2,
            self.roberta3,
            self.roberta4,
            self.videoAudio,
            self.trainVid,
            self.testVid,
        ) = pickle.load(open(path, 'rb'))

    def set_split_keys(self):
        self.keys = self.trainVid if self.train else self.testVid

    def __getitem__(self, index):
        vid = self.keys[index]
        text_feat = torch.FloatTensor(self.videoText[vid])
        audio_feat = torch.FloatTensor(self.videoAudio[vid])
        speaker = torch.FloatTensor([1, 0] if self.videoSpeakers[vid] == 'M' else [0, 1])
        mask = torch.FloatTensor([1])
        label = torch.LongTensor([self.videoLabels[vid]])
        return text_feat, audio_feat, speaker, mask, label, vid