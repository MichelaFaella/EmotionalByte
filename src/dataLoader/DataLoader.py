import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd


class IEMOCAPDataset(Dataset):
    def __init__(self, pickle_path, speaker_bios_path=None, train=True):
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

        # use BIOS information
        self.use_bios = speaker_bios_path is not None
        self.speaker_bios = torch.load(speaker_bios_path) if self.use_bios else {}

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):

        vid = self.keys[index]

        videoTextTensor = torch.FloatTensor(self.videoText[vid])
        audioTensor = torch.FloatTensor(self.videoAudio[vid])
        speakerTensor = torch.LongTensor(self.videoSpeakers[vid])

        maskTensor = torch.FloatTensor([1]*len(self.videoLabels[vid]))

        labelTensor = torch.LongTensor(self.videoLabels[vid])

        # use BIOS information
        bios_tensor = None
        if self.use_bios:
            bios_embedding = []
            for spk in self.videoSpeakers[vid]:
                if isinstance(spk, list) and isinstance(spk[0], list):  # [[0, 1]] or [[1, 0]]
                    spk = spk[0]
                spk_label = decode_speaker(spk)
                bios = self.speaker_bios.get((vid, spk_label), torch.zeros(768))
                bios_embedding.append(bios)
                
            bios_tensor = torch.stack(bios_embedding)

        return videoTextTensor, audioTensor, speakerTensor, maskTensor, labelTensor, bios_tensor, vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        padded = [pad_sequence(dat[i]) if i<3 else pad_sequence(dat[i], True) if i<5 else dat[i].tolist() for i in dat]
        bios_list = dat[5].tolist()
        vids = dat[6].tolist()
        
        # pad bios if present
        if bios_list[0] is not None:
            bios_padded = pad_sequence(bios_list)  # [seq_len, batch, 768]
        else:
            bios_padded = None
        
        return padded[:5] + [bios_padded, vids]
    

def decode_speaker(one_hot):
    if one_hot == [1, 0]:
        return 'M'
    elif one_hot == [0, 1]:
        return 'F'
    else:
        raise ValueError(f"Unknown one-hot speaker encoding: {one_hot}")