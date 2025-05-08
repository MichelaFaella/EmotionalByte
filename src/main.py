import torch
from torch.utils.data import DataLoader

from DataLoader import DataLoaderOld as dlo
from DataLoader import DataLoaderNuovo as dln
from DataLoader import DataLoaderSDT as SDT


from Models import provaModel

data_par = "../data/iemocap_multimodal_features_par.pkl"
data_SDT = "../data/IEMOCAP_features.pkl"
data_dialog = "../data/iemocap_dialog_level.pkl"
data = "../data/iemocap_multimodal_features.pkl"


def get_IEMOCAP_loaders(batch_size, Dataset):
    # Create dataset for training
    if Dataset == 0:
        dataset = dlo.IEMOCAPDataset("../data/iemocap_multimodal_features_par.pkl", train=True)
        # Create DataLoader with batch_size=2 to visualize the output
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    elif Dataset == 1:
        datasets = SDT.IEMOCAPDataset(pickle_path=data, train=True)
        loader = DataLoader(datasets, batch_size=batch_size, shuffle=True, collate_fn=datasets.collate_fn)
    else:
        dataset = dln.IEMOCAPDataset(pickle_path="../data/iemocap_dialog_level.pkl", train=True)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    return loader

def train_or_eval_model(model, loss_fun, dataloader, epoch, optimizer=None, train=False ):
    #assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        text_feature, audio_feature, qmask, umask, labels, dialog_id = data
        speaker_id = qmask.squeeze(-1)
        #qmask = qmask.permute(1, 0, 2) # For transformers compatibility

        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))] # Compute the real length of a sequence

        t_t_transformer_out, a_a_transformer_out, t_a_transformer_out, a_t_transformer_out = model(text_feature, audio_feature, speaker_id, umask)
        print(f"t_t_transformer_out: {t_t_transformer_out}\n a_a_transformer_out: {a_a_transformer_out} \n t_a_transformer_out: {t_a_transformer_out} \n a_t_transformer_out: {a_t_transformer_out}")


def main():
    DatasetLoader = 1

    loader = get_IEMOCAP_loaders(batch_size=32, Dataset=DatasetLoader)

    if DatasetLoader == 2:
        # Take one batch and test forward pass
        for batch_idx, (text, audio, speaker, mask, label, vids) in enumerate(loader):
            print(f"\n=== Batch {batch_idx} ===")
            print("Video IDs:", vids)
            print("Text shape:", text.shape)      # (batch, seq_len_text, dim)
            print("Audio shape:", audio.shape)    # (batch, seq_len_audio, dim)
            print("Speaker shape:", speaker.shape)  # (batch, seq_len_speaker, 2)
            print("Mask shape:", mask.shape)
            print("Label shape:", label.shape)
    elif DatasetLoader == 1:
        for batch_idx, (text, audio, speaker, mask, label, vids) in enumerate(loader):
            print(f"\n=== Batch {batch_idx} ===")
            print("Video IDs:", vids)
            print("Text shape:", text.shape)      # (seq_len_text, batch, dim)
            print("Audio shape:", audio.shape)    # (batch, seq_len_audio, dim)
            print("Speaker (qmask) shape:", speaker.shape)  # (seq_len_speaker, batch, 2)
            print("(u)Mask shape:", mask.shape) # (batch, seq_len)
            print("Label shape:", label.shape)  # (batch, seq_len)
    else:
        for batch_idx, (text, audio, qmask, umask, labels, dialog_ids) in enumerate(loader):
            #text, audio, qmask, umask, labels, dialog_ids = batch

            print("Text shape:", text.shape)       # [B, T_max, D_text]
            #print("Audio shape:", audio.shape)     # [B, T_max, D_audio]
            #print("Qmask shape:", qmask.shape)     # [B, T_max, 2]
            #print("Sperkers:", qmask)
            #print("Umask shape:", umask.shape)     # [B, T_max]
            #print("Labels shape:", labels.shape)   # [B, T_max]
            #print("Dialog IDs:", dialog_ids)


        #input_dim = {'text': 768, 'audio': 88, 'speaker':2}

        #model = provaModel.Transformer_Based_Model( dataset=loader,  input_dimension=input_dim, model_dimension=128, n_head=8, n_classes=10, dropout=0.1)
        #train_or_eval_model(model=model, loss_fun=None, dataloader=loader, epoch=1, optimizer=None, train=False)




if __name__ == "__main__":
    main()

