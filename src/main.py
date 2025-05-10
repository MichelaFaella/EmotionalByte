import torch
from torch.utils.data import DataLoader

from DataLoader import DataLoader as dl

from Models import provaModel

data = "../data/iemocap_multimodal_features.pkl"


def get_IEMOCAP_loaders(batch_size):
    # Create dataset for training
    dataset = dl.IEMOCAPDataset(data, train=True)
    # Create DataLoader with batch_size=2 to visualize the output
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    return loader


def train_or_eval_model(model, loss_fun, dataloader, epoch, optimizer=None, train=False ):
    #assert not train or optimizer != None
    """
    if train:
        model.train()
    else:
        model.eval()
    """

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        text_feature, audio_feature, qmask, umask, labels, vid = data


        text_feature = text_feature.permute(0, 1, 3, 2).squeeze(-1)
        audio_feature = audio_feature.permute(0, 1, 3, 2).squeeze(-1)
        labels = labels.squeeze(-1)
        umask = umask.permute(1, 0)
        qmask = qmask.permute(0, 1, 3, 2)[:, :, 0, 0]

        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))] # Compute the real length of a sequence

        t_t_transformer_out, a_a_transformer_out, t_a_transformer_out, a_t_transformer_out = model(text_feature, audio_feature, qmask, umask)
        print(f"t_t_transformer_out: {t_t_transformer_out}\n a_a_transformer_out: {a_a_transformer_out} \n t_a_transformer_out: {t_a_transformer_out} \n a_t_transformer_out: {a_t_transformer_out}")


def main():
    loader = get_IEMOCAP_loaders(batch_size=32)

    """
    for batch_idx, (text, audio, speaker, mask, label, vids) in enumerate(loader):
        print(f"\n=== Batch {batch_idx} ===")

        text = text.permute(0, 1, 3, 2).squeeze(-1)
        audio = audio.permute(0, 1, 3, 2).squeeze(-1)
        label = label.squeeze(-1)
        speaker = speaker.permute(0, 1, 3, 2).squeeze(-1)

        print("Video IDs:", vids)
        print("Text shape:", text.shape)  # (seq_len_text, batch, dim)
        print("Audio shape:", audio.shape)  # (seq_len_audio, batch, dim)
        print("Speaker shape:", speaker.shape)  # (seq_len_speaker, batch, 2)
        print("Mask shape:", mask.shape)  # (batch, seq_len)
        print("Label shape:", label.shape)  # (batch, seq_len)
    """



    input_dim = {'text': 768, 'audio': 88, 'speaker':2}

    model = provaModel.Transformer_Based_Model( dataset=loader,  input_dimension=input_dim, model_dimension=128, n_head=8, n_classes=10, dropout=0.1)
    train_or_eval_model(model=model, loss_fun=None, dataloader=loader, epoch=1, optimizer=None, train=False)




if __name__ == "__main__":
    main()

