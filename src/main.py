import torch
from torch.utils.data import DataLoader

from DataLoader import DataLoader as dl

from Models import provaModel
from training.getDataset import get_IEMOCAP_loaders
from training.Train import train_or_eval_model

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

