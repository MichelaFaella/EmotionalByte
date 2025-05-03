import torch
from torch.utils.data import DataLoader
from DataLoader import IEMOCAPDataset

def main():
    # Create dataset for training
    dataset = IEMOCAPDataset("../../data/iemocap_multimodal_features_par.pkl", train=True)

    # Create DataLoader with batch_size=2 to visualize the output 
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)

    # Iterate and print a batch
    for batch_idx, (text, audio, speaker, mask, label, vids) in enumerate(loader):
        print(f"\n=== Batch {batch_idx} ===")
        print("Video IDs:", vids)
        print("Text shape:", text.shape)      # (batch, seq_len_text, dim)
        print("Audio shape:", audio.shape)    # (batch, seq_len_audio, dim)
        print("Speaker shape:", speaker.shape)  # (batch, seq_len_speaker, 2)
        print("Mask:", mask)
        print("Label:", label)


if __name__ == "__main__":
    main()
