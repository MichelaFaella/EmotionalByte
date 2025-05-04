import torch
from torch.utils.data import DataLoader
from DataLoader import DataLoader as dl
from Models import ModalityEncoder as me

def main():
    # Create dataset for training
    dataset = dl.IEMOCAPDataset("../data/iemocap_multimodal_features_par.pkl", train=True)

    # Create DataLoader with batch_size=2 to visualize the output 
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=dataset.collate_fn)
    """
    # Instantiate the model
    input_dims = {'text': 768, 'audio': 88}
    model_dim = 128  # Puoi scegliere un valore arbitrario
    model = me.ModalityEncoder(input_dims, model_dim)

    # Set the model to eval mode (o .train() se stai testando il training)
    model.train()
    """

    # Take one batch and test forward pass
    for batch_idx, (text, audio, speaker, mask, label, vids) in enumerate(loader):
        print(f"\n=== Batch {batch_idx} ===")
        print("Video IDs:", vids)
        print("Text shape:", text.shape)      # (batch, seq_len_text, dim)
        print("Audio shape:", audio.shape)    # (batch, seq_len_audio, dim)
        print("Speaker shape:", speaker.shape)  # (batch, seq_len_speaker, 2)
        print("Mask shape:", mask.shape)
        print("Label shape:", label.shape)


        """
        # Forward pass
        with torch.no_grad():
            text_out, audio_out = model(text, audio, speaker)

        print("Text output shape:", text_out.shape)
        print("Audio output shape:", audio_out.shape)
        break  # Solo il primo batch per ora
        """



if __name__ == "__main__":
    main()
