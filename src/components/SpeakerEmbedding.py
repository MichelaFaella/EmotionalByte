import torch.nn as nn
import torch

class SpeakerBiosEmbedding(nn.Module):
    def __init__(self, model_dim, bios_dim=768):
        super().__init__()

        # We assume two speakers
        self.embedding = nn.Embedding(num_embeddings=2, embedding_dim=model_dim)
        self.bios_proj = nn.Linear(bios_dim, model_dim)

    def forward(self, speaker_id, bios_embedding=None):
        """
        :param speaker_id: LongTensor of shape (batch_size, seq_len), with values 0 or 1
        :param bios_embedding: (B, T, 768) tensor or None
        :return:  Tensor
        """
        speaker_vec = self.embedding(speaker_id)

        if bios_embedding is not None:
            projected_bios = self.bios_proj(bios_embedding)
            return speaker_vec + projected_bios
        else:
            return speaker_vec

