import torch.nn as nn

class SpeakerEmbedding(nn.Module):
    """
    Implements learnable embeddings for speakers (e.g., 'M' or 'F').
    """

    def __init__(self, model_dim):
        """
        :param _dim: size of the speaker embedding vectors
        """

        super().__init__()

        # We assume two speakers
        self.embedding = nn.Embedding(num_embeddings=2, embedding_dim=model_dim)

    def forward(self, speaker_id):
        """
        :param speaker_id: LongTensor of shape (batch_size, seq_len), with values 0 or 1
        :return:  Tensor
        """

        return self.embedding(speaker_id)
