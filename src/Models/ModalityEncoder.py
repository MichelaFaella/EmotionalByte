import torch
import torch.nn as nn

from src.Models.PositionalEncoding import PositionalEncoding
from src.Models.SpeakerEmbedding import SpeakerEmbedding


class ModalityEncoder(nn.Module):
    """
    Encodes unimodal input sequences using convolution, positional encoding,
    speaker embeddings, and intra-modal transformer layers.
    """

    def __init__(self, input_dimension, model_dimension):
        """
        :param input_dimension: dictionary with 'text' and 'audio' input dimensions
        :param model_dimension: output embedding dimension for all modalities
        """
        super(ModalityEncoder, self).__init__()
        # 1D convolution to project each modality to the shared model_dimension space
        self.conv_text = nn.Conv1d(input_dimension['text'], model_dimension, kernel_size=1)
        self.conv_audio = nn.Conv1d(input_dimension['audio'], model_dimension, kernel_size=1)

        # Positional and speaker embedding modules
        self.positional_encoder = PositionalEncoding(model_dimension)
        self.speaker_encoder = SpeakerEmbedding(model_dimension)

        # Intra-modal transformer encoder (shared across modalities)
        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model=model_dimension,
            nhead=8,
            dim_feedforward=model_dimension,
            batch_first=True,
        )

    def forward(self, text_feats, audio_feats, speaker_ids):
        """
        :param text_feats: Tensor
        :param audio_feats: Tensor
        :param speaker_ids: Tensor
        :return: Tuple of encoded sequences: (text_repr, audio_repr)
        """

        # Apply 1D convolution to map to model_dimension dimension
        text = self.conv_text(text_feats.transpose(1, 2)).transpose(1,2)
        audio = self.conv_audio(audio_feats.transpose(1, 2)).transpose(1,2)

        # Generate positional and speaker embeddings
        pos = self.positional_encoder(text)
        spk = self.speaker_encoder(speaker_ids)

        # Add embeddings and pass through intra-model transformer
        text_out = self.transformer_encoder(text + pos + spk)
        audio_out = self.transformer_encoder(audio + pos +spk)

        return text_out, audio_out