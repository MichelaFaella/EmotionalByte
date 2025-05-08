import torch
import torch.nn as nn

from .Transformers import TransformerEncoder


class Transformer_Based_Model(nn.Module):
    """
    Encodes unimodal input sequences using convolution, positional encoding,
    speaker embeddings, and intra-modal transformer layers.
    """

    def __init__(self, dataset,  input_dimension, model_dimension, n_head, n_classes, dropout=0.1):
        """
        :param input_dimension: dictionary with 'text','audio' and 'speaker' input dimensions
        :param model_dimension: output embedding dimension for all modalities
        """
        super(Transformer_Based_Model, self).__init__()
        self.n_classes = n_classes

        # 1D convolution to project each modality to the shared model_dimension space
        self.conv_text = nn.Conv1d(input_dimension['text'], model_dimension, kernel_size=1, padding=0, bias=False)
        self.conv_audio = nn.Conv1d(input_dimension['audio'], model_dimension, kernel_size=1, padding=0, bias=False)

        # Intra-modal Transformers
        self.t_t = TransformerEncoder(dim_model=model_dimension, heads=n_head, layers=1,  dim_ff=model_dimension, dim_speaker=input_dimension['speaker'], dropout=dropout)
        self.a_a = TransformerEncoder(dim_model=model_dimension, heads=n_head, layers=1,  dim_ff=model_dimension, dim_speaker=input_dimension['speaker'], dropout=dropout)

        # Inter-modal Transformers
        self.t_a = TransformerEncoder(dim_model=model_dimension, heads=n_head, layers=1,  dim_ff=model_dimension, dim_speaker=input_dimension['speaker'], dropout=dropout)
        self.a_t = TransformerEncoder(dim_model=model_dimension, heads=n_head, layers=1,  dim_ff=model_dimension, dim_speaker=input_dimension['speaker'], dropout=dropout)

        # Unimodal-level Gated Fusion

        # Multimodal-level Gated Fusion

        # Emotion Classifier
        self.t_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dimension, n_classes)
            )
        self.a_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dimension, n_classes)
            )

        self.all_output_layer = nn.Linear(model_dimension, n_classes)


    def forward(self, text_feats, audio_feats, speaker_ids, u_mask):
        """
        :param text_feats: Tensor
        :param audio_feats: Tensor
        :param speaker_ids: Tensor
        :param u_mask: Tensor
        :return: Tuple of encoded sequences: (text_repr, audio_repr)
        """

        # Apply 1D convolution to map to model_dimension dimension
        text = self.conv_text(text_feats.transpose(1, 2)).transpose(1,2)
        audio = self.conv_audio(audio_feats.transpose(1, 2)).transpose(1,2)

        # Intra-modal Transformers
        t_t_transformer_out = self.t_t(query_input=text, key_value_input=text, mask=u_mask, speaker_id=speaker_ids)
        a_a_transformer_out = self.a_a(query_input=audio, key_value_input=audio, mask=u_mask, speaker_id=speaker_ids)

        # Inter-modal Transformers
        t_a_transformer_out = self.t_a(query_input=text, key_value_input=audio, mask=u_mask, speaker_id=speaker_ids)
        a_t_transformer_out = self.a_t(query_input=audio, key_value_input=text, mask=u_mask, speaker_id=speaker_ids)

        # Unimodal-level Gated Fusion

        # Multimodal-level Gated Fusion

        # Emotion Classifier



        return t_t_transformer_out, a_a_transformer_out, t_a_transformer_out, a_t_transformer_out