import torch
import torch.nn as nn
import torch.nn.functional as F

from components.Transformers import TransformerEncoder
from components.GatedFusion import Unimodal_GatedFusion, Multimodal_GatedFusion, concat


class Transformer_Based_Model(nn.Module):
    """
    Encodes unimodal input sequences using convolution, positional encoding,
    speaker embeddings, and intra-modal transformer layers.
    """

    def __init__(
            self, 
            input_dimension, 
            model_dimension, 
            temp, 
            n_head, 
            n_classes, 
            dropout=0.1,
            train_mode='multi'
    ):
        """
        :param input_dimension: dictionary with 'text','audio' and 'speaker' input dimensions
        :param model_dimension: output embedding dimension for all modalities
        """
        super(Transformer_Based_Model, self).__init__()
        assert(
            train_mode in ['multi', 'text', 'audio']
        ), f'Unexpected train mode: {train_mode}. Expect multi, text or audio instead.'
        self.train_mode = train_mode
        self.n_classes = n_classes
        self.temp = temp        
        
        if self.train_mode is ('multi' or 'text'):
            # 1D convolution to project each modality to the shared model_dimension space
            self.conv_text = nn.Conv1d(input_dimension['text'], model_dimension, kernel_size=1, padding=0, bias=False)
            # Intra-modal Transformers
            self.t_t = TransformerEncoder(dim_model=model_dimension, heads=n_head, layers=1,  dim_ff=model_dimension, dim_speaker=input_dimension['speaker'], dropout=dropout)
            # Unimodal-level Gated Fusion
            self.tt_gate = Unimodal_GatedFusion(model_dimension)
            # Emotion Classifier
            self.t_output_layer = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(model_dimension, n_classes)
            )

        if self.train_mode is ('multi' or 'audio'):
            # 1D convolution to project each modality to the shared model_dimension space
            self.conv_audio = nn.Conv1d(input_dimension['audio'], model_dimension, kernel_size=1, padding=0, bias=False)
            # Intra-modal Transformers
            self.a_a = TransformerEncoder(dim_model=model_dimension, heads=n_head, layers=1,  dim_ff=model_dimension, dim_speaker=input_dimension['speaker'], dropout=dropout)
            # Unimodal-level Gated Fusion
            self.aa_gate = Unimodal_GatedFusion(model_dimension)
            # Emotion Classifier
            self.a_output_layer = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(model_dimension, n_classes)
            )

        if self.train_mode is 'multi':
            # Inter-modal Transformers
            self.t_a = TransformerEncoder(dim_model=model_dimension, heads=n_head, layers=1,  dim_ff=model_dimension, dim_speaker=input_dimension['speaker'], dropout=dropout)
            self.a_t = TransformerEncoder(dim_model=model_dimension, heads=n_head, layers=1,  dim_ff=model_dimension, dim_speaker=input_dimension['speaker'], dropout=dropout)
            # Gated Fusion
            self.concat_t_layer = nn.Linear(model_dimension * 2, model_dimension)
            self.concat_a_layer = nn.Linear(model_dimension * 2, model_dimension)
            # Unimodal-level Gated Fusion
            self.ta_gate = Unimodal_GatedFusion(model_dimension)
            self.at_gate = Unimodal_GatedFusion(model_dimension)
            # Multimodal-level Gated Fusion
            self.multimodal = Multimodal_GatedFusion(model_dimension)
            # Emotion Classifier
            self.all_output_layer = nn.Linear(model_dimension, n_classes)


    def forward(self, speaker_ids, u_mask, **kwargs):
        """
        :param text_feats: Tensor
        :param audio_feats: Tensor
        :param speaker_ids: Tensor
        :param u_mask: Tensor
        :return: Tuple of encoded sequences: (text_repr, audio_repr)
        """
        if self.train_mode is ('multi' or 'text'):
            text_feats = kwargs['text_feats']
            # Apply 1D convolution to map to model_dimension dimension
            text = self.conv_text(text_feats.transpose(1, 2)).transpose(1,2)
            # Intra-modal Transformers
            t_t_transformer_out = self.t_t(query_input=text, key_value_input=text, mask=u_mask, speaker_id=speaker_ids)
            if self.train_mode is 'text':
                Gt = self.tt_gate(t_t_transformer_out)
                t_out = self.t_output_layer(Gt)
                t_log_prob = F.log_softmax(t_out, 2)
                t_kl_log_prob = F.log_softmax(t_out/self.temp, 2)
                return t_log_prob, t_kl_log_prob
        if self.train_mode is ('multi' or 'audio'):
            audio_feats = kwargs['audio_feats']
            # Apply 1D convolution to map to model_dimension dimension
            audio = self.conv_audio(audio_feats.transpose(1, 2)).transpose(1,2)
            # Intra-modal Transformers
            a_a_transformer_out = self.a_a(query_input=audio, key_value_input=audio, mask=u_mask, speaker_id=speaker_ids)
            if self.train_mode is 'audio':
                Ga = self.tt_gate(a_a_transformer_out)
                a_out = self.t_output_layer(Ga)
                a_log_prob = F.log_softmax(a_out, 2)
                a_kl_log_prob = F.log_softmax(a_out/self.temp, 2)
                return a_log_prob, a_kl_log_prob
        
        # Inter-modal Transformers
        t_a_transformer_out = self.t_a(query_input=text, key_value_input=audio, mask=u_mask, speaker_id=speaker_ids)
        a_t_transformer_out = self.a_t(query_input=audio, key_value_input=text, mask=u_mask, speaker_id=speaker_ids)

        # Unimodal-level Gated Fusion
        Gt = concat(
            self.concat_t_layer,
            self.tt_gate(t_t_transformer_out),
            self.ta_gate(t_a_transformer_out)
        )
        Ga = concat(
            self.concat_a_layer,
            self.aa_gate(a_a_transformer_out),
            self.at_gate(a_t_transformer_out)
        )

        # Multimodal-level Gated Fusion
        multi_output = self.multimodal(Gt, Ga)
        
        # Emotion Classifier
        t_out = self.t_output_layer(Gt)
        a_out = self.a_output_layer(Ga)

        all_out = self.all_output_layer(multi_output)

        # Softmax on all the outputs
        t_log_prob = F.log_softmax(t_out, 2)
        a_log_prob = F.log_softmax(a_out, 2)

        all_log_prob = F.log_softmax(all_out, 2)
        all_prob = F.softmax(all_out, 2) # Used for predictions
        
        # KL divergence
        t_kl_log_prob = F.log_softmax(t_out /self.temp, 2)
        a_kl_log_prob = F.log_softmax(a_out /self.temp, 2)

        all_kl_prob = F.softmax(all_out /self.temp, 2)

        return t_log_prob, a_log_prob, all_log_prob, all_prob, t_kl_log_prob, a_kl_log_prob, all_kl_prob