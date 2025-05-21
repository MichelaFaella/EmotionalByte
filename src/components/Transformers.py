import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.components.SpeakerEmbedding import SpeakerEmbedding
from src.components.PositionalEncoding import PositionalEncoding

class PositionwiseFeedForward(nn.Module):

    def __init__(self, dim_model, dim_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(dim_model, dim_ff)
        self.w_2 = nn.Linear(dim_ff, dim_model)
        self.layer_norm = nn.LayerNorm(dim_model, eps=1e-6)
        self.actvfun = F.gelu
        self.dropout_1 = nn.Dropout(dropout)
        #self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        #interim = self.dropout_1(self.actvfun(self.w_1(self.layer_norm(x))))
        #output = self.dropout_2(self.w_2(interim))
        output = self.w_2(self.dropout_1(self.actvfun(self.w_1(self.layer_norm(x)))))
        return output + x



class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim_model, heads, dim_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(dim_model, heads, dropout=dropout, batch_first=True)
        self.feed_forward = PositionwiseFeedForward(dim_model, dim_ff, dropout)
        self.layer_norm = nn.LayerNorm(dim_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_a, input_b, mask):
        if input_a.equal(input_b):  # Intra-modality
            query = key = value = input_b
        else:  # Inter-modality
            query = input_b
            key = value = input_a

        query = self.layer_norm(query) # Query normalization
        context,_ = self.self_attn(query,key,value,key_padding_mask=mask)

        output = self.dropout(context) + input_b # residual connection
        return self.feed_forward(output)

class TransformerEncoder(nn.Module):
    def __init__(self, dim_model, heads, layers,  dim_ff, dim_speaker, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.dim_model = dim_model
        self.layers = layers
        self.pos_emb = PositionalEncoding(dim_model)
        self.speaker_emb = SpeakerEmbedding(dim_model)
        self.transformer_layers = nn.ModuleList(
            [TransformerEncoderLayer(dim_model, heads, dim_ff, dropout)
             for _ in range(layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_input, key_value_input, mask, speaker_id):
        is_self_attention = query_input.equal(key_value_input)

        key_padding_mask = ~mask.bool()

        pos_emb = self.pos_emb(key_value_input, key_padding_mask)
        speaker_emb = self.speaker_emb(speaker_id)


        # Application of positional and speaker embeddings
        key_value_input = pos_emb + speaker_emb + key_value_input

        key_value_input = self.dropout(key_value_input)

        if not is_self_attention:
            query_input = self.pos_emb(query_input, mask) + self.speaker_emb(speaker_id) + query_input
            query_input = self.dropout(query_input)

        # Compute attention on layers
        for layer_index in range(self.layers):
            if is_self_attention:
                key_value_input = self.transformer_layers[layer_index](key_value_input, key_value_input, key_padding_mask)
            else:
                key_value_input = self.transformer_layers[layer_index](query_input, key_value_input, key_padding_mask)
        return key_value_input