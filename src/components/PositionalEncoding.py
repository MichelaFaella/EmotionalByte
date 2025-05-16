import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for sequential inputs.
    """

    def __init__(self, model_dimension, max_len=500):
        """
        :param model_dimension: dimension of the model (output features)
        :param max_len: maximum sequence length supported
        """

        super().__init__()

        # Create positional encoding matrix
        positional_encoding = torch.zeros(max_len, model_dimension)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dimension, 2) * (-math.log(10000.0) / model_dimension))

        # Apply sine to even indices, cosine to odd indices
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        # Shape: (1, max_len, model_dimension) to broadcast over batch dimension
        self.register_buffer('positional_encoding', positional_encoding.unsqueeze(0))

    def forward(self, x, mask):
        """
        :param x: input tensor
        :return: input tensor with positional encoding added
        """
        x = x.masked_fill(mask.unsqueeze(-1) == 0, 0)

        return x + self.positional_encoding[:, :x.size(1)]
