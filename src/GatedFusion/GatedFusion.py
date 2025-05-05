import torch
import torch.nn as nn

class Unimodal_GatedFusion(nn.Module):

  def __init__(self, hidden_size):
    super(Unimodal_GatedFusion, self).__init__()
    self.fc = nn.Linear(hidden_size, hidden_size, bias=False)

  def forward(self, H):
    g = torch.sigmoid(self.fc(H))
    return H*g
  
class Multimodal_GatedFusion(nn.Module):
  
  def __init__(self, hidden_size):
    super(Multimodal_GatedFusion, self).__init__()
    self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
    self.softmax = nn.Softmax(dim=-2)

  def forward(self, Gt, Ga):
    G = torch.cat([Gt.unsqueeze(-2), Ga.unsqueeze(-2)], dim=-2)
    G_fc = torch.cat([self.fc(Gt).unsqueeze(-2), self.fc(Ga).unsqueeze(-2)], dim=-2)
    G_softmax = self.softmax(G_fc)
    G_model = G_softmax*G
    return torch.sum(G_model, dim=-2, keepdim=False)
  
def concat(features_layer: nn.Linear, Gmm, Gnm):
  return features_layer(torch.cat([Gmm, Gnm], dim=-1))