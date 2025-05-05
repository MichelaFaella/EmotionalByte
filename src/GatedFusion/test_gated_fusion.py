import unittest
import torch
import torch.nn as nn

from GatedFusion import Unimodal_GatedFusion, Multimodal_GatedFusion, concat

class TestGatedFusionModules(unittest.TestCase):
  def setUp(self):
    # Common setup for all tests
    self.N = 10
    self.hidden_size = 16

    self.tt_out = torch.randn(self.N, self.hidden_size)
    self.ta_out = torch.randn(self.N, self.hidden_size)

    self.aa_out = torch.randn(self.N, self.hidden_size)
    self.at_out = torch.randn(self.N, self.hidden_size)

    self.unimodal = Unimodal_GatedFusion(self.hidden_size)
    self.multimodal = Multimodal_GatedFusion(self.hidden_size)
    self.concat_t_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)
    self.concat_a_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)

  def test_unimodal_output_shape(self):
    output = self.unimodal(self.tt_out)
    self.assertEqual(output.shape, self.tt_out.shape, "Unimodal output shape mismatch")

  def test_multimodal_output_shape(self):
    Gt = concat(
      self.concat_t_layer,
      self.unimodal(self.tt_out),
      self.unimodal(self.ta_out)
    )
    Ga = concat(
      self.concat_a_layer,
      self.unimodal(self.aa_out),
      self.unimodal(self.at_out)
    )
    output = self.multimodal(Gt, Ga)
    self.assertEqual(output.shape, self.tt_out.shape, "Multimodal output shape mismatch")

  def test_concat_output_shape(self):
    Gtt = self.unimodal(self.tt_out)
    Gta = self.unimodal(self.ta_out)
    output = concat(self.concat_t_layer, Gtt, Gta)
    self.assertEqual(output.shape, self.tt_out.shape, "Concat output shape mismatch")

if __name__ == "__main__":
  torch.manual_seed(42)
  unittest.main()
