import torch.nn as nn
import torch.nn.functional as F

class MaskedKLDivLoss(nn.Module):
    def __init__(self, tau=1.0):
        super(MaskedKLDivLoss, self).__init__()
        self.tau = tau

    def forward(self, student_logits, teacher_logits, mask):
        mask = mask.contiguous().view(-1, 1).float()
        teacher_soft = F.softmax(teacher_logits / self.tau, dim=-1)
        student_log_soft = F.log_softmax(student_logits / self.tau, dim=-1)
        kl = F.kl_div(student_log_soft, teacher_soft, reduction='none').sum(dim=1)
        loss = (kl * mask.squeeze()).sum() / mask.sum()
        return loss * (self.tau ** 2)

class MaskedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight

    def forward(self, log_probs, target, mask):
        mask = mask.contiguous().view(-1).float()
        nll = F.nll_loss(log_probs, target, weight=self.weight, reduction='none')
        loss = (nll * mask).sum() / mask.sum()
        return loss