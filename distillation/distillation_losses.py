import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temperature=3, alpha=0.7):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)  # Added label smoothing for better generalization
    
    def forward(self, student_logits, teacher_logits, labels):
        soft_loss = self.kl_div(
            nn.functional.log_softmax(student_logits / self.temperature, dim=1),
            nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2)
        hard_loss = self.ce_loss(student_logits, labels)
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


class IntermediateFeatureLoss(nn.Module):
    """
    Computes MSE loss between intermediate features of student and teacher.
    Optionally uses a 1x1 conv adapter to align channel dimensions.
    """
    def __init__(self, student_channels, teacher_channels, use_adapter=True):
        super(IntermediateFeatureLoss, self).__init__()
        self.use_adapter = use_adapter
        if use_adapter:
            self.adapter = nn.Conv2d(student_channels, teacher_channels, kernel_size=1)
        else:
            self.adapter = None
        self.mse = nn.MSELoss()

    def forward(self, student_feat, teacher_feat):
        if self.use_adapter and self.adapter:
            student_feat = self.adapter(student_feat)
        return self.mse(student_feat, teacher_feat)

def compute_total_loss(student_logits, teacher_logits, student_feat, teacher_feat, labels,
                       distill_criterion, intermediate_loss_fn,
                       alpha=0.7, beta=1.0):
    """
    Combines soft distillation loss and intermediate feature loss.

    Returns:
        Weighted total loss: alpha * distill_loss + beta * feature_loss
    """
    distill_loss = distill_criterion(student_logits, teacher_logits, labels)
    feat_loss = intermediate_loss_fn(student_feat, teacher_feat)
    return alpha * distill_loss + beta * feat_loss

