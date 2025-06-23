import torch
import torch.nn as nn
from torchvision.models import resnet18

class StudentModel(nn.Module):
    """Enhanced student model with minimal regularization for better convergence."""
    
    def __init__(self, num_classes=10, dropout_rate=0.05):
        super(StudentModel, self).__init__()
        
        # Base ResNet18
        self.backbone = resnet18(weights=None)
        
        # Replace final layer with minimal dropout
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )
        
        # Use default batch normalization momentum for better convergence
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.1  # Default momentum for better convergence
    
    def forward(self, x):
        return self.backbone(x)

def get_student(dropout_rate=0.05):
    """
    Get a student model with minimal regularization for better convergence.
    
    Args:
        dropout_rate (float): Dropout probability (default: 0.05)
        
    Returns:
        nn.Module: Configured student model
    """
    return StudentModel(num_classes=10, dropout_rate=dropout_rate)

def save_student(model, path):
    """Save student model weights."""
    torch.save(model.state_dict(), path)

def load_student(path, device=None):
    """
    Load a saved student model.
    
    Args:
        path (str): Path to saved model weights
        device (str, optional): Device to load model on
        
    Returns:
        nn.Module: Loaded student model
    """
    model = get_student()
    model.load_state_dict(torch.load(path, map_location=device))
    if device:
        model = model.to(device)
    return model 