import torch
import torch.nn as nn
from torchvision.models import resnet18

class StudentModel(nn.Module):
    """Enhanced student model with dropout and regularization."""
    
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(StudentModel, self).__init__()
        
        # Base ResNet18
        self.backbone = resnet18(weights=None)
        
        # Modify for CIFAR-10
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        
        # Get feature dimension
        num_features = self.backbone.fc.in_features
        
        # Replace final layer with dropout regularization
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )
        
        # Add batch normalization momentum for better regularization
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.01  # Slower moving average for better regularization
    
    def forward(self, x):
        return self.backbone(x)

def get_student(dropout_rate=0.3):
    """
    Get a student model with enhanced regularization.
    
    Args:
        dropout_rate (float): Dropout probability
        
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