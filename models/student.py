import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from models.teacher import unfreeze_layers_progressively

def get_student(dropout_rate=0.05, quantize=True):
    """
    Returns a MobileNetV2 model adapted for CIFAR-10.
    
    Args:
        dropout_rate (float): Dropout before final layer
        quantize (bool): Whether to prepare the model for quantization
    
    Returns:
        nn.Module: MobileNetV2-based student model
    """
    net = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Modify classifier for CIFAR-10
    net.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)

    net.classifier = nn.Sequential(
        #nn.Dropout(dropout_rate),
        nn.Linear(net.last_channel, 10)
    )

    return net


def save_student(model, path):
    """Save student model weights."""
    torch.save(model.state_dict(), path)


def load_student(path, device=None, quantize=False):
    """
    Load a saved student model.
    
    Args:
        path (str): Path to saved model weights
        device (str, optional): Device to load model on
        quantize (bool): If True, prepare the model for QAT
        
    Returns:
        nn.Module: Loaded student model
    """
    model = get_student(quantize=quantize)
    model.load_state_dict(torch.load(path, map_location=device))
    if device:
        model = model.to(device)
    return model
