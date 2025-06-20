import torch
import torch.quantization
import torch.nn as nn
import time
import os
from tqdm import tqdm

def quantize_student_model(model, calibration_loader, device=None):
    """
    Quantize the student model using static quantization.
    
    Args:
        model (nn.Module): Student model to quantize
        calibration_loader (DataLoader): DataLoader for calibration
        device (str, optional): Device to run calibration on
        
    Returns:
        nn.Module: Quantized model
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Prepare model for quantization
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Prepare for quantization
    torch.quantization.prepare(model, inplace=True)
    
    # Calibrate with training data
    with torch.no_grad():
        for images, _ in tqdm(calibration_loader, desc="Calibrating"):
            images = images.to(device)
            model(images)
    
    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)
    
    return model

def save_quantized_model(model, path):
    """Save quantized model."""
    torch.save(model.state_dict(), path)

def load_quantized_model(model_class, path, device=None):
    """
    Load a quantized model.
    
    Args:
        model_class: Model class to instantiate
        path (str): Path to saved quantized model
        device (str, optional): Device to load model on
        
    Returns:
        nn.Module: Loaded quantized model
    """
    model = model_class()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    model.load_state_dict(torch.load(path, map_location=device))
    torch.quantization.convert(model, inplace=True)
    
    if device:
        model = model.to(device)
    return model

def evaluate_quantized_model(model, test_loader, device=None):
    """
    Evaluate quantized model performance.
    
    Args:
        model (nn.Module): Quantized model to evaluate
        test_loader (DataLoader): Test data loader
        device (str, optional): Device to run evaluation on
        
    Returns:
        dict: Evaluation metrics
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    correct = 0
    total = 0
    inference_times = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()
            inference_times.append(end_time - start_time)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_inference_time = sum(inference_times) / len(inference_times)
    
    # Calculate model size
    model_size = os.path.getsize(torch.save(model.state_dict(), "temp.pt"))
    os.remove("temp.pt")  # Clean up temporary file
    
    return {
        'accuracy': accuracy,
        'avg_inference_time': avg_inference_time,
        'model_size_bytes': model_size
    } 