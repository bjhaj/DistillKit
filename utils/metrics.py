import torch
import time
import os
from tqdm import tqdm

def evaluate_model(model, test_loader, device=None):
    """
    Evaluate model accuracy on test set.
    
    Args:
        model (nn.Module): Model to evaluate
        test_loader (DataLoader): Test data loader
        device (str, optional): Device to run evaluation on
        
    Returns:
        float: Test accuracy
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total

def measure_inference_time(model, test_loader, num_batches=5, device=None):
    """
    Measure model inference time.
    
    Args:
        model (nn.Module): Model to evaluate
        test_loader (DataLoader): Test data loader
        num_batches (int): Number of batches to measure
        device (str, optional): Device to run evaluation on
        
    Returns:
        float: Average inference time per batch
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    inference_times = []
    
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= num_batches:
                break
                
            images = images.to(device)
            
            # Warmup
            if i == 0:
                for _ in range(10):
                    model(images)
            
            # Measure inference time
            start_time = time.time()
            model(images)
            end_time = time.time()
            inference_times.append(end_time - start_time)
    
    return sum(inference_times) / len(inference_times)

def get_model_size(model, path=None):
    """
    Get model size in bytes.
    
    Args:
        model (nn.Module): Model to measure
        path (str, optional): Path to save temporary file
        
    Returns:
        int: Model size in bytes
    """
    if path is None:
        path = "temp_model.pt"
    
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path)
    os.remove(path)  # Clean up temporary file
    
    return size

def format_size(size_bytes):
    """
    Format size in bytes to human readable format.
    
    Args:
        size_bytes (int): Size in bytes
        
    Returns:
        str: Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB" 