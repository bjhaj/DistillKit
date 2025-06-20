import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

def generate_soft_labels(teacher_model, trainloader, temperature=4.0, device=None):
    """
    Generate soft labels using the teacher model.
    
    Args:
        teacher_model (nn.Module): Trained teacher model
        trainloader (DataLoader): Training data loader
        temperature (float): Temperature for softmax scaling
        device (str, optional): Device to run inference on
        
    Returns:
        tuple: (soft_targets, hard_labels, images)
    """
    if device is None:
        device = next(teacher_model.parameters()).device
    
    teacher_model.eval()
    soft_targets = []
    hard_labels = []
    images = []
    
    with torch.no_grad():
        for batch in tqdm(trainloader, desc="Generating soft labels"):
            if len(batch) == 2:  # Regular dataloader
                img, label = batch
            else:  # Custom dataloader
                img, label = batch[0], batch[1]
                
            img = img.to(device)
            logits = teacher_model(img)
            soft = F.softmax(logits / temperature, dim=1)
            
            soft_targets.append(soft.cpu())
            hard_labels.append(label)
            images.append(img.cpu())
    
    # Combine all batches
    soft_targets = torch.cat(soft_targets, dim=0)
    hard_labels = torch.cat(hard_labels, dim=0)
    images = torch.cat(images, dim=0)
    
    return soft_targets, hard_labels, images

def save_distillation_data(soft_targets, hard_labels, images, save_dir='./data'):
    """
    Save distillation data to disk.
    
    Args:
        soft_targets (torch.Tensor): Soft targets from teacher
        hard_labels (torch.Tensor): Hard labels
        images (torch.Tensor): Training images
        save_dir (str): Directory to save data
    """
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(soft_targets, os.path.join(save_dir, "cifar10_soft_targets.pt"))
    torch.save(hard_labels, os.path.join(save_dir, "cifar10_labels.pt"))
    torch.save(images, os.path.join(save_dir, "cifar10_train_images.pt"))

def load_distillation_data(load_dir='./data'):
    """
    Load distillation data from disk.
    
    Args:
        load_dir (str): Directory containing saved data
        
    Returns:
        tuple: (soft_targets, hard_labels, images)
    """
    soft_targets = torch.load(os.path.join(load_dir, "cifar10_soft_targets.pt"))
    hard_labels = torch.load(os.path.join(load_dir, "cifar10_labels.pt"))
    images = torch.load(os.path.join(load_dir, "cifar10_train_images.pt"))
    
    return soft_targets, hard_labels, images 