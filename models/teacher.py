import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet152, ResNet152_Weights
from tqdm import tqdm
import logging
from utils.paths import get_model_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_teacher(pretrained=True, freeze_backbone=True):
    """
    Get a ResNet152 teacher model configured for CIFAR-10 with regularization.
    
    Args:
        pretrained (bool): Whether to use ImageNet pretrained weights
        freeze_backbone (bool): Whether to freeze backbone layers initially
        
    Returns:
        nn.Module: Configured ResNet152 model
    """
    # --- Define ResNet152 (ImageNet Pretrained) ---
    net = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)  # use pretrained weights
    
    # 🔧 Patch input layers for CIFAR-10 (32x32 input)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()  # remove the downsampling maxpool
    
    # Add dropout to the final layer for regularization
    net.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(net.fc.in_features, 10)
    )
    
    # Freeze early layers to prevent overfitting
    if freeze_backbone:
        # Freeze all layers except the last few
        for name, param in net.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
    
    return net


def unfreeze_layers_progressively(model, epoch, unfreeze_schedule=None):
    """
    Progressively unfreeze layers during training to reduce overfitting.
    
    Args:
        model: The teacher model
        epoch: Current epoch number
        unfreeze_schedule: Dict mapping epochs to layers to unfreeze
    """
    if unfreeze_schedule is None:
        unfreeze_schedule = {
            5: ['layer4'],
            10: ['layer3'],
            15: ['layer2']
        }
    
    if epoch in unfreeze_schedule:
        layers_to_unfreeze = unfreeze_schedule[epoch]
        for layer_name in layers_to_unfreeze:
            for name, param in model.named_parameters():
                if layer_name in name:
                    param.requires_grad = True
        logger.info(f"Unfroze layers: {layers_to_unfreeze} at epoch {epoch}")


def train_supervised(model, train_loader, test_loader, num_epochs=20, lr=0.01, device='cuda', 
                    early_stopping_patience=7, min_delta=0.001, use_progressive_unfreezing=True, model_kind='teacher'):
    """
    Enhanced supervised training loop with aggressive regularization for teacher model.
    """
    model = model.to(device)
    
    # --- Loss & Optimizer with stronger regularization ---
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)  # Increased label smoothing
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)  # Increased weight decay
    
    # More aggressive learning rate scheduling
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, 
                                                   patience=3, min_lr=1e-6)
    
    history = []
    best_acc = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Progressive unfreezing to reduce overfitting
        if use_progressive_unfreezing:
            unfreeze_layers_progressively(model, epoch)
        
        # Training phase with stronger regularization
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': train_loss / (pbar.n + 1),
                'acc': 100. * train_correct / train_total
            })
        
        # Evaluation phase
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        # Update learning rate based on validation accuracy
        scheduler.step(100. * test_correct / test_total)
        
        # Record metrics
        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'train_acc': 100. * train_correct / train_total,
            'test_loss': test_loss / len(test_loader),
            'test_acc': 100. * test_correct / test_total
        }
        history.append(metrics)
        
        # Overfitting detection
        train_val_gap = metrics['train_acc'] - metrics['test_acc']
        if train_val_gap > 10.0:  # Warning threshold
            logger.warning(f"Overfitting detected! Train-Val gap: {train_val_gap:.2f}%")
        
        # Early stopping and best model saving
        current_acc = metrics['test_acc']
        if current_acc > best_acc + min_delta:
            best_acc = current_acc
            patience_counter = 0
            save_teacher(model, get_model_path(model_kind))
        else:
            patience_counter += 1
        
        # Check early stopping (also consider overfitting)
        if patience_counter >= early_stopping_patience or train_val_gap > 20.0:
            if train_val_gap > 20.0:
                logger.info(f"Stopping due to severe overfitting (gap: {train_val_gap:.2f}%)")
            else:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
        
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {metrics['train_loss']:.4f} | "
            f"Train Acc: {metrics['train_acc']:.2f}% | "
            f"Test Loss: {metrics['test_loss']:.4f} | "
            f"Test Acc: {metrics['test_acc']:.2f}% | "
            f"Gap: {train_val_gap:.2f}%"
        )
    
    return history

def save_teacher(model, path):
    """Save teacher model weights."""
    torch.save(model.state_dict(), path)

def load_teacher(path, device=None):
    """
    Load a saved teacher model.
    
    Args:
        path (str): Path to saved model weights
        device (str, optional): Device to load model on
        
    Returns:
        nn.Module: Loaded teacher model
    """
    model = get_teacher(pretrained=False, freeze_backbone=False)  # Match the saved model structure
    
    # Load state dict with error handling for structure changes
    try:
        model.load_state_dict(torch.load(path, map_location=device))
    except RuntimeError as e:
        logger.warning(f"State dict mismatch: {e}")
        # Try to load with strict=False to handle fc layer structure changes
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded model with relaxed matching due to architecture changes")
    
    if device:
        model = model.to(device)
    return model