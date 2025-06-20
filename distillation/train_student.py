import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def distillation_loss(student_logits, soft_targets, hard_labels, temperature, alpha):
    """
    Compute the distillation loss.
    
    Args:
        student_logits (torch.Tensor): Student model logits
        soft_targets (torch.Tensor): Teacher's soft targets
        hard_labels (torch.Tensor): Ground truth labels
        temperature (float): Temperature for softmax scaling
        alpha (float): Weight for soft loss vs hard loss
        
    Returns:
        torch.Tensor: Combined distillation loss
    """
    # Soft loss: KL Divergence
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        soft_targets,
        reduction='batchmean'
    ) * (temperature ** 2)
    
    # Hard loss: Cross Entropy
    hard_loss = F.cross_entropy(student_logits, hard_labels)
    
    # Combine losses
    return alpha * soft_loss + (1 - alpha) * hard_loss

def mixup_data(x, y, alpha=1.0, device='cuda'):
    """Mixup augmentation for better regularization."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss calculation."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_student(student_model, train_loader, test_loader, 
                 temperature=4.0, alpha=0.7, lr=0.1, 
                 num_epochs=100, device=None, mixup_alpha=0.0,
                 early_stopping_patience=10, min_delta=0.001):
    """
    Enhanced student training with knowledge distillation and overfitting prevention.
    
    Args:
        student_model (nn.Module): Student model to train
        train_loader (DataLoader): Training data loader with soft targets
        test_loader (DataLoader): Test data loader
        temperature (float): Temperature for softmax scaling
        alpha (float): Weight for soft loss vs hard loss
        lr (float): Learning rate
        num_epochs (int): Number of training epochs
        device (str, optional): Device to train on
        mixup_alpha (float): Mixup alpha parameter (0 to disable)
        early_stopping_patience (int): Early stopping patience
        min_delta (float): Minimum improvement for early stopping
        
    Returns:
        list: Training history (losses and accuracies)
    """
    if device is None:
        device = next(student_model.parameters()).device
    
    student_model = student_model.to(device)
    
    # Enhanced optimizer with stronger regularization (same as baseline)
    optimizer = optim.SGD(student_model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
    
    # Same scheduler as baseline model for consistency
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, 
                                                   patience=3, min_lr=1e-6)
    
    history = []
    best_acc = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        student_model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in pbar:
            if len(batch) == 3:  # Distillation dataset
                images, soft_targets, hard_labels = batch
            else:  # Regular dataset
                images, hard_labels = batch
                soft_targets = None
            
            images = images.to(device)
            hard_labels = hard_labels.to(device)
            if soft_targets is not None:
                soft_targets = soft_targets.to(device)
            
            # Apply Mixup if enabled
            if mixup_alpha > 0:
                images, hard_labels_a, hard_labels_b, lam = mixup_data(images, hard_labels, mixup_alpha, device)
            
            optimizer.zero_grad()
            student_logits = student_model(images)
            
            if soft_targets is not None:
                if mixup_alpha > 0:
                    # For mixup with distillation, we use the hard loss part with mixup
                    soft_loss = F.kl_div(
                        F.log_softmax(student_logits / temperature, dim=1),
                        soft_targets, reduction='batchmean'
                    ) * (temperature ** 2)
                    hard_loss = mixup_criterion(F.cross_entropy, student_logits, hard_labels_a, hard_labels_b, lam)
                    loss = alpha * soft_loss + (1 - alpha) * hard_loss
                else:
                    loss = distillation_loss(
                        student_logits, soft_targets, hard_labels,
                        temperature, alpha
                    )
            else:
                if mixup_alpha > 0:
                    loss = mixup_criterion(F.cross_entropy, student_logits, hard_labels_a, hard_labels_b, lam)
                else:
                    loss = F.cross_entropy(student_logits, hard_labels)
            
            loss.backward()
            
            # Gradient clipping (same as baseline model)
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = student_logits.max(1)
            train_total += hard_labels.size(0)
            train_correct += predicted.eq(hard_labels).sum().item()
            
            pbar.set_postfix({
                'loss': train_loss / (pbar.n + 1),
                'acc': 100. * train_correct / train_total
            })
        
        # Evaluation
        student_model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = student_model(images)
                loss = F.cross_entropy(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        # Update learning rate based on validation accuracy (same as baseline)
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
        
        # Overfitting detection (same as baseline model)
        train_val_gap = metrics['train_acc'] - metrics['test_acc']
        if train_val_gap > 10.0:  # Warning threshold
            logger.warning(f"Overfitting detected! Train-Val gap: {train_val_gap:.2f}%")
        
        # Early stopping and best model saving
        current_acc = metrics['test_acc']
        if current_acc > best_acc + min_delta:
            best_acc = current_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Check early stopping (also consider overfitting like baseline)
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