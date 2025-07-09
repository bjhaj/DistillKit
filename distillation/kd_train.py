import torch
import torch.nn as nn 
import torch.nn.functional as F
from tqdm import tqdm
from models.teacher import load_teacher, get_model_path
from models.teacher import unfreeze_layers_progressively, save_teacher
from utils.additional_augmentation import mixup_data, cutmix_data
import copy
import torch.optim as optim
import os
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Distillation Loss Function
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



def train_distill(student, teacher, train_loader, test_loader, criterion, optimizer, num_epochs=20, lr=0.01, device='cuda', 
                    early_stopping_patience=7, min_delta=0.001, use_progressive_unfreezing=True, model_kind='student', augmentation=None, mix_alpha=1.0, qat=False):
    """
    Enhanced supervised training loop with aggressive regularization for teacher model.
    """
    student = student.to(device)
    teacher = teacher.to(device)
    # Ensure teacher is in eval mode
    teacher.eval()
    # More aggressive learning rate scheduling
    # replace with cosine annealing if needed
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=3, min_lr=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    ce_loss = nn.CrossEntropyLoss() 
    history = []
    best_acc = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Progressive unfreezing to reduce overfitting
        if use_progressive_unfreezing:
            unfreeze_layers_progressively(student, epoch)
        # 🔥 QAT warmup control
        if qat:
            if epoch == 0:
                student.apply(torch.ao.quantization.disable_observer)
                student.apply(torch.ao.quantization.disable_fake_quant)
            elif epoch == 20:
                student.apply(torch.ao.quantization.enable_observer)
                student.apply(torch.ao.quantization.enable_fake_quant)
            
        # Training phase with stronger regularization
        student.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)
            # add in data augmentation if specified
            if augmentation == 'mixup':
                images, y_a, y_b, lam = mixup_data(images, targets, alpha=mix_alpha)
            elif augmentation == 'cutmix':
                images, y_a, y_b, lam = cutmix_data(images, targets, alpha=mix_alpha)

            with torch.no_grad():
                teacher_outputs = teacher(images)
            
            optimizer.zero_grad()
            student_outputs = student(images)
            #loss = criterion(student_outputs, teacher_outputs, targets)
            if augmentation in ['mixup', 'cutmix']:
                loss = lam * criterion(student_outputs, teacher_outputs, y_a) + (1 - lam) * criterion(student_outputs, teacher_outputs, y_b)
            else:
                loss = criterion(student_outputs, teacher_outputs, targets)
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = student_outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': train_loss / (pbar.n + 1),
                'acc': 100. * train_correct / train_total
            })
        
        # Evaluation phase
        student.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = student(images)
                loss = ce_loss(outputs, targets)

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
            # Save the best model
            save_teacher(student, get_model_path(model_kind))
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