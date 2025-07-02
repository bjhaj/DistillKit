import torch
import torch.nn as nn 
import torch.nn.functional as F
from tqdm import tqdm
from models.teacher import load_teacher, get_model_path
from models.student import get_student, save_student
from data.cifar10_loader import get_cifar10_loaders, get_distillation_loader
import copy
import os

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



def distill_teacher(student, teacher, train_loader, test_loader, criterion, optimizer, epochs=40):
    teacher.eval()
    student.train()

    best_acc = 0.0
    best_model = None
    history = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to('cuda'), labels.to('cuda')
            
            with torch.no_grad():
                teacher_outputs = teacher(images)
            
            student_outputs = student(images)
            loss = criterion(student_outputs, teacher_outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = student_outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        # Evaluation
        student.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to('cuda'), labels.to('cuda')
                outputs = student(images)
                _, preds = outputs.max(1)
                test_correct += preds.eq(labels).sum().item()
                test_total += labels.size(0)

        test_acc = 100. * test_correct / test_total
        student.train()

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_model = copy.deepcopy(student.state_dict())

        # Save training history
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'test_acc': test_acc
        })

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")

    if best_model is not None:
        student.load_state_dict(best_model)

    return student, history
