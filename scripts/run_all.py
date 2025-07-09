import argparse
import torch
import json
import logging
from pathlib import Path
import torch.optim as optim

from data.cifar10_loader import get_cifar10_loaders
from models.teacher import get_teacher, train_supervised, load_teacher
from models.student import get_student
from models.quant_student import QuantizableStudent
from distillation.kd_train import DistillationLoss, train_distill

from utils.paths import (
    get_model_path, TRAINING_HISTORY_PATH, OUTPUT_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Knowledge Distillation Pipeline')
    
    # Step selection
    parser.add_argument('--train-teacher', action='store_true',
                      help='Train the teacher model using supervised learning')
    parser.add_argument('--generate-soft-labels', action='store_true',
                      help='Generate soft labels from the teacher model')
    parser.add_argument('--train-student', action='store_true',
                      help='Train the student model using knowledge distillation')
    parser.add_argument('--train-baseline', action='store_true',
                      help='Train a baseline model using supervised learning')
    parser.add_argument('--quantize', action='store_true',
                      help='Quantize the student model')
    parser.add_argument('--evaluate', action='store_true',
                      help='Run final evaluation on all models')
    
    # Common training parameters
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to train on')
    parser.add_argument('--save-path', type=str, default=OUTPUT_DIR,
                      help='Path to save model or training history')
    
    # Distillation-specific parameters
    parser.add_argument('--temperature', type=float, default=4.0, 
                      help='Temperature for distillation')
    parser.add_argument('--alpha', type=float, default=0.7,
                      help='Weight for soft loss vs hard loss')
    
    # Regularization parameters
    parser.add_argument('--dropout-rate', type=float, default=0.05,
                      help='Dropout rate for student model')
    parser.add_argument('--early-stopping-patience', type=int, default=20,
                      help='Early stopping patience')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                      help='Label smoothing factor')
    parser.add_argument('--mix-alpha', type=float, default=1.0,
                      help='Weight for mixup loss vs hard loss')
    parser.add_argument('--augmentation', type=str, default=None,
                      help='Data augmentation strategy (none, mixup, cutmix)')

    # Quantization parameters
    parser.add_argument('--model-to-quantize', type=str, default='student', help='Model to quantize (student or teacher)')
    parser.add_argument('--qat', action='store_true',
                      help='Perform Quantization-Aware Training (QAT) on the student model')
    return parser.parse_args()


def train_teacher_model(train_loader, test_loader, num_epochs, lr, device, early_stopping_patience=20):
    """Train the teacher model using standard supervised learning with minimal early stopping."""
    logger.info("Training teacher model...")
    teacher_model = get_teacher(pretrained=True)
    teacher_history = train_supervised(
        teacher_model,
        train_loader,
        test_loader,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        early_stopping_patience=early_stopping_patience
    )
    # Save teacher training history
    with open(TRAINING_HISTORY_PATH.replace('.json', '_teacher.json'), 'w') as f:
        json.dump(teacher_history, f, indent=2)
    return teacher_history

def train_student_model(train_loader, test_loader, num_epochs, device, dropout_rate, qat=False):
    """Train the student model using knowledge distillation with minimal regularization."""
    logger.info("Training student model with distillation...")
    # Load teacher model
    teacher_model = load_teacher(get_model_path('teacher'), device=device)
    teacher_model.to(device)
    # Load student model
    student_model = QuantizableStudent(dropout=0.1)  # Use dropout rate for student model
    if qat:
        logger.info("Preparing student model for Quantization-Aware Training (QAT)...")
        student_model.fuse_model()  # Fuse layers for QAT
        student_model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
        torch.ao.quantization.prepare_qat(student_model, inplace=True)

    student_model = student_model.to(device)

    logger.info("Initializing distillation loss and optimizer...")
    distillation_loss = DistillationLoss(temperature=3, alpha=0.7)
    #optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
    optimizer = optim.SGD(student_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)  # Increased weight decay
    model_kind = 'student' if not qat else 'qat_student'
    student_history = train_distill(
        student=student_model,
        teacher=teacher_model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=distillation_loss,
        optimizer=optimizer,
        num_epochs=num_epochs,
        lr=0.05,
        device=device,
        early_stopping_patience=50,
        model_kind=model_kind,
        # drop to 0.1
        mix_alpha=1.0,
        augmentation='cutmix',  # Use cutmix for student training
        qat=qat
    )
    if qat:
        logger.info("Converting student model to quantized version after QAT...")
        student_model.to('cpu')
        student_model.eval()
        torch.ao.quantization.convert(student_model, inplace=True)
        torch.save(student_model.state_dict(), "models/qat_quantized_student.pth")

    # Save student training history
    with open(TRAINING_HISTORY_PATH.replace('.json', '_student.json'), 'w') as f:
        json.dump(student_history, f, indent=2)
    return student_history

def train_small_model(train_loader, test_loader, num_epochs, lr, device, dropout_rate=0.05, early_stopping_patience=20):
    """Train the small model using standard supervised learning with minimal regularization."""
    logger.info("Training small model without distillation...")
    small_model = get_student(dropout_rate=dropout_rate)  # Minimal dropout
    small_model = small_model.to(device)
    
    history = train_supervised(
        small_model, train_loader, test_loader,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        early_stopping_patience=early_stopping_patience,
        model_kind='small',
    )
    # Save small model training history
    with open(TRAINING_HISTORY_PATH.replace('.json', '_baseline.json'), 'w') as f:
        json.dump(history, f, indent=2)
    return history

def main():
    args = parse_args()
    logger.info(f"Using device: {args.device}")
    
    # Load CIFAR-10 data
    train_loader, test_loader, classes = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True
    )
    
    # Train teacher model
    if args.train_teacher:
        logger.info("Training teacher model...")
        teacher_history = train_teacher_model(train_loader, test_loader, args.num_epochs, args.lr, args.device, args.early_stopping_patience)

    # Train student model with distillation
    if args.train_student:
        logger.info("Training student model with distillation...")
        history = train_student_model(
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=args.num_epochs,
            device=args.device,
            dropout_rate=args.dropout_rate,
            qat=args.qat
        )
    
    # Train baseline model
    if args.train_baseline:
        logger.info("Training baseline model...")
        history = train_small_model(
            train_loader, test_loader, args.num_epochs, args.lr, args.device,
            dropout_rate=args.dropout_rate, early_stopping_patience=args.early_stopping_patience
        )
    
    # Quantize student model
    if args.quantize:
        model_fp32 = QuantizableStudent(dropout=0.1)
        model_fp32.quantize_model(
            train_loader=train_loader,
            state_dict_path=args.model_to_quantize,
            save_path=args.save_path
        )
    
    

if __name__ == '__main__':
    main() 