import argparse
import torch
import json
import logging
from pathlib import Path
import torch.optim as optim

from data.cifar10_loader import get_cifar10_loaders, get_distillation_loader
from models.teacher import get_teacher, save_teacher, train_supervised, load_teacher
from models.student import get_student, save_student
from distillation.generate_soft_labels import generate_soft_labels, save_distillation_data
from distillation.train_student import distill_model
from distillation.train_student_online_kd import distill_teacher, DistillationLoss

from quantization.quantize_model import quantize_student_model, evaluate_quantized_model
from utils.metrics import evaluate_model, measure_inference_time, get_model_size, format_size
from utils.paths import (
    get_model_path, get_data_path, TRAINING_HISTORY_PATH,
    EVALUATION_RESULTS_PATH, OUTPUT_DIR
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
    
    # Distillation-specific parameters
    parser.add_argument('--temperature', type=float, default=4.0, 
                      help='Temperature for distillation')
    parser.add_argument('--alpha', type=float, default=0.7,
                      help='Weight for soft loss vs hard loss')
    
    # Regularization parameters
    parser.add_argument('--dropout-rate', type=float, default=0.05,
                      help='Dropout rate for student model')
    parser.add_argument('--mixup-alpha', type=float, default=0.0,
                      help='Mixup alpha parameter (0 to disable)')
    parser.add_argument('--early-stopping-patience', type=int, default=20,
                      help='Early stopping patience')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                      help='Label smoothing factor')
    
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
    return teacher_model

def generate_soft_labels_from_teacher(teacher_model, train_loader, temperature, device):
    """Generate soft labels from the teacher model."""
    logger.info("Generating soft labels...")
    soft_targets, hard_labels, images = generate_soft_labels(
        teacher_model, train_loader, temperature, device
    )
    save_distillation_data(soft_targets, hard_labels, images)
    return soft_targets, hard_labels, images

def train_student_model(train_loader, test_loader, num_epochs, device, dropout_rate):
    """Train the student model using knowledge distillation with minimal regularization."""
    logger.info("Training student model with distillation...")
    # Load teacher model
    teacher_model = load_teacher(get_model_path('teacher'), device=device)
    teacher_model.to(device)
    # Load student model
    student_model = get_student(0.1)  # Use dropout rate for student model
    student_model = student_model.to(device)

    logger.info("Initializing distillation loss and optimizer...")
    distillation_loss = DistillationLoss(temperature=3, alpha=0.7)
    #optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
    optimizer = optim.SGD(student_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)  # Increased weight decay

    student_model, history = distill_teacher(
        student=student_model,
        teacher=teacher_model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=distillation_loss,
        optimizer=optimizer,
        epochs=num_epochs
    )
    
    # Save student model
    save_student(student_model, get_model_path('student'))
    return student_model, history

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
    # Save small model
    save_student(small_model, get_model_path('small'))
    return small_model, history

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
        teacher_model = train_teacher_model(train_loader, test_loader, args.num_epochs, args.lr, args.device, args.early_stopping_patience)
    
    # Generate soft labels
    if args.generate_soft_labels:
        logger.info("Loading teacher model for soft label generation...")
        teacher_model = load_teacher(get_model_path('teacher'), device=args.device)
        generate_soft_labels_from_teacher(teacher_model, train_loader, args.temperature, args.device)
    
    # Train student model with distillation
    if args.train_student:
        logger.info("Training student model with distillation...")
        student_model, history = train_student_model(
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=args.num_epochs,
            device=args.device,
            dropout_rate=args.dropout_rate
        )
        # Save training history
        with open(TRAINING_HISTORY_PATH, 'w') as f:
            json.dump(history, f, indent=2)
    
    # Train baseline model
    if args.train_baseline:
        logger.info("Training baseline model...")
        baseline_model, history = train_small_model(
            train_loader, test_loader, args.num_epochs, args.lr, args.device,
            dropout_rate=args.dropout_rate, early_stopping_patience=args.early_stopping_patience
        )
        # Save training history
        with open(TRAINING_HISTORY_PATH.replace('.json', '_baseline.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    # Quantize student model
    if args.quantize:
        logger.info("Quantizing student model...")
        student_model = get_student()
        student_model.load_state_dict(torch.load(get_model_path('student')))
        student_model = student_model.to(args.device)
        quantized_model = quantize_student_model(student_model, train_loader, args.device)
        quantized_path = get_model_path('student', quantized=True)
        save_student(quantized_model, quantized_path)
    
    # Evaluate models
    if args.evaluate:
        logger.info("Evaluating models...")
        results = {}
        
        # Load and evaluate teacher
        teacher_model = load_teacher(get_model_path('teacher'), device=args.device)
        results['teacher'] = {
            'accuracy': evaluate_model(teacher_model, test_loader, args.device),
            'inference_time': measure_inference_time(teacher_model, test_loader, device=args.device),
            'model_size': format_size(get_model_size(teacher_model))
        }
        
        # Load and evaluate student
        student_model = get_student()
        student_model.load_state_dict(torch.load(get_model_path('student')))
        student_model = student_model.to(args.device)
        results['student'] = {
            'accuracy': evaluate_model(student_model, test_loader, args.device),
            'inference_time': measure_inference_time(student_model, test_loader, device=args.device),
            'model_size': format_size(get_model_size(student_model))
        }
        
        # Load and evaluate baseline
        baseline_model = get_student()
        baseline_model.load_state_dict(torch.load(get_model_path('small')))
        baseline_model = baseline_model.to(args.device)
        results['baseline'] = {
            'accuracy': evaluate_model(baseline_model, test_loader, args.device),
            'inference_time': measure_inference_time(baseline_model, test_loader, device=args.device),
            'model_size': format_size(get_model_size(baseline_model))
        }
        
        # Evaluate quantized model if it exists
        try:
            quantized_model = get_student()
            quantized_model.load_state_dict(torch.load(get_model_path('student', quantized=True)))
            quantized_model = quantized_model.to(args.device)
            results['quantized'] = evaluate_quantized_model(quantized_model, test_loader, args.device)
            results['quantized']['model_size'] = format_size(get_model_size(quantized_model))
        except FileNotFoundError:
            logger.info("Quantized model not found, skipping evaluation")
        
        # Save evaluation results
        with open(EVALUATION_RESULTS_PATH, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print results
        logger.info("\nEvaluation Results:")
        for model_name, metrics in results.items():
            logger.info(f"\n{model_name.upper()}:")
            for metric_name, value in metrics.items():
                logger.info(f"{metric_name}: {value}")

if __name__ == '__main__':
    main() 