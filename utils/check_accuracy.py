#!/usr/bin/env python3
"""
Check model accuracy and predictions.
This script helps debug if the model is working correctly.
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from data.cifar10_loader import get_cifar10_loaders
from models.student import get_student
from models.teacher import get_teacher
from utils.paths import get_model_path

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on CIFAR-10.")
    parser.add_argument('--model_name', type=str, default='small',
                        help='Name of the model to load (default: small)')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading CIFAR-10 data...")
    _, test_loader, _ = get_cifar10_loaders(batch_size=256, num_workers=2)
    
    # Load baseline model
    print(f"Loading model '{args.model_name}'...")
    if args.model_name == 'small' or args.model_name == 'student':
        baseline_model = get_student()
    else:
        baseline_model = get_teacher()
    
    model_path = get_model_path(args.model_name)
    print(f"Looking for model at: {model_path}")

    try:
        baseline_model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ '{args.model_name}' loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model '{args.model_name}': {e}")
        return
    
    baseline_model.to(device)
    baseline_model.eval()
    
    # Test on a small batch first
    print("\n=== Testing on first batch ===")
    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx > 0:
            break
            
        data, target = data.to(device), target.to(device)
        
        with torch.no_grad():
            output = baseline_model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            
        print(f"Batch {batch_idx + 1}:")
        print(f"  Batch size: {data.size(0)}")
        print(f"  Correct predictions: {correct}/{data.size(0)} ({100.0 * correct / data.size(0):.2f}%)")
        print(f"  Target labels (first 10): {target[:10].cpu().numpy()}")
        print(f"  Predicted labels (first 10): {pred[:10].squeeze().cpu().numpy()}")
        print(f"  Output logits range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        # Check if outputs are reasonable
        probs = torch.softmax(output, dim=1)
        max_probs = probs.max(dim=1)[0]
        print(f"  Max probabilities (first 10): {max_probs[:10].cpu().numpy()}")
        break
    
    # Test on full test set
    print("\n=== Testing on full test set ===")
    total_correct = 0
    total_samples = 0
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = baseline_model(data)
            pred = output.argmax(dim=1, keepdim=True)
            
            total_correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size(0)
            
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(pred.squeeze().cpu().numpy())
    
    accuracy = 100.0 * total_correct / total_samples
    print(f"Overall Test Accuracy: {total_correct}/{total_samples} ({accuracy:.2f}%)")
    
    # Analyze predictions per class
    print("\n=== Class-wise Analysis ===")
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    for class_idx in range(10):
        class_mask = all_targets == class_idx
        if class_mask.sum() > 0:
            class_correct = (all_predictions[class_mask] == class_idx).sum()
            class_total = class_mask.sum()
            class_acc = 100.0 * class_correct / class_total
            print(f"  {class_names[class_idx]:12}: {class_correct:4}/{class_total:4} ({class_acc:5.1f}%)")
    
    # Check confusion matrix sample
    print("\n=== Confusion Analysis (sample) ===")
    for true_class in range(3):  # Just show first 3 classes
        class_mask = all_targets == true_class
        if class_mask.sum() > 0:
            predictions_for_class = all_predictions[class_mask]
            print(f"True class {class_names[true_class]}:")
            for pred_class in range(10):
                count = (predictions_for_class == pred_class).sum()
                if count > 0:
                    print(f"  -> predicted as {class_names[pred_class]}: {count}")
    
    print(f"\n=== Summary ===")
    print(f"{args.model_name} accuracy: {accuracy:.2f}%")
    if accuracy < 50:
        print(f"⚠️  WARNING: {args.model_name} is suspiciously low!")
    elif accuracy > 70:
        print(f"✓ {args.model_name} appears to be working well")
    else:
        print(f"? {args.model_name} has moderate accuracy")

if __name__ == "__main__":
    main()
