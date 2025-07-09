# Knowledge Distillation and Quantization for CIFAR-10

A comprehensive PyTorch implementation of knowledge distillation combined with model quantization for efficient deep learning on CIFAR-10. This project demonstrates how to train a compact student model using knowledge from a larger teacher model, then apply quantization techniques for further optimization.

## 🎯 Project Overview

This repository implements a complete pipeline for:

1. **Teacher Training**: Train a large ResNet152 teacher model on CIFAR-10
2. **Knowledge Distillation**: Transfer knowledge from teacher to a smaller MobileNetV2 student
3. **Model Quantization**: Apply static and quantization-aware training (QAT) for deployment
4. **Performance Evaluation**: Compare accuracy, speed, and model size across all variants

### Key Features

- 🧠 **Knowledge Distillation** with temperature scaling and soft/hard loss combination
- 📱 **Mobile-Friendly Models** using MobileNetV2 architecture optimized for CIFAR-10
- ⚡ **Quantization Support** with both static and quantization-aware training
- 📊 **Comprehensive Evaluation** including accuracy, latency, throughput, and model size
- 🔧 **Modular Design** with separate components for easy experimentation

## 📁 Project Structure

```
kd_quantization_project/
├── data/
│   ├── __init__.py                 # Package initialization
│   ├── cifar10_loader.py           # CIFAR-10 data loading utilities
│   ├── cifar-10-batches-py/        # CIFAR-10 dataset files
│   ├── cifar-10-python.tar.gz      # Original CIFAR-10 archive
│   ├── cifar10_soft_targets.pt     # Pre-computed soft targets from teacher
│   ├── cifar10_train_images.pt     # Processed training images
│   └── cifar10_labels.pt           # Training labels
├── distillation/
│   ├── __init__.py                 # Package initialization
│   ├── distillation_losses.py     # Knowledge distillation loss functions
│   └── kd_train.py                 # Online knowledge distillation training
├── models/
│   ├── __init__.py                 # Package initialization
│   ├── student.py                  # MobileNetV2 student model
│   ├── teacher.py                  # ResNet152 teacher model
│   ├── quant_student.py            # Quantizable student model variants
│   ├── teacher_model.pth           # Trained teacher model weights
│   ├── student_model.pth           # Trained student model weights
│   ├── small_model.pth             # Baseline model weights
│   ├── qat_quantized_student.pth   # QAT quantized model weights
│   └── quantized.pth               # Static quantized model weights
├── scripts/
│   ├── __init__.py                 # Package initialization
│   ├── run_all.py                  # Main training pipeline
│   ├── train_all.sh                # Complete training script
│   └── check_accuracy.py           # Model evaluation utility
├── utils/
│   ├── __init__.py                 # Package initialization
│   ├── metrics.py                  # Performance measurement tools
│   ├── paths.py                    # Path management
│   └── additional_augmentation.py  # Data augmentation techniques
├── outputs/
│   ├── training_history_teacher.json    # Teacher training metrics
│   ├── training_history_student.json    # Student training metrics
│   ├── training_history_baseline.json   # Baseline training metrics
│   ├── training_history_quant_mod.json  # QAT training metrics
│   └── evaluation_results.json          # Final evaluation results
├── .git/                           # Git repository files
├── .gitignore                      # Git ignore file
└── requirements.txt                # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

```bash
# Required packages
torch>=1.11.0
torchvision>=0.12.0
numpy
tqdm
matplotlib
```

### Installation

```bash
git clone <repository-url>
cd kd_quantization_project
pip install -r requirements.txt  # If available, or install packages manually
```

### Running the Complete Pipeline

```bash
# Run the full knowledge distillation and quantization pipeline
bash scripts/train_all.sh
```

This will sequentially:
1. Train the ResNet152 teacher model (50 epochs)
2. Train the MobileNetV2 student with knowledge distillation (20 epochs)
3. Train a baseline model for comparison
4. Apply quantization to the student model

### Individual Components

```bash
# Train only the teacher model
python3 scripts/run_all.py --train-teacher --num-epochs 50

# Train baseline model
python3 scripts/run_all.py --train-baseline --num-epochs 20 --dropout-rate 0.1

# Train the student model with distillation
python3 scripts/run_all.py --train-student --num-epochs 20 --dropout-rate 0.1

# Apply static quantization to the student model
python3 scripts/run_all.py --quantiz

# Train the student model with QAT and distillation
python3 scripts/run_all.py --train-student --num-epochs 20 --dropout-rate 0.1 --qat

# Evaluate all models
python3 scripts/run_all.py --evaluate
```

### Model Evaluation

```bash
# Check teacher model accuracy
python3 scripts/check_accuracy.py --model_kind teacher --model_path models/teacher_model.pth

# Check student model accuracy
python3 scripts/check_accuracy.py --model_kind student --model_path models/student_model.pth

# Check baseline model accuracy
python3 scripts/check_accuracy.py --model_kind classic_student --model_path models/small_model.pth

# Check quantized model (QAT)
python3 scripts/check_accuracy.py --model_kind student --model_path models/qat_quantized_student.pth

# Check quantized model (Static)
python3 scripts/check_accuracy.py --quantized --model_path models/quantized.pth
```

## 🏗️ Architecture Details

### Teacher Model (ResNet152)
- **Architecture**: ResNet152 with ImageNet pretraining
- **Adaptation**: Modified for CIFAR-10 (32x32 input, 10 classes)
- **Training**: 50 epochs with standard supervised learning
- **Purpose**: Provides rich knowledge for distillation
- **Expected Accuracy**: ~93-94%

### Student Model (MobileNetV2)
- **Architecture**: MobileNetV2 optimized for CIFAR-10
- **Size**: ~2.3M parameters (vs ~60M for teacher)
- **Features**: 
  - Depthwise separable convolutions for efficiency
  - Configurable dropout rates
  - Quantization-friendly design
- **Expected Accuracy**: ~87-89% (with distillation)

### Quantizable Student (QuantizableStudent)
- **Base**: MobileNetV2 with quantization-aware modifications
- **Features**:
  - Fused conv-bn-relu layers for quantization
  - QAT (Quantization Aware Training) support
  - Static quantization capability
- **Expected Accuracy**: ~85-87% (quantized)

### Knowledge Distillation
- **Method**: Temperature-scaled softmax with combined loss
- **Loss Function**: `α * KL_divergence(soft_targets) + (1-α) * CrossEntropy(hard_targets)`
- **Default Parameters**: 
  - Temperature: 4.0
  - Alpha: 0.7 (70% soft knowledge, 30% hard targets)
- **Data Pipeline**: Uses pre-computed soft targets stored in `data/cifar10_soft_targets.pt`

## 📈 Performance Metrics

The project tracks multiple performance dimensions:

| Model Type | Parameters | Accuracy | Size (MB) | Inference Time | Compression |
|------------|------------|----------|-----------|----------------|-------------|
| Teacher (ResNet152) | ~60M | ~96.6% | ~240MB | Baseline | 1x |
| Student (MobileNetV2) | ~2.3M | ~94% | ~9MB | 3x faster | 27x smaller |
| Quantized Student | ~2.3M | ~93.5% | ~2.96MB | 5x faster | 81x smaller |
| Baseline (no distillation) | ~2.3M | ~91% | ~9MB | 3x faster | 27x smaller |

*Note: Actual results may vary based on training configuration and hardware*

## ⚙️ Configuration

### Training Parameters

```python
# Teacher Training
TEACHER_EPOCHS = 50
TEACHER_LR = 0.05
TEACHER_ARCHITECTURE = "ResNet152"
TEACHER_EARLY_STOPPING = 20

# Student Training  
STUDENT_EPOCHS = 20
STUDENT_LR = 0.05
DISTILLATION_TEMPERATURE = 4.0
DISTILLATION_ALPHA = 0.7
DROPOUT_RATE = 0.1

# Quantization
QUANTIZATION_BACKEND = "fbgemm"  # For x86 CPUs
CALIBRATION_BATCHES = 100
```

### Data Augmentation

- Standard CIFAR-10 augmentations (RandomCrop, RandomHorizontalFlip)
- Normalization with CIFAR-10 statistics: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Optional advanced augmentations (CutMix, MixUp) available in utils

### Model Files Generated

After training, the following model files will be created:
- `models/teacher_model.pth` - Trained teacher model
- `models/student_model.pth` - Student model with knowledge distillation
- `models/small_model.pth` - Baseline model without distillation
- `models/qat_quantized_student.pth` - QAT quantized student model
- `models/quantized.pth` - Static quantized student model

### Training History Files

Training metrics are saved in the outputs folder:
- `outputs/training_history_teacher.json` - Teacher training metrics
- `outputs/training_history_student.json` - Student training metrics  
- `outputs/training_history_baseline.json` - Baseline training metrics
- `outputs/training_history_quant_mod.json` - QAT training metrics
- `outputs/evaluation_results.json` - Final evaluation results

## 🔬 Evaluation Tools

### Model Accuracy Checker (`check_accuracy.py`)
```bash
python3 scripts/check_accuracy.py --model_kind student --model_path models/student_model.pth
```
Provides:
- Overall test accuracy on CIFAR-10
- Class-wise performance analysis (per class accuracy)
- Confusion matrix insights
- Model profiling (latency, throughput)
- Batch-level debugging information
- Model size measurement


### Performance Expectations

If you see these results, something may be wrong:
- **Teacher accuracy < 94%**: Check training epochs and learning rate
- **Student accuracy < 88%**: Check knowledge distillation pipeline
- **Quantized accuracy drop > 5%**: Review quantization calibration
- **No speed improvement from quantization**: Verify model is properly quantized

## 🔧 Customization

### Modifying Distillation
1. Edit `distillation/distillation_losses.py` for new loss functions
2. Adjust temperature/alpha in training scripts
3. Experiment with different teacher architectures
4. Try different soft label generation strategies

### Custom Quantization
1. Quantization functionality is integrated into the main training pipeline via `scripts/run_all.py`
2. Try different quantization schemes (dynamic, QAT) by modifying the `--qat` flag
3. Implement custom calibration datasets in the data loading utilities
4. Experiment with mixed-precision quantization by extending the model classes

### Hyperparameter Tuning
Key parameters to experiment with:
- **Distillation temperature**: 3.0, 4.0, 6.0
- **Alpha (soft/hard loss balance)**: 0.5, 0.7, 0.9
- **Student dropout rate**: 0.05, 0.1, 0.2
- **Learning rates**: Teacher (0.05, 0.1), Student (0.01, 0.05)

## 📚 References

- [Hinton et al. - Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Howard et al. - MobileNets: Efficient Convolutional Neural Networks](https://arxiv.org/abs/1704.04861)
- [Sandler et al. - MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- [PyTorch Quantization Documentation](https://pytorch.org/docs/stable/quantization.html)
- [He et al. - Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

## 📄 License

MIT License
---

**Note**: This project is designed for educational and research purposes. The implementation demonstrates key concepts in knowledge distillation and model quantization. For production deployment, additional optimizations and testing may be required based on specific hardware and performance requirements. 
