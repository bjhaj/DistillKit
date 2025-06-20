# Knowledge Distillation with Quantization

This project implements a knowledge distillation pipeline for CIFAR-10 classification, with optional model quantization. The pipeline trains a large teacher model (ResNet152), distills its knowledge into a smaller student model (ResNet18), and optionally quantizes the student model for deployment.

## Project Structure

```
DistillKit/
├── data/
│   └── cifar10_loader.py            # Dataset loading and transforms
├── models/
│   ├── student.py                   # Student model definition
│   └── teacher.py                   # Teacher model definition
├── distillation/
│   ├── generate_soft_labels.py      # Generate soft targets using teacher
│   └── train_student.py             # Train distilled student model
├── quantization/
│   └── quantize_model.py            # Quantize student model and evaluate
├── utils/
│   ├── metrics.py                   # Accuracy, size, inference speed utilities
│   └── paths.py                     # Paths to data/model assets
├── scripts/
│   └── run_all.py                   # End-to-end run script
└── README.md
```

## Features

- Knowledge distillation from ResNet152 (teacher) to ResNet18 (student)
- Soft target generation with temperature scaling
- Student model training with combined soft and hard losses
- Model quantization for deployment
- Comprehensive evaluation metrics (accuracy, size, inference speed)
- Modular and reusable code structure

## Requirements

- Python 3.7+
- PyTorch 1.8+
- torchvision
- tqdm
- numpy

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the complete pipeline:
```bash
python scripts/run_all.py
```

Optional arguments:
```bash
python scripts/run_all.py \
    --batch-size 128 \
    --num-epochs 100 \
    --temperature 4.0 \
    --alpha 0.7 \
    --lr 0.1 \
    --device cuda \
    --skip-teacher \
    --skip-distillation \
    --skip-quantization
```

## Pipeline Steps

1. **Data Loading**: Load and preprocess CIFAR-10 dataset
2. **Teacher Training**: Train or load a ResNet152 teacher model
3. **Soft Target Generation**: Generate soft labels using the teacher model
4. **Student Training**: Train a ResNet18 student model using knowledge distillation
5. **Quantization**: Optionally quantize the student model
6. **Evaluation**: Compare teacher, student, and quantized student models

## Outputs

The pipeline generates:
- Trained model weights in `models/`
- Soft targets and labels in `data/`
- Training history and evaluation results in `outputs/`

## Customization

- Modify model architectures in `models/`
- Adjust distillation parameters in `scripts/run_all.py`
- Add new evaluation metrics in `utils/metrics.py`

## License

MIT License 
