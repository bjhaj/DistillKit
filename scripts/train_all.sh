#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH

echo "Step 1: Training teacher model with pretrained ResNet152..."
python3 scripts/run_all.py --train-teacher --num-epochs 50

echo "Step 3: Training student model with distillation and moderate regularization..."
python3 scripts/run_all.py --train-student --num-epochs 20 --dropout-rate 0.1

echo "Step 4: Training baseline model with minimal regularization..."
python3 scripts/run_all.py --train-baseline --num-epochs 20 --dropout-rate 0.1

echo "Step 5: Quantizing student model..."
python3 scripts/run_all.py --quantize --model-to-quantize '/scratch/bjhaj/kd_quantization_project/models/student_model.pth' --save-path '/scratch/bjhaj/kd_quantization_project/models/quantized.pth'

echo "step 6: check accuracy"
python3 scripts/check_accuracy.py --model_kind 'quantized_mobilenet' --model_path '/scratch/bjhaj/kd_quantization_project/models/qat_quantized_student.pth' --quantized
