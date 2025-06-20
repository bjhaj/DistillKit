#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH

echo "Step 1: Training teacher model with pretrained ResNet152..."
python3 scripts/run_all.py --train-teacher --num-epochs 150 --lr 0.001 --early-stopping-patience 5

echo "Step 2: Generating soft labels from teacher..."
python3 scripts/run_all.py --generate-soft-labels

echo "Step 3: Training student model with distillation..."
python3 scripts/run_all.py --train-student --num-epochs 150 --dropout-rate 0.3 --mixup-alpha 0.2

echo "Step 4: Training baseline model..."
python3 scripts/run_all.py --train-baseline --num-epochs 150 --dropout-rate 0.3 --early-stopping-patience 7

echo "Step 5: Quantizing student model..."
python3 scripts/run_all.py --quantize

echo "Step 6: Running final evaluation..."
python3 scripts/run_all.py --evaluate

echo "Training and evaluation complete! Results are saved in:"
echo "- models/teacher_model.pth"
echo "- models/student_model.pth"
echo "- models/small_model.pth"
echo "- models/quantized_student_model.pth"
echo "- outputs/training_history.json"
echo "- outputs/training_history_baseline.json"
echo "- outputs/evaluation_results.json"