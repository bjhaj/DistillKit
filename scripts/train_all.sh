#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH

echo "Step 1: Training teacher model with pretrained ResNet152..."
python3 scripts/run_all.py --train-teacher --num-epochs 50

echo "Step 2: Generating soft labels from teacher..."
python3 scripts/run_all.py --generate-soft-labels --temperature 4.0

echo "Step 3: Training student model with distillation and moderate regularization..."
python3 scripts/run_all.py --train-student --num-epochs 50 --dropout-rate 0.3

echo "Step 4: Training baseline model with minimal regularization..."
python3 scripts/run_all.py --train-baseline --num-epochs 20 --dropout-rate 0.1

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