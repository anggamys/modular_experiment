#!/bin/bash

# Quick Start Guide untuk IndoBERT POS Tagging Pipeline

set -e

echo "IndoBERT POS Tagging - Quick Start"
echo ""

# Check if dataset is provided
if [ -z "$1" ]; then
    echo "Usage: bash quickstart.sh <path_to_dataset.csv>"
    echo ""
    echo "Example:"
    echo "  bash quickstart.sh ../pos-tagging/token/result/tabular/final_validation_data.csv"
    exit 1
fi

DATASET=$1

echo "Step 1: Enable Logging"
echo "---"
mkdir -p logs checkpoints
echo "✓ Created directories"
echo ""

echo "Step 2: Explore Dataset"
echo "---"
python main.py --mode explore --dataset_file "$DATASET" --log_file
echo "✓ Dataset exploration completed"
echo ""

echo "Step 3: Test Model Embedding"
echo "---"
python main.py --mode embed --model_name indobenchmark/indobert-base-p1 --log_file
echo "✓ Embedding test completed"
echo ""

echo "Step 4: Start Training"
echo "---"
python train.py --dataset "$DATASET" --config config.yml
echo "✓ Training completed"
echo ""

echo "=========================================="
echo "Quick Start Completed Successfully!"
echo "=========================================="
echo ""
echo "Results are saved in:"
echo "  - checkpoints/model_epoch_*.pt (trained models)"
echo "  - checkpoints/training_results.json (training metrics)"
echo "  - checkpoints/evaluation_results.json (test metrics)"
echo "  - logs/*.log (detailed logs)"
echo ""

echo "Next Steps:"
echo "  1. Check the results in checkpoints/ directory"
echo "  2. Use best model for inference:"
echo "     python inference.py --checkpoint checkpoints/model_epoch_*.pt --token <your_token>"
echo "  3. Review training/evaluation metrics in JSON files"
