# IndoBERT POS Tagging

Single-token POS tagging with IndoBERT using BERT → Linear → Softmax.

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python main.py --dataset path/to/data.csv --config config.yml --log_file
```

Args:
- `--dataset`: CSV file with `token` and `final_pos_tag` (required)
- `--config`: Config file path (default: config.yml)
- `--log_file`: Enable file logging

## Dataset Format

```csv
token,final_pos_tag
0,NUM-PK
1,NUM-PK
```

## Config (config.yml)

- `batch_size`: 32 (↑ for better GPU use, ↓ if OOM)
- `device`: "cuda" or "cpu"
- `num_epochs`: 5
- `learning_rate`: 2e-5

Optimizations: Mixed precision AMP, multi-worker loading, early stopping.

## Output

- `checkpoints/model_epoch_*.pt`: Best models
- `checkpoints/training_results.json`: Metrics
- `checkpoints/evaluation_results.json`: Test metrics
- `logs/`: Execution logs

## Architecture

Token → Tokenizer → IndoBERT(768) → Dropout(0.1) → Linear → Softmax
