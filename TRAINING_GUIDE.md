# IndoBERT POS Tagging - Complete Training Guide

## Overview

Pipeline lengkap untuk fine-tuning IndoBERT pada tugas single-token POS (Part-of-Speech) tagging. Arsitektur mengikuti baseline model yang sederhana dan efektif:

**BERT → Linear Classification Head → Softmax**

Data yang dihandle adalah single token (bukan sequence), sesuai dengan format dataset:

```csv
token,final_pos_tag
0,NUM-PK
1,NUM-PK
97536,NUM-PK
```

---

## Component Overview

### 1. **data.py** - Data Loading & Preprocessing

- Load CSV dataset
- Create label-to-id dan id-to-label mappings
- Handle missing files dan format validation

### 2. **hugging_face.py** - HuggingFace Integration

- Download model dari HuggingFace Hub
- Load tokenizer dan model
- Extract embedding vectors
- Model info logging

### 3. **model.py** - IndoBERTForTokenClassification

```
IndoBERT (768-dim)
    ↓ Dropout (0.1)
    ↓ Linear(768 → num_labels)
    ↓ Softmax (inference) / CrossEntropyLoss (training)
```

Key methods:

- `forward()`: Main inference forward pass
- `get_embedding_features()`: Extract BERT representations
- `freeze_bert_encoder()`: Option untuk transfer learning

### 4. **train.py** - Training Pipeline

Complete training workflow dengan:

- Data preparation (train/val/test split)
- Model initialization
- Training loop dengan early stopping
- Checkpoint saving
- Evaluation metrics calculation

### 5. **inference.py** - Model Inference

```bash
# Single prediction
python inference.py --checkpoint checkpoints/model_epoch_1.pt --token "0"

# Batch prediction
python inference.py --checkpoint checkpoints/model_epoch_1.pt --tokens "0" "1" "3"
```

### 6. **visualize.py** - Results Visualization

```bash
# Show all results
python visualize.py --checkpoint_dir ./checkpoints --all

# Show specific results
python visualize.py --training
python visualize.py --evaluation
```

---

## Quick Start

### Setup (5 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare your dataset (CSV format)
# Required columns: token, final_pos_tag

# 3. (Optional) Update config.yml for your needs
vim config.yml

# 4. Run training
python train.py --dataset your_data.csv --config config.yml
```

### Full Automated Setup

```bash
bash quickstart.sh path/to/your/data.csv
```

This will:

1. Install dependencies
2. Create directories
3. Explore dataset
4. Test embeddings
5. Train model
6. Save results

---

## Detailed Workflow

### Phase 1: Data Exploration

```bash
python main.py --mode explore --dataset_file data.csv --log_file
```

**Output:**

- Dataset shape dan statistics
- Unique labels dan mappings
- Sample rows

**Files Modified:**

- `logs/run_YYYY-MM-DD_HH-MM-SS.log` - Exploration log

---

### Phase 2: Embedding Test

```bash
python main.py --mode embed --model_name indolem/indobert-base-p1 --log_file
```

**What it does:**

1. Download IndoBERT model
2. Load tokenizer
3. Tokenize sample text
4. Extract embedding vector
5. Log model info (vocab size, hidden size)

**Output:**

- Model configuration info
- Sample embeddings
- Logs in `logs/` directory

---

### Phase 3: Model Training

```bash
python train.py --dataset data.csv --config config.yml
```

**Process:**

1. **Data Preparation**
   - Load CSV
   - Create label mappings
   - Split: 70% train, 10% val, 20% test
   - Create DataLoader instances

2. **Model Setup**
   - Load pretrained IndoBERT
   - Add Linear classification head
   - Initialize optimizer (AdamW)
   - Setup LR scheduler (linear warmup + decay)

3. **Training Loop** (per epoch)
   - Forward pass
   - Calculate loss
   - Backward pass
   - Gradient clipping
   - Optimizer step
   - Validation
   - Checkpoint saving if improved

4. **Early Stopping**
   - Monitor validation loss
   - Stop if no improvement for 3 epochs
   - Save best model

**Output:**

```
checkpoints/
├── model_epoch_1.pt       # Best model di epoch 1
├── model_epoch_2.pt       # Best model di epoch 2
├── training_results.json  # Training metrics per epoch
└── evaluation_results.json # Test set metrics
```

---

### Phase 4: Results Visualization

```bash
python visualize.py --checkpoint_dir ./checkpoints --all
```

**Output Format:**

```
TRAINING RESULTS SUMMARY
Epoch | Train Loss | Val Loss | Val Accuracy | Val F1-Score
---   | 0.5234     | 0.4821   | 0.8234       | 0.8156
---   | ...

Best Val Loss: 0.4821 (Epoch 2)
Best Val Accuracy: 0.8234 (Epoch 1)
Best Val F1-Score: 0.8156 (Epoch 2)

EVALUATION RESULTS (TEST SET)
Accuracy:  0.8150
Precision: 0.8200
Recall:    0.8150
F1-Score:  0.8120
```

---

### Phase 5: Model Inference

```bash
# Single token
python inference.py --checkpoint checkpoints/model_epoch_1.pt --token "0"

# Batch tokens
python inference.py --checkpoint checkpoints/model_epoch_1.pt --tokens "0" "1" "97536"
```

**Output:**

```
Prediction for token: 0
Predicted Label: NUM-PK
Confidence: 0.9234

Top Predictions:
  1. NUM-PK: 0.9234
  2. ADJ-P: 0.0456
  3. NOUN-P: 0.0310
```

---

## Configuration Guide

### config.yml

```yaml
# Model Configuration
model:
  model_name: 'indolem/indobert-base-p1' # HuggingFace model ID
  num_labels: null # Auto-inferred dari data
  hidden_size: 768 # BERT hidden size
  freeze_bert: false # false = full fine-tune, true = linear probe

# Training Configuration
training:
  batch_size: 32 # Adjust if OOM (reduce) or GPU idle (increase)
  num_epochs: 5 # 3-5 adalah typical
  learning_rate: 2e-5 # Standard untuk BERT fine-tune
  weight_decay: 0.01 # L2 regularization
  warmup_steps: 500 # Linear warmup untuk stable training
  max_grad_norm: 1.0 # Gradient clipping
  device: 'cuda' # "cuda" atau "cpu"

# Data Configuration
data:
  test_size: 0.2 # 20% untuk test
  validation_size: 0.1 # 10% dari train+val untuk validation
  random_state: 42 # Reproducibility
  max_length: 128 # Max tokens dalam sequence

# Output Configuration
output:
  model_save_dir: './checkpoints'
  log_dir: './logs'

# Experiment Configuration
experiment:
  name: 'indobert_postag_baseline'
  description: 'IndoBERT fine-tuning untuk POS tagging single token'
  seed: 42
```

### Hyperparameter Tuning Tips

| Parameter       | Tips                                            | Range        |
| --------------- | ----------------------------------------------- | ------------ |
| `batch_size`    | Larger = faster training, needs more GPU memory | 8-64         |
| `learning_rate` | 2e-5 adalah standard untuk BERT                 | 1e-5 to 5e-5 |
| `num_epochs`    | 3-5 biasanya cukup untuk BERT                   | 3-10         |
| `warmup_steps`  | ~10% dari total steps                           | 100-1000     |
| `weight_decay`  | Prevents overfitting                            | 0.0-0.1      |

---

## Understanding the Model

### Architecture

```
Single Token Input
    ↓
Token ID (via AutoTokenizer)
    ↓
[CLS] token_1 [SEP]  (padded to 128)
    ↓
IndoBERT Encoder
(12 layers, 12 heads, 768 dims)
    ↓
Last Hidden States (batch_size, seq_len=128, 768)
    ↓
Extract [CLS] representation (batch_size, 768)
    ↓
Dropout(0.1)
    ↓
Linear Layer (768 → num_labels)
    ↓
Logits (batch_size, num_labels)
    ↓
Softmax → Probabilities
Argmax → Label Prediction
```

### Why This Design?

- **[CLS] Token**: Contains aggregated information dari seluruh sequence
- **Dropout**: Regularization untuk prevent overfitting
- **Linear Head**: Sederhana tapi efektif untuk classification
- **Single Token**: Cocok untuk POS tagging token-level

---

## Handling Common Issues

### Issue: CUDA Out of Memory (OOM)

**Solution:** Kurangi batch_size di config.yml

```yaml
training:
  batch_size: 16 # Dari 32 ke 16
```

### Issue: Training terlalu lambat

**Solution:**

- Gunakan GPU: `device: "cuda"`
- Naik batch_size (jika GPU memory cukup)
- Reduce num_epochs atau gunakan early stopping

### Issue: Model overfitting (train loss turun, val loss naik)

**Solution:**

- Naik weight_decay
- Tambah dropout
- Reduce learning_rate
- Freeze BERT encoder

### Issue: Label mismatch atau missing labels

**Solution:** Pastikan CSV punya format yang benar

```csv
token,final_pos_tag
value1,label1
value2,label2
```

---

## Performance Expectations

Typical results untuk single-token POS tagging:

| Metric    | Typical Range |
| --------- | ------------- |
| Accuracy  | 0.75 - 0.90   |
| Precision | 0.75 - 0.90   |
| Recall    | 0.75 - 0.90   |
| F1-Score  | 0.75 - 0.90   |

Tergantung dari:

- Dataset size dan quality
- Label distribution balance
- Hyperparameter tuning

---

## Next Steps

### 1. Advanced Training Techniques

- Implement stratified k-fold cross-validation
- Use mixed-precision training untuk speed
- Implement learning rate finder
- Use curriculum learning

### 2. Model Improvements

- Add sequence context (not just single token)
- Implement ensemble methods
- Use different backbones (DistilBERT, RoBERTa)
- Add CRF layer untuk label correlation

### 3. Production Deployment

- Model quantization untuk inference speedup
- Create REST API dengan FastAPI
- Containerize dengan Docker
- Deploy ke cloud (AWS, GCP, Azure)

### 4. Experiment Tracking

- Integrate dengan Weights & Biases
- Track all experiments dan hyperparameters
- Compare model versions
- Version control datasets

---

## File Structure Reference

```
modular_experiment/
├── config.yml                      # Training configuration
├── main.py                         # Data exploration & embedding test
├── data.py                         # Data loading module
├── hugging_face.py                 # HuggingFace integration
├── model.py                        # Model architecture
├── train.py                        # Training script
├── inference.py                    # Inference script
├── visualize.py                    # Results visualization
├── type.py                         # Type definitions
├── utils.py                        # Utility functions
├── requirements.txt                # Dependencies
├── quickstart.sh                   # Automated setup script
├── README.md                       # Quick reference
├── TRAINING_GUIDE.md               # This file
├── checkpoints/                    # Model checkpoints
│   ├── model_epoch_*.pt
│   ├── training_results.json
│   └── evaluation_results.json
├── logs/                           # Log files
│   └── run_*.log
└── hugging_face/                   # Downloaded models cache
    └── indobenchmark/
        └── indobert-base-p1/
```

---

## Support & Debugging

### Enable Verbose Logging

```bash
# Console + File logging
python train.py --dataset data.csv --config config.yml

# With file logging
python main.py --mode explore --dataset_file data.csv --log_file
```

### Check GPU Status

```bash
# See GPU usage
nvidia-smi

# Monitor during training
watch -n 1 nvidia-smi
```

### Verify Installation

```bash
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
```

---

## References

- **IndoBERT**: https://github.com/indobenchmark/indonlp
- **HuggingFace**: https://huggingface.co
- **Transformers Paper**: https://arxiv.org/abs/1810.04805
- **PyTorch**: https://pytorch.org

---

**Last Updated**: April 2026  
**Version**: 1.0  
**Status**: Production Ready
