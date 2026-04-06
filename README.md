# IndoBERT POS Tagging Pipeline

Pipeline lengkap untuk fine-tuning IndoBERT pada tugas POS Tagging dengan single token, menggunakan arsitektur BERT → Linear → Softmax.

## Overview

Proyek ini mengimplementasikan fine-tuning IndoBERT untuk POS (Part-of-Speech) tagging dengan fokus pada single token classification. Arsitektur model terdiri dari:

- **Encoder**: IndoBERT (indobert-base-p1)
- **Feature Extraction**: Token classification menggunakan [CLS] token representation
- **Classification Head**: Linear layer + Softmax untuk prediksi label POS
- **Loss Function**: Cross-Entropy Loss

## Struktur Folder

```
├── config.yml              # Konfigurasi training
├── main.py                 # Script untuk explore data dan test embedding
├── data.py                 # Module untuk data loading & preprocessing
├── model.py                # Model architecture (IndoBERTForTokenClassification)
├── hugging_face.py         # Module untuk interact dengan HuggingFace Hub
├── train.py                # Script training lengkap
├── type.py                 # Type definitions dan enums
├── utils.py                # Utility functions
├── checkpoints/            # Direktori untuk menyimpan model checkpoints
├── logs/                   # Direktori untuk logging
└── README.md               # File ini
```

## Setup

### 1. Install Dependencies

```bash
pip install torch transformers pandas scikit-learn pyyaml tqdm huggingface-hub
```

### 2. Dataset

Dataset harus dalam format CSV dengan kolom minimal:

- `token`: Token individual (string atau numeric)
- `final_pos_tag`: Label POS tag

Contoh:

```csv
token,final_pos_tag
0,NUM-PK
1,NUM-PK
3,NUM-PK
```

## Usage

### 1. Explore Dataset

Untuk melihat struktur dan komposisi dataset:

```bash
python main.py --mode explore --dataset_file data.csv --log_file
```

**Output:**

- Informasi shape dataset
- Jumlah dan mapping unique labels
- Sample data (5 baris pertama)

### 2. Test Embedding Extraction

Untuk test embedding extraction dari model:

```bash
python main.py --mode embed --model_name indolem/indobert-base-p1 --log_file
```

**Output:**

- Model configuration info
- Tokenization example
- Sample embedding vector

### 3. Training (Main Pipeline)

Untuk melakukan training model:

```bash
python train.py --dataset data.csv --config config.yml
```

**Proses:**

1. **Data Preparation**
   - Load dataset dari CSV
   - Split: 70% train, 10% validation, 20% test
   - Create label mappings

2. **Model Initialization**
   - Load pretrained IndoBERT
   - Add classification head (Linear layer)
   - Move to GPU/CPU

3. **Training**
   - Training untuk N epochs (default: 5)
   - Validation setiap epoch
   - Early stopping jika validation loss tidak improve (patience=3)
   - Save best model checkpoint

4. **Evaluation**
   - Evaluate pada test set
   - Calculate: Accuracy, Precision, Recall, F1-Score
   - Generate classification report per label

**Output Checkpoints:**

- `checkpoints/model_epoch_X.pt` - Best model untuk setiap epoch
- `checkpoints/training_results.json` - Training metrics per epoch
- `checkpoints/evaluation_results.json` - Test set metrics

## Configuration (config.yml)

```yaml
# Model Configuration
model:
  model_name: 'indolem/indobert-base-p1' # Model dari HuggingFace
  num_labels: null # Auto-detected dari data
  hidden_size: 768
  freeze_bert: false # Full fine-tuning

# Training Configuration
training:
  batch_size: 32
  num_epochs: 5
  learning_rate: 2e-5
  weight_decay: 0.01
  warmup_steps: 500
  max_grad_norm: 1.0
  device: 'cuda' # atau "cpu"

# Data Configuration
data:
  test_size: 0.2
  validation_size: 0.1
  random_state: 42
  max_length: 128

# Output Configuration
output:
  model_save_dir: './checkpoints'
  log_dir: './logs'
```

## Model Architecture

```
Input Tokens
    ↓
Tokenizer (AutoTokenizer from HuggingFace)
    ↓
Input IDs + Attention Mask + Token Type IDs
    ↓
IndoBERT Encoder
    ↓
Last Hidden States (batch_size, seq_len, 768)
    ↓
Select [CLS] Token (batch_size, 768)
    ↓
Dropout (0.1)
    ↓
Linear Layer (768 → num_labels)
    ↓
Logits (batch_size, num_labels)
    ↓
Softmax (untuk inference) / CrossEntropyLoss (untuk training)
    ↓
Label Predictions
```

## Features

### 1. Data Handling

- ✅ Flexible CSV loading
- ✅ Automatic label mapping
- ✅ Train/Val/Test split
- ✅ Tokenization dengan padding & truncation

### 2. Model

- ✅ IndoBERT backbone
- ✅ Configurable freeze BERT encoder
- ✅ Simple linear classification head
- ✅ Dropout for regularization

### 3. Training

- ✅ AdamW optimizer dengan weight decay
- ✅ Linear warmup + decay learning rate scheduler
- ✅ Gradient clipping
- ✅ Mixed precision support (ready)
- ✅ Early stopping
- ✅ Checkpoint saving

### 4. Evaluation

- ✅ Accuracy, Precision, Recall, F1-Score
- ✅ Per-class classification report
- ✅ Training curves (loss, F1)

### 5. Logging

- ✅ Console logging dengan timestamp
- ✅ File logging
- ✅ Structured experiment tracking
- ✅ Results in JSON format

## Next Steps / Extensions

Untuk pengembangan lebih lanjut:

1. **Hyperparameter Tuning**
   - Gunakan Optuna/Ray Tune untuk automated hyperparameter search
   - Experiment dengan different batch sizes, learning rates

2. **Model Improvements**
   - Tambah bidirectional context modeling
   - Implement sequence labeling untuk multi-token
   - Add CRF layer untuk label dependencies

3. **Data Augmentation**
   - Token-level augmentation
   - Back-translation untuk bahasa Indonesia

4. **Inference**
   - Create inference script untuk production
   - Model quantization untuk faster inference
   - Batch inference optimization

5. **Experiment Tracking**
   - Integrate dengan WandB/MLflow untuk experiment tracking
   - Model versioning

## Training Tips

- **GPU Memory**: Kurangi batch_size jika OOM
- **Learning Rate**: 2e-5 adalah standard untuk fine-tuning BERT
- **Epochs**: 3-5 epochs biasanya sudah cukup
- **Warmup**: Warmup steps membantu stabilisasi training
- **Early Stopping**: Gunakan untuk prevent overfitting

## Citation & References

- IndoBERT: https://github.com/indobenchmark/indonlp
- HuggingFace Transformers: https://huggingface.co/transformers/
- BERT Paper: https://arxiv.org/abs/1810.04805

## License

MIT
