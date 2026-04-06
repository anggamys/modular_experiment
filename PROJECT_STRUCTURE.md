# Project Structure & File Guide

## Complete File Organization

### 📦 Core Training Pipeline

| File             | Purpose                                       | Size       | Status |
| ---------------- | --------------------------------------------- | ---------- | ------ |
| **train.py**     | Complete training script dengan trainer class | ~450 lines | ✅ New |
| **model.py**     | IndoBERTForTokenClassification architecture   | ~150 lines | ✅ New |
| **inference.py** | Model inference & prediction                  | ~200 lines | ✅ New |
| **visualize.py** | Results visualization & metrics display       | ~280 lines | ✅ New |

### 📊 Data & Utilities

| File                | Purpose                              | Status      |
| ------------------- | ------------------------------------ | ----------- |
| **data.py**         | Data loading & label mapping         | ✅ Existing |
| **hugging_face.py** | HuggingFace Hub integration          | ✅ Existing |
| **utils.py**        | Logging, argument parsing, utilities | ✅ Existing |
| **type.py**         | Type definitions & enums             | ✅ Existing |

### 🔧 Configuration & Setup

| File                 | Purpose                                 | Status |
| -------------------- | --------------------------------------- | ------ |
| **config.yml**       | Training hyperparameters & model config | ✅ New |
| **requirements.txt** | Python dependencies                     | ✅ New |
| **quickstart.sh**    | Automated setup & training script       | ✅ New |

### 📚 Documentation

| File                  | Purpose                      | Pages      |
| --------------------- | ---------------------------- | ---------- |
| **README.md**         | Quick reference & overview   | 3          |
| **TRAINING_GUIDE.md** | Comprehensive training guide | 15         |
| **examples.py**       | Programmatic usage examples  | ~300 lines |

### 📁 Generated Directories

| Directory         | Purpose                              |
| ----------------- | ------------------------------------ |
| **checkpoints/**  | Model checkpoints & training results |
| **logs/**         | Training & execution logs            |
| **hugging_face/** | Cached HuggingFace models            |

---

## Quick Navigation

### For Getting Started

1. **README.md** - Start here for quick overview
2. **quickstart.sh** - Automated setup (1 command)
3. **TRAINING_GUIDE.md** - Comprehensive guide

### For Training

- **config.yml** - Adjust hyperparameters
- **train.py** - Run training
- **visualize.py** - Check results

### For Inference

- **inference.py** - Make predictions
- **examples.py** - Usage examples

### For Development

- **model.py** - Model architecture
- **data.py** - Data handling
- **train.py** - Trainer class

---

## File Dependencies

```
train.py
├── model.py
├── data.py
├── hugging_face.py
├── type.py
├── utils.py
└── config.yml

inference.py
├── model.py
├── hugging_face.py
├── type.py
└── utils.py

main.py
├── data.py
├── hugging_face.py
├── type.py
└── utils.py

visualize.py
├── config.yml (optional)
└── checkpoints/
    ├── training_results.json
    └── evaluation_results.json

examples.py
├── train.py
├── inference.py
└── config.yml
```

---

## File Descriptions

### train.py

**Complete training pipeline**

- `TokenDataset`: PyTorch Dataset class for token data
- `Trainer`: Main trainer class with train/validate/evaluate
- Data preparation & splitting
- Checkpoint saving & early stopping
- Comprehensive metrics calculation

**Key Classes:**

- `TokenDataset(Dataset)` - ~50 lines
- `Trainer` - ~400 lines

**Usage:**

```bash
python train.py --dataset data.csv --config config.yml
```

---

### model.py

**IndoBERT token classification model**

- `IndoBERTForTokenClassification(nn.Module)` class
- BERT encoder + Linear classification head
- Forward pass with optional loss calculation
- Feature extraction utilities
- BERT freezing support for transfer learning

**Key Methods:**

- `forward()` - Main inference
- `get_embedding_features()` - Extract BERT representations
- `freeze_bert_encoder()` - Control fine-tuning strategy

---

### inference.py

**Model inference & prediction**

- `Inference` class for loaded model prediction
- Single token and batch prediction
- Confidence scores & top-k results
- Command-line interface

**Usage:**

```bash
# Single prediction
python inference.py --checkpoint checkpoints/model_epoch_1.pt --token "0"

# Batch prediction
python inference.py --checkpoint checkpoints/model_epoch_1.pt --tokens "0" "1" "3"
```

---

### visualize.py

**Results visualization**

- `plot_training_curves()` - Show training progress
- `print_evaluation_report()` - Test set metrics
- `print_config()` - Configuration details
- Table-formatted output

**Usage:**

```bash
python visualize.py --checkpoint_dir ./checkpoints --all
python visualize.py --training  # Only training results
python visualize.py --evaluation  # Only test results
```

---

### config.yml

**Training configuration file**

Sections:

- `model` - Model settings (name, hidden_size, freeze_bert)
- `training` - Hyperparameters (batch_size, lr, epochs)
- `data` - Data split ratios & max_length
- `output` - Checkpoint & log directories
- `experiment` - Metadata

---

### requirements.txt

**Python dependencies**

```
torch>=2.0.0
transformers>=4.30.0
pandas>=1.5.0
scikit-learn>=1.2.0
pyyaml>=6.0
tqdm>=4.65.0
huggingface-hub>=0.16.0
numpy>=1.23.0
```

---

### quickstart.sh

**Automated setup & training**

Runs:

1. Dependency installation
2. Data exploration
3. Embedding test
4. Model training
5. Results summary

**Usage:**

```bash
bash quickstart.sh /path/to/data.csv
```

---

### examples.py

**Programmatic usage examples**

Includes:

1. Simple training
2. Custom config training
3. Transfer learning (frozen BERT)
4. Batch inference
5. Load & analyze results
6. Compare model checkpoints

**Usage:**

```bash
python examples.py 1  # Run example 1
python examples.py 4  # Run example 4
```

---

### README.md

**Quick reference guide**

- Setup instructions
- Usage examples
- Model architecture diagram
- Configuration guide
- Features & extensions

---

### TRAINING_GUIDE.md

**Comprehensive training documentation**

- Overview & architecture
- Component descriptions
- Quick start guide
- Detailed workflow instructions
- Configuration guide with tips
- Common issues & solutions
- Performance expectations
- Next steps

---

## Data Flow

```
Data (CSV)
    ↓ data.py
Loaded DataFrame
    ↓ Data Splitting (train: 70%, val: 10%, test: 20%)
    ↓
HuggingFace.download() ← hugging_face.py
    ↓
Tokenizer & Model
    ↓ Data Tokenization
TokenDataset Instances
    ↓ DataLoader (batching)
Batch Tensors
    ↓
IndoBERTForTokenClassification ← model.py
    ↓
Logits & Loss
    ↓ Backward pass
Gradients
    ↓ Optimizer (AdamW)
Updated Weights
    ↓
checkpoints/*.pt (Checkpoint Saving)
    ↓
visualize.py (Results Visualization)
```

---

## Typical Workflow

```
1. Setup
   ├── bash quickstart.sh data.csv
   └── OR manual: pip install -r requirements.txt

2. Data Exploration (Optional)
   └── python main.py --mode explore --dataset_file data.csv

3. Training
   ├── python train.py --dataset data.csv --config config.yml
   └── OR customize: Edit config.yml first

4. Results Visualization
   ├── python visualize.py --all
   └── OR check JSON files directly

5. Inference
   ├── python inference.py --checkpoint checkpoints/model_epoch_1.pt --token "0"
   └── For multiple: --tokens "0" "1" "3"

6. Analysis (Optional)
   └── python examples.py 5  # Load & analyze results
```

---

## Output Files Location

After training, you'll find:

```
checkpoints/
├── model_epoch_1.pt              # Best model from epoch 1
├── model_epoch_2.pt              # Best model from epoch 2
├── training_results.json         # Epoch-wise metrics
└── evaluation_results.json       # Test set metrics

logs/
└── run_2024-04-06_12-34-56.log  # Execution log

hugging_face/
└── indobenchmark/
    └── indobert-base-p1/         # Cached model files
        ├── config.json
        ├── tokenizer_config.json
        ├── vocab.txt
        └── model weights...
```

---

## Key Features Added

✅ **Complete Training Pipeline**

- Train/Val/Test split
- Batch data loading
- Checkpoint saving
- Early stopping
- Comprehensive metrics

✅ **Flexible Model**

- Full fine-tuning support
- Transfer learning (freeze BERT)
- Feature extraction

✅ **Inference Ready**

- Single & batch prediction
- Confidence scores
- Top-k results

✅ **Results Tracking**

- Training curves
- Evaluation metrics
- JSON export

✅ **Documentation**

- Quick start
- Comprehensive guide
- Code examples
- File organization

✅ **Utilities**

- Data exploration tool
- Results visualization
- Automated setup
- Example scripts

---

## Statistics

| Metric          | Value       |
| --------------- | ----------- |
| Total Files     | 17          |
| Python Code     | ~2000 lines |
| Documentation   | ~2000 lines |
| Config Files    | 1           |
| Shell Scripts   | 1           |
| Example Scripts | 1           |

---

**Version**: 1.0  
**Last Updated**: April 6, 2026  
**Status**: Production Ready
