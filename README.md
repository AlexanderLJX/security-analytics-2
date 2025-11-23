# ICT3214 Security Analytics - Coursework 2
# Email Phishing Detection using ML/AI Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlexanderLJX/security-analytics-2/blob/main/ICT3214_Phishing_Detection_Demo.ipynb)
[![Models](https://img.shields.io/badge/Models-3-blue)](https://github.com/AlexanderLJX/security-analytics-2)
[![Dataset](https://img.shields.io/badge/Dataset-Enron%2029K-orange)](https://github.com/AlexanderLJX/security-analytics-2)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow)](https://github.com/AlexanderLJX/security-analytics-2)
[![LLM Model](https://img.shields.io/badge/HuggingFace-LLM--GRPO-green)](https://huggingface.co/AlexanderLJX/phishing-detection-qwen3-grpo)

---

## Overview

This project implements and compares **three machine learning approaches** for detecting phishing emails. The models are trained on the **Enron Email Corpus** (29,767 labeled emails) to identify phishing attempts with high accuracy.

### Models Implemented

| Model | Accuracy | F1-Score | Training Time | GPU Required |
|-------|----------|----------|---------------|--------------|
| **Random Forest** | 81.6% | 79.6% | ~4 seconds | No |
| **XGBoost** | 89.2% | 88.5% | ~3 minutes | No |
| **LLM-GRPO** | 99.0% | 99.0% | ~2-4 hours | Yes (15GB VRAM) |

**Pre-trained LLM Model:** [AlexanderLJX/phishing-detection-qwen3-grpo](https://huggingface.co/AlexanderLJX/phishing-detection-qwen3-grpo)

---

## Table of Contents

1. [Quick Start - Google Colab](#quick-start---google-colab)
2. [Project Structure](#project-structure)
3. [Dataset Information](#dataset-information)
4. [Environment Setup](#environment-setup)
5. [Data Processing](#data-processing)
6. [Model Training & Evaluation](#model-training--evaluation)
7. [Reproducing Results](#reproducing-results)
8. [API Usage](#api-usage)
9. [Troubleshooting](#troubleshooting)
10. [References](#references)

---

## Quick Start - Google Colab

The fastest way to reproduce all results is using our Google Colab notebook:

### **[Open Demo Notebook in Google Colab](https://colab.research.google.com/github/AlexanderLJX/security-analytics-2/blob/main/ICT3214_Phishing_Detection_Demo.ipynb)**

### Colab Notebook Features

1. **Automatic Environment Setup** - All dependencies installed automatically
2. **Dataset Auto-Download** - Enron.csv cloned from GitHub
3. **Train All 3 Models** - Random Forest, XGBoost, and LLM-GRPO evaluation
4. **Real-time Metrics** - Accuracy, Precision, Recall, F1-Score, ROC-AUC
5. **Visualization** - Performance comparison charts
6. **Optional LLM Training** - Train your own LLM-GRPO model from scratch

### Colab Usage Instructions

1. Click the "Open in Colab" badge above
2. Ensure GPU is enabled: `Runtime → Change runtime type → GPU (T4)`
3. Run cells sequentially from top to bottom
4. **For LLM evaluation**: Restart runtime after RF/XGBoost cells (GPU memory constraint)

### Notebook Cell Structure

| Cell | Description | Time |
|------|-------------|------|
| 1-6 | Environment setup & dependency installation | ~5 min |
| 7-9 | Random Forest training & evaluation | ~10 sec |
| 10-12 | XGBoost training & evaluation | ~3 min |
| 13-16 | LLM-GRPO evaluation (pre-trained model) | ~5 min |
| 17-19 | (Optional) Train your own LLM-GRPO | ~2-4 hours |
| 20-24 | Model comparison & visualization | ~30 sec |

---

## Project Structure

```
security-analytics-2/
├── README.md                              # This file - User Manual
├── ICT3214_Phishing_Detection_Demo.ipynb  # Google Colab Demo Notebook
├── Enron.csv                              # Dataset (29,767 emails)
│
├── Random-Forest/                         # Model 1: Random Forest
│   ├── train_rf_phishing.py              # Training script
│   ├── predict_rf_phishing.py            # Prediction interface
│   ├── feature_extraction_rf.py          # Feature engineering (46 features)
│   ├── evaluate_rf_benchmark.py          # Benchmark evaluation
│   ├── api_server_fastapi.py             # REST API server
│   ├── requirements.txt                  # Dependencies
│   └── README.md                         # Model documentation
│
├── XgBoost/                               # Model 2: XGBoost
│   ├── train_text_phishing.py            # Training script
│   ├── predict_phishing.py               # Prediction interface
│   ├── feature_extraction_text.py        # Feature engineering (43 features + 946 polynomial)
│   ├── evaluate_benchmark.py             # Benchmark evaluation
│   ├── api_server_fastapi.py             # REST API server
│   ├── phishing_text_model.joblib        # Pre-trained model
│   ├── metrics_report.json               # Evaluation metrics
│   ├── requirements.txt                  # Dependencies
│   └── README.md                         # Model documentation
│
└── LLM-GRPO/                              # Model 3: LLM with GRPO
    ├── train_phishing_llm_grpo.py        # Full training script
    ├── predict_phishing_llm.py           # Prediction interface
    ├── evaluate_phishing_model_detailed.py  # Detailed evaluation
    ├── api_server_fastapi.py             # REST API server
    ├── requirements_llm.txt              # LLM dependencies
    ├── Enron.csv                         # Dataset copy
    └── README_LLM_GRPO.md                # Comprehensive documentation
```

---

## Dataset Information

### Enron Email Corpus

| Property | Value |
|----------|-------|
| **Total Emails** | 29,767 |
| **Legitimate (Ham)** | 15,791 (53.1%) |
| **Phishing (Spam)** | 13,778 (46.3%) |
| **Format** | CSV |
| **Columns** | `subject`, `body`, `label` |

### Data Split (used by all models)

| Split | Samples | Percentage |
|-------|---------|------------|
| Training | 20,106 | 68% |
| Validation | 3,549 | 12% |
| Test | 5,914 | 20% |

**Random State:** 42 (for reproducibility)

### Label Encoding

| Original Label | Standardized | Binary |
|----------------|--------------|--------|
| `0`, `ham`, `legitimate` | LEGITIMATE | 0 |
| `1`, `spam`, `phishing` | PHISHING | 1 |

---

## Environment Setup

### Option 1: Google Colab (Recommended)

No setup required - all dependencies are installed automatically in the notebook.

### Option 2: Local Installation

#### System Requirements

| Component | Random Forest / XGBoost | LLM-GRPO |
|-----------|------------------------|----------|
| **Python** | 3.8+ | 3.8+ |
| **RAM** | 4GB+ | 16GB+ |
| **GPU** | Not required | NVIDIA GPU, 15GB+ VRAM |
| **CUDA** | Not required | 11.8 or 12.1 |
| **Disk** | 1GB | 20GB+ |

#### Installation Steps

**Step 1: Clone Repository**
```bash
git clone https://github.com/AlexanderLJX/security-analytics-2.git
cd security-analytics-2
```

**Step 2: Create Virtual Environment**
```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**Step 3: Install Dependencies**

For Random Forest:
```bash
cd Random-Forest
pip install -r requirements.txt
```

For XGBoost:
```bash
cd XgBoost
pip install -r requirements.txt
```

For LLM-GRPO:
```bash
cd LLM-GRPO

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install Unsloth and dependencies
pip install unsloth vllm transformers trl datasets pandas scikit-learn

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

---

## Data Processing

### Preprocessing Pipeline

All models follow a consistent preprocessing pipeline:

```python
import pandas as pd
import re

# 1. Load dataset
df = pd.read_csv('Enron.csv')

# 2. Handle missing values
df = df.dropna(subset=['body'])

# 3. Standardize labels
def standardize_label(label):
    label_str = str(label).lower().strip()
    if any(word in label_str for word in ['spam', 'phishing', '1']):
        return "PHISHING"  # or 1
    elif any(word in label_str for word in ['ham', 'legit', '0']):
        return "LEGITIMATE"  # or 0
    return None

df['label_binary'] = df['label'].apply(standardize_label)

# 4. Clean text
def clean_text(text):
    text = str(text)
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()

df['clean_body'] = df['body'].apply(clean_text)

# 5. Train/Test split (consistent across all models)
from sklearn.model_selection import train_test_split

train_val, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label_binary'])
train, val = train_test_split(train_val, test_size=0.15, random_state=42, stratify=train_val['label_binary'])
```

### Feature Extraction

#### Random Forest (46 features)
- URL analysis (count, suspicious TLDs, IP addresses)
- Text statistics (length, word count, exclamation marks)
- Keyword detection (urgency, financial, credentials)
- Pattern matching (PII requests, repeated characters)

#### XGBoost (43 base + 946 polynomial features)
- Subject/body length and entropy
- Suspicious word ratios
- HTML detection
- URL pattern analysis
- Polynomial feature interactions (degree 2)

#### LLM-GRPO (contextual embeddings)
- Full email text (up to 2048 tokens)
- No manual feature engineering
- Learns representations during training

---

## Model Training & Evaluation

### Model 1: Random Forest

```bash
cd Random-Forest
python train_rf_phishing.py
```

**Training Output:**
```
============================================================
RANDOM FOREST PHISHING EMAIL DETECTION
============================================================
[1] Loading and preparing dataset...
   Total samples: 29569
   Phishing ratio: 46.60%

[2] Extracting features...
   Features extracted: 46

[5] Training Random Forest model...
   Training time: 3.65s
   OOB score: 0.8218

[7] Test Set Evaluation:
   Accuracy:  0.8160
   Precision: 0.8243
   Recall:    0.7692
   F1 Score:  0.7958
   ROC-AUC:   0.9015

[9] Saving model...
   Model saved: checkpoints/phishing_detector/rf_phishing_detector.joblib
```

### Model 2: XGBoost

```bash
cd XgBoost
python train_text_phishing.py
```

**Training Output:**
```
============================================================
TEXT-BASED PHISHING EMAIL DETECTION
============================================================
[1] Loading Enron dataset...
   Total samples: 29767

[6] Training model with early stopping...
   Feature count after polynomial interactions: 946
[0]    validation_0-aucpr:0.85508
[500]  validation_0-aucpr:0.94206
[1207] validation_0-aucpr:0.94452

[8] Test set performance:
   ROC-AUC: 0.9547

[10] Classification Report (with optimal threshold):
   Accuracy: 0.8921
   Precision: 0.8815
   Recall: 0.8879
   F1 Score: 0.8847

[12] Saving model...
   Model saved to: phishing_text_model.joblib
```

### Model 3: LLM-GRPO

#### Option A: Use Pre-trained Model (Recommended)

```bash
cd LLM-GRPO
python evaluate_phishing_model_detailed.py
```

The script automatically downloads the pre-trained model from HuggingFace:
`AlexanderLJX/phishing-detection-qwen3-grpo`

**Evaluation Output:**
```
================================================================================
PHISHING DETECTION MODEL - DETAILED EVALUATION
================================================================================
[1/4] Loading model...
Loaded LoRA from: AlexanderLJX/phishing-detection-qwen3-grpo

[3/4] Running evaluation...
Evaluating on 500 samples

[4/4] Computing metrics...
================================================================================
EVALUATION RESULTS
================================================================================
Overall Metrics:
  Accuracy:  0.9899
  Precision: 1.0000
  Recall:    0.9796
  F1 Score:  0.9897
================================================================================
```

#### Option B: Train Your Own Model

```bash
cd LLM-GRPO
python train_phishing_llm_grpo.py
```

**Training Stages:**
1. **Pre-finetuning (SFT)**: Teaches model the output format (~10 min)
2. **GRPO Training**: Reinforcement learning optimization (~2-4 hours)

**Configuration (in train_phishing_llm_grpo.py):**
```python
MAX_SEQ_LENGTH = 4096      # Token limit per email
LORA_RANK = 32             # LoRA adapter rank
PRE_FINETUNE_SAMPLES = 2000
GRPO_MAX_STEPS = 1000      # Training iterations
```

---

## Reproducing Results

### Complete Reproduction via Colab

1. Open: [Colab Notebook](https://colab.research.google.com/github/AlexanderLJX/security-analytics-2/blob/main/ICT3214_Phishing_Detection_Demo.ipynb)

2. Enable GPU: `Runtime → Change runtime type → T4 GPU`

3. Run cells 1-6 (Setup)

4. Run cells 7-9 (Random Forest)
   - Expected: Accuracy ~81.6%, F1 ~79.6%

5. Run cells 10-12 (XGBoost)
   - Expected: Accuracy ~89.2%, F1 ~88.5%

6. **Restart runtime** (required for GPU memory)

7. Run cells 1-6 again (re-setup after restart)

8. Run cells 13-16 (LLM-GRPO evaluation)
   - Expected: Accuracy ~99%, F1 ~99%

9. Run cells 20-24 (Comparison visualization)

### Expected Final Comparison

```
================================================================================
MODEL COMPARISON
================================================================================
        Model  Accuracy  Precision   Recall  F1-Score  ROC-AUC
Random Forest   0.8160     0.8243   0.7692    0.7958   0.9015
      XGBoost   0.8921     0.8815   0.8879    0.8847   0.9547
     LLM-GRPO   0.9899     1.0000   0.9796    0.9897   0.9899
================================================================================
```

### Local Reproduction

```bash
# Clone and setup
git clone https://github.com/AlexanderLJX/security-analytics-2.git
cd security-analytics-2

# Random Forest
cd Random-Forest
pip install -r requirements.txt
python train_rf_phishing.py
# Metrics saved to: checkpoints/phishing_detector/rf_phishing_detector.joblib

# XGBoost
cd ../XgBoost
pip install -r requirements.txt
python train_text_phishing.py
# Metrics saved to: metrics_report.json

# LLM-GRPO (requires GPU)
cd ../LLM-GRPO
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install unsloth vllm transformers trl datasets pandas scikit-learn
python evaluate_phishing_model_detailed.py
```

---

## API Usage

### Starting API Servers

```bash
# Random Forest API (port 8001)
cd Random-Forest
python api_server_fastapi.py

# XGBoost API (port 8002)
cd XgBoost
python api_server_fastapi.py

# LLM-GRPO API (port 8003)
cd LLM-GRPO
python api_server_fastapi.py
```

### API Endpoints

#### Health Check
```bash
curl http://localhost:8001/health
```

#### Single Prediction
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "URGENT: Verify Your Account",
    "body": "Click here to verify: http://suspicious-link.tk"
  }'
```

#### Response Format
```json
{
  "prediction": "PHISHING",
  "confidence": 0.94,
  "reasoning": "Urgency tactics, suspicious URL domain...",
  "recommended_action": "BLOCK"
}
```

### Python API

```python
# LLM-GRPO Example
from predict_phishing_llm import load_model, predict_single_email

model, tokenizer = load_model()

email = """
Subject: Your package delivery failed

Dear Customer,
Your package could not be delivered. Click here to reschedule:
http://delivery-tracking.tk/reschedule
Please confirm your payment details.
"""

result = predict_single_email(model, tokenizer, email)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Reasoning: {result['reasoning']}")
```

---

## Troubleshooting

### Issue 1: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Restart Colab runtime before LLM cells
2. Reduce evaluation samples: `EVAL_SAMPLES = 20`
3. Use 4-bit quantization: `load_in_4bit=True`

### Issue 2: Model Not Found

**Error:**
```
Repository Not Found for url: https://huggingface.co/...
```

**Solution:**
Ensure correct HuggingFace model path:
```python
LORA_PATH = "AlexanderLJX/phishing-detection-qwen3-grpo"
```

### Issue 3: Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'unsloth'
```

**Solution:**
```bash
pip install unsloth vllm transformers trl
```

### Issue 4: Slow Training

**Expected Times:**
- Random Forest: ~5 seconds
- XGBoost: ~3 minutes
- LLM-GRPO: ~2-4 hours (T4 GPU)

**If slower:**
- Check GPU utilization: `nvidia-smi`
- Reduce `GRPO_MAX_STEPS` for testing

### Issue 5: Low LLM Accuracy

**Solutions:**
1. Increase `PRE_FINETUNE_SAMPLES` to 200+
2. Increase `GRPO_MAX_STEPS` to 500+
3. Verify dataset labels are correct

---

## References

### Datasets
- Enron Email Corpus: https://www.cs.cmu.edu/~enron/

### Frameworks
- Scikit-learn: https://scikit-learn.org/
- XGBoost: https://xgboost.readthedocs.io/
- Unsloth: https://github.com/unslothai/unsloth
- Qwen3 Model: https://huggingface.co/Qwen/Qwen3-4B-Base

### Research Papers
1. Breiman, L. (2001). "Random Forests". Machine Learning.
2. Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System". KDD.
3. Shao, Z. et al. (2024). "DeepSeekMath: Pushing the Limits of Mathematical Reasoning". arXiv:2402.03300.

---

## License

This project is developed for **ICT3214 Security Analytics Coursework 2**.

- Random Forest & XGBoost: MIT License
- LLM Implementation: Apache 2.0 (Qwen3 license)
- Unsloth Framework: LGPL-3.0

---

**ICT3214 Security Analytics - Coursework 2**
**Singapore Institute of Technology**

---

*End of User Manual*
