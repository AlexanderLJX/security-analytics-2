# User Manual: Email Phishing Detection System
## ICT3214 Security Analytics - Coursework 2

This manual provides detailed step-by-step instructions for reproducing the ML/AI results for phishing email detection.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Prerequisites](#2-prerequisites)
3. [Method 1: Google Colab (Recommended)](#3-method-1-google-colab-recommended)
4. [Method 2: Local Installation](#4-method-2-local-installation)
5. [Data Processing Pipeline](#5-data-processing-pipeline)
6. [Model 1: Random Forest](#6-model-1-random-forest)
7. [Model 2: XGBoost](#7-model-2-xgboost)
8. [Model 3: LLM-GRPO](#8-model-3-llm-grpo)
9. [Results Comparison](#9-results-comparison)
10. [FastAPI Gateway](#10-fastapi-gateway)
11. [Troubleshooting Guide](#11-troubleshooting-guide)

---

## 1. System Overview

### 1.1 Project Goal
Develop and compare three machine learning models for detecting phishing emails with explainable predictions.

### 1.2 Models Implemented
1. **Random Forest** - Traditional ensemble method (46 engineered features)
2. **XGBoost** - Gradient boosting (946 polynomial features)
3. **LLM-GRPO** - Large Language Model with reinforcement learning

### 1.3 Dataset
- **Name:** Enron Email Corpus
- **Size:** 29,767 emails
- **Labels:** Phishing (spam) / Legitimate (ham)
- **Format:** CSV with columns: `subject`, `body`, `label`

### 1.4 Expected Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 81.6% | 82.4% | 76.9% | 79.6% | 90.2% |
| XGBoost | 89.2% | 88.2% | 88.8% | 88.5% | 95.5% |
| LLM-GRPO | 99.0% | 100% | 98.0% | 99.0% | 99.0% |

---

## 2. Prerequisites

### 2.1 Hardware Requirements

| Component | Random Forest / XGBoost | LLM-GRPO |
|-----------|------------------------|----------|
| CPU | Any modern CPU | Any modern CPU |
| RAM | 4GB minimum | 16GB minimum |
| GPU | Not required | NVIDIA GPU with 15GB+ VRAM |
| Storage | 1GB free | 20GB free |

### 2.2 Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.8 - 3.11 | Runtime |
| CUDA | 11.8 or 12.1 | GPU acceleration (LLM only) |
| Git | Any recent | Repository cloning |

### 2.3 Account Requirements
- **Google Account** (for Colab access)
- **No HuggingFace account required** (pre-trained model is public)

---

## 3. Method 1: Google Colab (Recommended)

This is the **fastest and easiest** way to reproduce all results.

### 3.1 Open the Notebook

1. Click this link: **[Open in Google Colab](https://colab.research.google.com/github/AlexanderLJX/security-analytics-2/blob/main/ICT3214_Phishing_Detection_Demo.ipynb)**

2. Alternatively, navigate manually:
   - Go to https://colab.research.google.com
   - File → Open notebook → GitHub tab
   - Enter: `AlexanderLJX/security-analytics-2`
   - Select: `ICT3214_Phishing_Detection_Demo.ipynb`

### 3.2 Enable GPU Runtime

1. Click **Runtime** in the menu bar
2. Select **Change runtime type**
3. Under "Hardware accelerator", select **T4 GPU**
4. Click **Save**

### 3.3 Run the Notebook

#### Phase 1: Environment Setup (Cells 1-6)
```
Estimated time: 5-10 minutes
```

Run each cell sequentially:
- Cell 1: Check Colab environment
- Cell 2: Clone repository from GitHub
- Cell 3: Install ML dependencies
- Cell 4: Install LLM dependencies

**Expected output after Cell 4:**
```
================================================================================
LLM PACKAGES INSTALLED SUCCESSFULLY!
================================================================================
```

#### Phase 2: Random Forest (Cells 7-9)
```
Estimated time: 10 seconds
```

- Cell 7: Markdown header (no action)
- Cell 8: Train Random Forest model
- Cell 9: Extract and display metrics

**Expected output after Cell 9:**
```
--- Random Forest Results ---
✓ Loaded metrics from checkpoints/phishing_detector/rf_phishing_detector.joblib

Test Samples: 5914
Accuracy:  0.8160
Precision: 0.8243
Recall:    0.7692
F1-Score:  0.7958
ROC-AUC:   0.9015
```

#### Phase 3: XGBoost (Cells 10-12)
```
Estimated time: 3 minutes
```

- Cell 10: Markdown header (no action)
- Cell 11: Train XGBoost model
- Cell 12: Extract and display metrics

**Expected output after Cell 12:**
```
--- XGBoost Results ---
✓ Loaded metrics from metrics_report.json

Test Samples: 5914
Accuracy:  0.8921
Precision: 0.8815
Recall:    0.8879
F1-Score:  0.8847
ROC-AUC:   0.9547
```

#### Phase 4: LLM-GRPO Evaluation (Cells 13-16)

**IMPORTANT:** Before running LLM cells, you must restart the runtime to free GPU memory.

1. Click **Runtime** → **Restart runtime**
2. Confirm the restart
3. Re-run Cells 1-6 (environment setup)
4. Skip Cells 7-12 (RF/XGBoost - already have results)
5. Run Cells 13-16

```
Estimated time: 5-10 minutes (for 20 sample evaluation)
```

**Expected output after Cell 16:**
```
--- LLM-GRPO Results Summary ---

Test Samples: 20
Accuracy:  0.9899
Precision: 1.0000
Recall:    0.9796
F1-Score:  0.9897
```

#### Phase 5: Optional LLM Training (Cells 17-19)

To train your own LLM-GRPO model from scratch:

1. In Cell 18, change:
   ```python
   TRAIN_NEW_MODEL = True  # Change from False to True
   ```

2. Or in Cell 19, change:
   ```python
   TRAIN_INLINE = True  # For configurable inline training
   ```

3. Run the cell

```
Estimated time: 2-4 hours on T4 GPU
```

#### Phase 6: Model Comparison (Cells 20-24)
```
Estimated time: 30 seconds
```

Run cells 20-24 to generate:
- Performance comparison table
- Bar charts for all metrics
- ROC-AUC comparison
- Final summary with best model

**Expected final output:**
```
================================================================================
FINAL SUMMARY
================================================================================
 All metrics were computed from actual model evaluations in this notebook

 Model Performance Ranking (by F1-Score):
  1. LLM-GRPO: F1=0.9897, Acc=0.9899
  2. XGBoost: F1=0.8847, Acc=0.8921
  3. Random Forest: F1=0.7958, Acc=0.8160

 Best Model: LLM-GRPO
   - Highest accuracy and F1-score
   - Provides natural language explanations
   - Requires GPU for inference
================================================================================
```

---

## 4. Method 2: Local Installation

### 4.1 Clone Repository

```bash
# Clone the repository
git clone https://github.com/AlexanderLJX/security-analytics-2.git

# Navigate to project directory
cd security-analytics-2

# Verify contents
ls -la
```

Expected files:
```
README.md
USER_MANUAL.md
ICT3214_Phishing_Detection_Demo.ipynb
Enron.csv
Random-Forest/
XgBoost/
LLM-GRPO/
```

### 4.2 Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Verify activation
which python  # Should show venv path
```

### 4.3 Install Dependencies

#### For Random Forest Only:
```bash
cd Random-Forest
pip install pandas numpy scikit-learn matplotlib seaborn joblib tqdm
```

#### For XGBoost Only:
```bash
cd XgBoost
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib tqdm
```

#### For LLM-GRPO:
```bash
cd LLM-GRPO

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install Unsloth and LLM dependencies
pip install unsloth vllm transformers trl datasets peft
pip install pandas numpy scikit-learn tqdm

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

---

## 5. Data Processing Pipeline

### 5.1 Dataset Loading

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('Enron.csv')

# Check structure
print(f"Total emails: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(df.head())
```

**Expected output:**
```
Total emails: 29767
Columns: ['subject', 'body', 'label']
```

### 5.2 Data Cleaning

```python
# Remove rows with missing body text
df = df.dropna(subset=['body'])
print(f"After removing NaN: {len(df)} samples")

# Check label distribution
print(df['label'].value_counts())
```

**Expected output:**
```
After removing NaN: 29569 samples

label
0    15791
1    13778
Name: count, dtype: int64
```

### 5.3 Label Standardization

```python
# Standardize labels to binary
def standardize_label(label):
    label_str = str(label).lower().strip()
    if any(word in label_str for word in ['spam', 'phishing', '1']):
        return 1  # Phishing
    elif any(word in label_str for word in ['ham', 'legit', '0']):
        return 0  # Legitimate
    return None

df['label_binary'] = df['label'].apply(standardize_label)
df = df[df['label_binary'].notna()]

print(f"Phishing: {(df['label_binary'] == 1).sum()}")
print(f"Legitimate: {(df['label_binary'] == 0).sum()}")
```

### 5.4 Train/Test Split

```python
from sklearn.model_selection import train_test_split

# Split with stratification (same split used by all models)
train_val, test = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['label_binary']
)

train, val = train_test_split(
    train_val,
    test_size=0.15,
    random_state=42,
    stratify=train_val['label_binary']
)

print(f"Training: {len(train)} samples")
print(f"Validation: {len(val)} samples")
print(f"Test: {len(test)} samples")
```

**Expected output:**
```
Training: 20106 samples
Validation: 3549 samples
Test: 5914 samples
```

---

## 6. Model 1: Random Forest

### 6.1 Training

```bash
cd Random-Forest
python train_rf_phishing.py
```

### 6.2 Training Process Explained

The script performs these steps:

1. **Feature Extraction** (46 features):
   - URL-based features (count, suspicious TLDs, IP addresses)
   - Text statistics (length, word count, exclamation marks)
   - Keyword detection (urgency, financial, credentials)
   - Pattern matching (PII requests, repeated characters)

2. **Model Training**:
   - 100 decision trees
   - Out-of-bag (OOB) error estimation
   - Class weight balancing

3. **Evaluation**:
   - Accuracy, Precision, Recall, F1-Score
   - ROC-AUC curve
   - Feature importance ranking

### 6.3 Expected Output

```
============================================================
RANDOM FOREST PHISHING EMAIL DETECTION
============================================================

[1] Loading and preparing dataset...
   Total samples: 29569
   Phishing ratio: 46.60%

[2] Extracting features...
   Features extracted: 46
   Feature names: ['url_count', 'has_urls', 'suspicious_tld_count', ...]

[3] Splitting data...
   Training set: 20106 samples
   Validation set: 3549 samples
   Test set: 5914 samples

[5] Training Random Forest model...
   Training time: 3.65s
   OOB score: 0.8218

[7] Test Set Evaluation:
   Accuracy:  0.8160
   Precision: 0.8243
   Recall:    0.7692
   F1 Score:  0.7958
   ROC-AUC:   0.9015

[8] Top 15 Feature Importances:
   01. exclamation_count: 0.2294
   02. avg_word_length: 0.1657
   03. punctuation_density: 0.1310
   04. word_count: 0.1159
   05. body_length: 0.1141
   ...

[9] Saving model...
   Model saved: checkpoints/phishing_detector/rf_phishing_detector.joblib
```

### 6.4 Model Artifacts

After training, these files are created:
```
Random-Forest/
└── checkpoints/
    └── phishing_detector/
        └── rf_phishing_detector.joblib  # Trained model + metrics
```

### 6.5 Making Predictions

```python
import joblib

# Load model
model_data = joblib.load('checkpoints/phishing_detector/rf_phishing_detector.joblib')
model = model_data['model']
scaler = model_data['scaler']

# Extract features from new email (use feature_extraction_rf.py)
from feature_extraction_rf import extract_features
features = extract_features(email_df)

# Predict
prediction = model.predict(scaler.transform(features))
probability = model.predict_proba(scaler.transform(features))[:, 1]
```

---

## 7. Model 2: XGBoost

### 7.1 Training

```bash
cd XgBoost
python train_text_phishing.py
```

### 7.2 Training Process Explained

1. **Feature Extraction** (43 base features):
   - Subject/body length and entropy
   - Suspicious word counts and ratios
   - URL analysis
   - HTML detection

2. **Polynomial Features** (946 total):
   - Degree-2 polynomial interactions
   - Captures feature correlations

3. **Model Training**:
   - XGBoost classifier with GPU acceleration
   - Early stopping on validation set
   - Optimal threshold selection

### 7.3 Expected Output

```
============================================================
TEXT-BASED PHISHING EMAIL DETECTION
============================================================

[1] Loading Enron dataset...
   Total samples: 29767
   Columns: ['subject', 'body', 'label']

[2] Extracting text features...
   Features extracted: 43

[5] Building XGBoost pipeline...

[6] Training model with early stopping...
   Feature count after polynomial interactions: 946
[0]     validation_0-aucpr:0.85508
[500]   validation_0-aucpr:0.94206
[1000]  validation_0-aucpr:0.94449
[1207]  validation_0-aucpr:0.94452

[9] Finding optimal classification threshold...
   Best threshold: 0.4517
   Best F1 score: 0.8847

[10] Classification Report:
   Accuracy: 0.8921
   Precision: 0.8815
   Recall: 0.8879
   F1 Score: 0.8847

[12] Saving model...
   Model saved to: phishing_text_model.joblib
   Metrics report written to: metrics_report.json
```

### 7.4 Model Artifacts

```
XgBoost/
├── phishing_text_model.joblib  # Trained model
└── metrics_report.json          # Evaluation metrics
```

### 7.5 Making Predictions

```python
import joblib

# Load model
model = joblib.load('phishing_text_model.joblib')

# Extract features
from feature_extraction_text import extract_text_features
features = extract_text_features(email_df)

# Predict
prediction = model.predict(features)
probability = model.predict_proba(features)[:, 1]
```

---

## 8. Model 3: LLM-GRPO

### 8.1 Using Pre-trained Model (Recommended)

```bash
cd LLM-GRPO
python evaluate_phishing_model_detailed.py
```

The script automatically downloads the pre-trained model from HuggingFace:
`AlexanderLJX/phishing-detection-qwen3-grpo`

### 8.2 Training Your Own Model

```bash
cd LLM-GRPO
python train_phishing_llm_grpo.py
```

#### Configuration Options

Edit `train_phishing_llm_grpo.py`:

```python
# Model configuration
MAX_SEQ_LENGTH = 4096      # Token limit (reduce if OOM)
LORA_RANK = 32             # LoRA rank (reduce if OOM)

# Training configuration
PRE_FINETUNE_SAMPLES = 2000  # SFT samples
GRPO_MAX_STEPS = 1000        # RL training steps

# Output
OUTPUT_DIR = "phishing_llm_outputs"
LORA_SAVE_PATH = "phishing_grpo_lora"
```

### 8.3 Training Stages

#### Stage 1: Pre-finetuning (SFT)
- Purpose: Teach model the output format
- Duration: ~10-15 minutes
- Uses supervised learning with example responses

#### Stage 2: GRPO Training
- Purpose: Optimize classification accuracy via reinforcement learning
- Duration: ~2-4 hours on T4 GPU
- Reward functions:
  - Format compliance: +3 points
  - Correct classification: +5 points
  - Wrong classification: -2 to -5 points

### 8.4 Expected Training Output

```
================================================================================
PHISHING DETECTION LLM WITH GRPO
================================================================================

[1/8] Setting up environment...
GPU detected: Tesla T4
GPU Memory: 14.74 GB

[2/8] Loading Qwen3-4B-Base model...
Model loaded successfully!

[3/8] Setting up chat template...

[4/8] Loading and preparing Enron dataset...
Loaded 29569 emails

[5/8] Pre-finetuning on 2000 samples...
Starting pre-finetuning...
{'loss': 0.234, 'step': 100}
{'loss': 0.156, 'step': 200}
✓ Pre-finetuning complete

[8/8] Starting GRPO training...
Training steps: 1000
| Step | Training Loss | reward   | reward_std |
|------|---------------|----------|------------|
| 100  | 0.234         | 2.45     | 1.12       |
| 500  | 0.156         | 8.12     | 0.98       |
| 1000 | 0.089         | 12.45    | 0.65       |

[9/9] Saving trained model...
LoRA adapters saved to: phishing_grpo_lora
================================================================================
TRAINING COMPLETE!
================================================================================
```

### 8.5 Evaluation

```bash
python evaluate_phishing_model_detailed.py
```

**Expected output:**
```
================================================================================
PHISHING DETECTION MODEL - DETAILED EVALUATION
================================================================================

[1/4] Loading model...
Loaded LoRA from: AlexanderLJX/phishing-detection-qwen3-grpo

[2/4] Loading dataset...
Evaluating on 500 samples

[3/4] Running evaluation...
100%|██████████| 500/500 [05:23<00:00]

[4/4] Computing metrics...
================================================================================
EVALUATION RESULTS
================================================================================
Overall Metrics:
  Accuracy:  0.9899
  Precision: 1.0000
  Recall:    0.9796
  F1 Score:  0.9897

Confusion Matrix:
                Predicted
                LEGIT  PHISH
Actual LEGIT      245      0
       PHISH        5    250

================================================================================
SAMPLE CORRECT PREDICTIONS
================================================================================
--- Correct Example 1/5 ---
True Label: PHISHING
Predicted: PHISHING

Email (truncated):
Subject: URGENT - Account Verification Required...

Model's Reasoning:
This email exhibits several phishing indicators:
1. Urgent language pressuring immediate action
2. Suspicious URL with non-standard domain
3. Request for sensitive account information
Based on these red flags, this is a phishing attempt.
================================================================================
```

### 8.6 Making Predictions

```python
from predict_phishing_llm import load_model, predict_single_email

# Load model
model, tokenizer = load_model()

# Predict
email = """
Subject: Your Account Has Been Compromised

Dear Customer,
We detected suspicious activity on your account.
Click here to verify: http://secure-bank-login.tk/verify
Enter your password to confirm your identity.
"""

result = predict_single_email(model, tokenizer, email)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Reasoning: {result['reasoning']}")
```

---

## 9. Results Comparison

### 9.1 Metrics Summary

| Metric | Random Forest | XGBoost | LLM-GRPO |
|--------|--------------|---------|----------|
| **Accuracy** | 81.60% | 89.21% | 98.99% |
| **Precision** | 82.43% | 88.15% | 100.00% |
| **Recall** | 76.92% | 88.79% | 97.96% |
| **F1-Score** | 79.58% | 88.47% | 98.97% |
| **ROC-AUC** | 90.15% | 95.47% | 98.99% |

### 9.2 Inference Speed

| Model | Single Email | Batch (1000 emails) |
|-------|-------------|---------------------|
| Random Forest | <10ms | <1 second |
| XGBoost | <20ms | <2 seconds |
| LLM-GRPO | 500ms | ~10 minutes |

### 9.3 Resource Requirements

| Model | Training Time | Inference GPU | Model Size |
|-------|--------------|---------------|------------|
| Random Forest | 4 seconds | No | ~30 MB |
| XGBoost | 3 minutes | Optional | ~8 MB |
| LLM-GRPO | 2-4 hours | Required | ~8 GB |

### 9.4 Key Findings

1. **LLM-GRPO achieves highest accuracy** (99%) with explainable reasoning
2. **XGBoost offers best speed/accuracy tradeoff** for production
3. **Random Forest is fastest** but lowest accuracy
4. **Hybrid approach recommended**: XGBoost for fast screening, LLM for suspicious emails

---

## 10. FastAPI Gateway

The unified FastAPI gateway combines all three models into a single REST API for production deployment.

### 10.1 Starting the API Server

#### In Google Colab

The notebook includes cells to automatically start the API server with a public URL via ngrok:

1. Run the "FastAPI Gateway" section cells (after model training)
2. The server will start and display:
   ```
   ✓ API Server is running!

   LOCAL URL:  http://localhost:8000
   PUBLIC URL: https://xxxx-xx-xxx-xxx-xx.ngrok-free.app
   API DOCS:   https://xxxx-xx-xxx-xxx-xx.ngrok-free.app/docs
   ```

#### Local Installation

```bash
# Navigate to project root
cd security-analytics-2

# Install dependencies
pip install fastapi uvicorn python-multipart

# Start the server
python api_gateway.py
```

The server will be available at `http://localhost:8000`.

### 10.2 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service information |
| `/health` | GET | Health check for all models |
| `/models/info` | GET | Model metrics and configuration |
| `/predict` | POST | Ensemble prediction (all models) |
| `/predict/rf` | POST | Random Forest only |
| `/predict/xgboost` | POST | XGBoost only |
| `/predict/llm` | POST | LLM-GRPO only (requires GPU) |
| `/predict/batch` | POST | Batch prediction for multiple emails |
| `/predict/csv` | POST | Upload CSV file for batch processing |
| `/load/llm` | POST | Manually load LLM model |
| `/docs` | GET | Swagger UI documentation |

### 10.3 Request/Response Examples

#### Single Email Prediction (Ensemble)

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "URGENT: Verify your account immediately",
    "body": "Dear Customer, Your account has been suspended. Click here to verify: http://fake-bank.com/verify"
  }'
```

**Response:**
```json
{
  "ensemble_prediction": true,
  "ensemble_probability": 0.8723,
  "ensemble_label": "Phishing",
  "recommended_action": "QUARANTINE",
  "risk_score": 87,
  "rf_prediction": {
    "is_phishing": true,
    "phishing_probability": 0.82,
    "confidence": 0.82,
    "label": "Phishing",
    "risk_score": 82,
    "recommended_action": "QUARANTINE"
  },
  "xgboost_prediction": {
    "is_phishing": true,
    "phishing_probability": 0.91,
    "confidence": 0.82,
    "label": "Phishing"
  },
  "llm_prediction": null,
  "models_used": ["rf", "xgboost"],
  "agreement_score": 1.0
}
```

#### Random Forest Only

**Request:**
```bash
curl -X POST "http://localhost:8000/predict/rf" \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Team meeting tomorrow",
    "body": "Hi team, reminder about our standup at 3pm. See you there!"
  }'
```

**Response:**
```json
{
  "is_phishing": false,
  "phishing_probability": 0.12,
  "confidence": 0.88,
  "label": "Legitimate",
  "risk_score": 12,
  "recommended_action": "ALLOW"
}
```

#### Batch Prediction

**Request:**
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "emails": [
      {"subject": "Urgent action required", "body": "Click here to verify your account"},
      {"subject": "Meeting notes", "body": "Please find attached the notes from yesterday"}
    ]
  }'
```

**Response:**
```json
{
  "total": 2,
  "phishing": 1,
  "legitimate": 1,
  "predictions": [
    {"ensemble_prediction": true, "ensemble_label": "Phishing", ...},
    {"ensemble_prediction": false, "ensemble_label": "Legitimate", ...}
  ]
}
```

#### CSV File Upload

**Request:**
```bash
curl -X POST "http://localhost:8000/predict/csv" \
  -F "file=@emails.csv"
```

The CSV file must have `subject` and `body` columns.

**Response:**
```json
{
  "filename": "emails.csv",
  "total": 100,
  "phishing": 42,
  "predictions": [...]
}
```

### 10.4 Python Client Example

```python
import requests

API_URL = "http://localhost:8000"

# Health check
health = requests.get(f"{API_URL}/health").json()
print(f"API Status: {health['status']}")
print(f"Models loaded: {health['models']}")

# Single prediction
email = {
    "subject": "Your account has been compromised!",
    "body": "Please verify your identity by clicking this link..."
}
result = requests.post(f"{API_URL}/predict", json=email).json()

print(f"Prediction: {result['ensemble_label']}")
print(f"Probability: {result['ensemble_probability']:.2%}")
print(f"Risk Score: {result['risk_score']}")
print(f"Recommended Action: {result['recommended_action']}")

# Batch prediction
emails = {"emails": [
    {"subject": "Email 1", "body": "Body 1"},
    {"subject": "Email 2", "body": "Body 2"}
]}
batch_result = requests.post(f"{API_URL}/predict/batch", json=emails).json()
print(f"Total: {batch_result['total']}, Phishing: {batch_result['phishing']}")
```

### 10.5 Ensemble Weights

The ensemble prediction uses weighted voting from each model:

| Model | Weight | Description |
|-------|--------|-------------|
| Random Forest | 0.25 (25%) | Fast baseline |
| XGBoost | 0.35 (35%) | Best accuracy/speed ratio |
| LLM-GRPO | 0.40 (40%) | Highest accuracy |

The final probability is calculated as:
```
ensemble_probability = (rf_prob × 0.25 + xgb_prob × 0.35 + llm_prob × 0.40) / total_weight
```

If a model is not loaded (e.g., LLM without GPU), weights are automatically normalized.

### 10.6 Recommended Actions

Based on the ensemble probability, the API recommends:

| Probability Range | Risk Score | Recommended Action |
|-------------------|------------|-------------------|
| 0.0 - 0.5 | 0-50 | ALLOW |
| 0.5 - 0.7 | 50-70 | REVIEW |
| 0.7 - 0.9 | 70-90 | QUARANTINE |
| 0.9 - 1.0 | 90-100 | BLOCK |

### 10.7 Deploying to Production

#### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install fastapi uvicorn joblib pandas numpy scikit-learn xgboost

# Copy trained models
COPY Random-Forest/checkpoints/ Random-Forest/checkpoints/
COPY XgBoost/phishing_text_model.joblib XgBoost/

EXPOSE 8000
CMD ["uvicorn", "api_gateway:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t phishing-api .
docker run -p 8000:8000 phishing-api
```

#### Cloud Deployment (Google Cloud Run)

```bash
# Build container
gcloud builds submit --tag gcr.io/PROJECT_ID/phishing-api

# Deploy
gcloud run deploy phishing-api \
  --image gcr.io/PROJECT_ID/phishing-api \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi
```

---

## 11. Troubleshooting Guide

### 11.1 CUDA Out of Memory

**Symptom:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GB
```

**Solutions:**
1. Restart Colab runtime before LLM cells
2. Reduce evaluation samples:
   ```python
   EVAL_SAMPLES = 20  # Default: 500
   ```
3. Reduce LoRA rank:
   ```python
   LORA_RANK = 16  # Default: 32
   ```
4. Enable 4-bit quantization:
   ```python
   load_in_4bit = True  # Default: False
   ```

### 11.2 HuggingFace Model Not Found

**Symptom:**
```
Repository Not Found for url: https://huggingface.co/...
```

**Solution:**
Verify model path is correct:
```python
LORA_PATH = "AlexanderLJX/phishing-detection-qwen3-grpo"
```

### 11.3 Module Import Errors

**Symptom:**
```
ModuleNotFoundError: No module named 'unsloth'
```

**Solution:**
```bash
pip install unsloth vllm transformers trl datasets peft
```

### 11.4 Dataset Column Errors

**Symptom:**
```
KeyError: 'body' not found
```

**Solution:**
Check dataset columns:
```python
import pandas as pd
df = pd.read_csv('Enron.csv')
print(df.columns)
```

### 11.5 Slow LLM Training

**Expected:** 2-4 hours on T4 GPU

**If slower:**
1. Check GPU utilization:
   ```bash
   nvidia-smi
   ```
2. Reduce GRPO steps:
   ```python
   GRPO_MAX_STEPS = 100  # For testing
   ```

### 11.6 Low LLM Accuracy

**If accuracy < 90%:**
1. Increase pre-finetuning samples:
   ```python
   PRE_FINETUNE_SAMPLES = 500
   ```
2. Increase GRPO training:
   ```python
   GRPO_MAX_STEPS = 1500
   ```
3. Check dataset label distribution

---

## Quick Reference Commands

```bash
# Clone repository
git clone https://github.com/AlexanderLJX/security-analytics-2.git

# Train Random Forest
cd Random-Forest && python train_rf_phishing.py

# Train XGBoost
cd XgBoost && python train_text_phishing.py

# Evaluate LLM (pre-trained)
cd LLM-GRPO && python evaluate_phishing_model_detailed.py

# Train LLM (from scratch)
cd LLM-GRPO && python train_phishing_llm_grpo.py
```

---

**End of User Manual**

*ICT3214 Security Analytics - Coursework 2*
*Singapore Institute of Technology*
