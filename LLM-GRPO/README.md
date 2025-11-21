# LLM-GRPO Phishing Detection

This folder contains all files related to the LLM-based phishing detection model using GRPO (Group Relative Policy Optimization) training.

## Directory Structure

### Main Scripts
- **train_phishing_llm_grpo.py** - Main training script for the phishing detection LLM
- **evaluate_phishing_model.py** - Basic evaluation script with metrics
- **evaluate_phishing_model_detailed.py** - Detailed evaluation showing model reasoning and errors
- **predict_phishing_llm.py** - Script for making predictions on new emails
- **quick_start_llm.py** - Quick start script for testing the model
- **compare_all_models.py** - Compare different model versions

### Configuration
- **config_llm.yaml** - Configuration file for LLM training
- **requirements_llm.txt** - Python dependencies for LLM training

### Setup Scripts
- **setup_and_train.sh** - Automated setup and training script
- **RUN_NOW.sh** - Quick run script

### Reference Implementation
- **qwen3_(4b)_grpo.py** - Reference implementation from Unsloth

### Documentation
- **README_LLM_GRPO.md** - Original LLM GRPO documentation
- **START_HERE.md** - Getting started guide

### Dataset
- **Enron.csv** - Email dataset (29,767 emails) for training and evaluation

### Model Outputs
- **phishing_grpo_lora/** - Trained LoRA adapters (main model)
- **grpo_trainer_lora_model/** - Alternative training output
- **phishing_llm_outputs/** - Prediction outputs

### Training Logs
- **training_COMPLETE.txt** - Successful training completion log
- **training_SUCCESS.txt** - Training success log
- **training_WORKING.txt** - Working training log
- **training_FINAL.txt** - Final training attempt log
- **training_log*.txt** - Various training iteration logs

### Evaluation Results
- **evaluation_results.txt** - Basic evaluation metrics (96% accuracy)
- **evaluation_detailed.txt** - Detailed evaluation with model reasoning

## Model Performance

**Current Model (phishing_grpo_lora/):**
- Accuracy: 96.00%
- Precision: 96.43%
- Recall: 94.74%
- F1 Score: 95.58%

Evaluated on 500 samples from the Enron dataset.

## Training Configuration

- Base Model: Qwen3-4B-Base (unsloth)
- Training Method: GRPO (Group Relative Policy Optimization)
- Max Sequence Length: 2048
- LoRA Rank: 32
- Training Samples: 93 (SFT) + 100 steps (GRPO)
- GPU: RTX 4090 (24GB)

## Quick Start

1. **Activate environment:**
   ```bash
   source .venv_wsl/bin/activate  # or appropriate venv
   ```

2. **Run evaluation:**
   ```bash
   python3 evaluate_phishing_model_detailed.py
   ```

3. **Make predictions:**
   ```bash
   python3 predict_phishing_llm.py
   ```

4. **Train new model:**
   ```bash
   python3 train_phishing_llm_grpo.py
   ```

## Key Differences from XGBoost Approach

This LLM-based approach:
- Uses natural language understanding vs. feature engineering
- Provides reasoning for predictions
- Requires more GPU memory but can handle nuanced phishing patterns
- Achieved 96% accuracy with minimal training

The XGBoost models are in the parent directory for comparison.
