# Installation Guide - ICT3214 Phishing Detection Demo

This guide explains how to properly set up and run the phishing detection notebook in Google Colab.

## ✅ Fixed Issues

The notebook has been **corrected** to properly install LLM packages for Google Colab Tesla T4 GPU.

### What was wrong before:
- Cell 5 had all LLM installation commands commented out
- Missing Tesla T4-specific version detection
- Would not install unsloth, vllm, or other required packages

### What's correct now:
- ✅ Automatic Tesla T4 detection
- ✅ Correct package versions (vllm==0.9.2, triton==3.2.0 for T4)
- ✅ Proper environment variable setup (UNSLOTH_VLLM_STANDBY=1)
- ✅ All required packages installed automatically

## 📋 Requirements

### For Basic Models (Random Forest & XGBoost):
- No GPU required
- Runs on free Colab tier
- Takes ~5-10 minutes

### For LLM-GRPO Model:
- **GPU**: Tesla T4 (16GB VRAM) - Available on free Colab tier
- **Runtime**: 1-2 hours for full training
- **Installation time**: 5-10 minutes
- **Note**: Pre-computed results are included in the notebook for demo purposes

## 🚀 Running in Google Colab

### Step 1: Setup GPU Runtime
1. Open the notebook in Google Colab
2. Go to `Runtime` → `Change runtime type`
3. Select:
   - Hardware accelerator: **GPU**
   - GPU type: **T4** (recommended)
4. Click **Save**

### Step 2: Run Cells in Order
1. **Cell 3**: Check environment (Colab vs Local)
2. **Cell 4**: Install basic ML packages (pandas, scikit-learn, xgboost)
3. **Cell 5**: ⚠️ **NEW - Install LLM packages** (this was broken before!)
   - Automatically detects Tesla T4
   - Installs correct package versions
   - Takes 5-10 minutes
4. **Cell 6**: Import libraries
5. **Continue**: Run remaining cells for data loading, model training, evaluation

### Step 3: Upload Dataset
When prompted, upload your `Enron.csv` file containing phishing email data.

## 📦 Installed Packages (Cell 5)

### For Tesla T4 GPU:
- `unsloth` - Efficient LLM training framework
- `vllm==0.9.2` - Fast inference (T4-compatible version)
- `triton==3.2.0` - GPU kernels (T4-compatible version)
- `transformers==4.56.2` - HuggingFace library
- `trl==0.22.2` - Transformer Reinforcement Learning
- `bitsandbytes` - 8-bit optimization
- `xformers` - Memory efficient attention

### For Other GPUs (A100, V100, etc.):
- Same packages but with latest versions (vllm==0.10.2, triton latest)

## 🔧 Troubleshooting

### Issue: "Out of Memory" error during LLM training
**Solution**: Reduce batch size or use gradient accumulation in training config

### Issue: Cell 5 installation takes too long
**Solution**: This is normal! LLM packages are large. Wait 5-10 minutes.

### Issue: "No GPU detected"
**Solution**:
1. Check Runtime → Change runtime type → GPU
2. Restart runtime
3. Re-run cells 3-5

### Issue: Package version conflicts
**Solution**: Use `Runtime` → `Restart runtime` and run cells in order

## 📁 Project Structure

```
security-analytics-2/
├── ICT3214_Phishing_Detection_Demo.ipynb  ← Main notebook (CORRECTED)
├── INSTALLATION_GUIDE.md                   ← This file
├── LLM-GRPO/                              ← LLM training code
│   ├── qwen3_(4b)_grpo.py                 ← Original Unsloth example
│   ├── train_phishing_llm_grpo.py         ← Phishing-specific training
│   ├── evaluate_phishing_model.py         ← Evaluation script
│   ├── requirements_llm.txt               ← Package requirements
│   └── phishing_llm_outputs/              ← Trained model checkpoints
└── Enron.csv                              ← Dataset (user-provided)
```

## 📝 Model Comparison

| Model | Training Time | GPU Required | Installation |
|-------|--------------|-------------|--------------|
| Random Forest | 5-10 seconds | No | Cell 4 only |
| XGBoost | 30-60 seconds | No | Cell 4 only |
| LLM-GRPO | 1-2 hours | Yes (T4) | Cells 4 + 5 |

## 🎯 Quick Start (Demo Mode)

If you want to see results without training the LLM:

1. Skip cell 5 (LLM installation)
2. Run all other cells
3. The notebook will use **pre-computed LLM results** from local training
4. You'll still see:
   - Random Forest training & results
   - XGBoost training & results
   - LLM results (from pre-computed metrics)
   - All comparisons and visualizations

## 🔬 Full Training Mode

To train all models including LLM:

1. **Must have Tesla T4 GPU** in Colab
2. Run **all cells including cell 5**
3. Training will take ~1-2 hours total:
   - Random Forest: 5-10 seconds
   - XGBoost: 30-60 seconds
   - LLM-GRPO: 1-2 hours

## 📊 Expected Results

### Random Forest:
- Accuracy: ~87-89%
- Training: <10 seconds
- GPU: Not required

### XGBoost:
- Accuracy: ~89-91%
- Training: 30-60 seconds
- GPU: Not required

### LLM-GRPO:
- Accuracy: ~99.4%
- Training: 1-2 hours
- GPU: **Required** (Tesla T4)

## 🆘 Support

If you encounter issues:
1. Check this guide first
2. Verify GPU is enabled (Runtime → Change runtime type)
3. Try `Runtime` → `Restart runtime` and run all cells again
4. Check the Unsloth documentation: https://docs.unsloth.ai/

## ✨ What Changed

**Original Cell 5** (Broken):
```python
# All installation code was commented out
"""
!pip install torch ...
!pip install unsloth ...
"""
# Nothing actually installed!
```

**New Cell 5** (Fixed):
```python
# Active installation code with Tesla T4 detection
if IN_COLAB:
    # Detect GPU type
    is_t4 = "Tesla T4" in subprocess.check_output(["nvidia-smi"])

    # Install correct versions for T4
    if is_t4:
        !uv pip install unsloth vllm==0.9.2 triton==3.2.0 ...
    else:
        !uv pip install unsloth vllm==0.10.2 triton ...
```

## 📖 Additional Resources

- **Unsloth Documentation**: https://docs.unsloth.ai/
- **GRPO Paper**: Group Relative Policy Optimization
- **Qwen3 Model**: https://huggingface.co/Qwen/Qwen3-4B-Base
- **Original Example**: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb

---

**Last Updated**: 2025-11-21
**Status**: ✅ Installation corrected and tested
**Compatible with**: Google Colab Tesla T4, Free Tier
