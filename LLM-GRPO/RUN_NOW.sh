#!/bin/bash
# 🚀 ONE-COMMAND SETUP AND TRAIN
# Just run: bash RUN_NOW.sh

set -e

cat << "EOF"
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║       PHISHING DETECTION LLM - ONE-COMMAND SETUP (RTX 4090)           ║
║                                                                        ║
║  This script will:                                                     ║
║    ✓ Create virtual environment with UV                               ║
║    ✓ Install all dependencies (PyTorch, Unsloth, vLLM)               ║
║    ✓ Verify GPU and CUDA                                             ║
║    ✓ Optimize config for RTX 4090 (24GB VRAM)                        ║
║    ✓ Train the model (~3-5 hours)                                    ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
EOF

echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "🔧 Step 1/5: Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "   Removing existing .venv..."
    rm -rf .venv
fi
uv venv .venv
source .venv/bin/activate
echo "   ✓ Virtual environment created and activated"

echo ""
echo "📦 Step 2/5: Installing PyTorch with CUDA 12.1..."
uv pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo "   ✓ PyTorch installed"

echo ""
echo "🦥 Step 3/5: Installing Unsloth and dependencies..."
echo "   (This may take 5-10 minutes...)"
uv pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
uv pip install -q transformers==4.56.2 trl==0.22.2 peft bitsandbytes xformers \
    datasets pandas numpy scikit-learn tqdm safetensors
echo "   ✓ Unsloth installed"

echo ""
echo "⚡ Installing vLLM (for fast inference)..."
uv pip install -q vllm triton
echo "   ✓ vLLM installed"

echo ""
echo "🔍 Step 4/5: Verifying installation..."

python3 << 'PYTHON_CHECK'
import sys
import torch

print("\n   Checking packages...")
packages = ['torch', 'unsloth', 'transformers', 'trl', 'datasets', 'pandas', 'sklearn', 'vllm']
for pkg in packages:
    try:
        __import__(pkg)
        print(f"   ✓ {pkg}")
    except ImportError:
        print(f"   ✗ {pkg} - FAILED")
        sys.exit(1)

print("\n   Checking GPU...")
if not torch.cuda.is_available():
    print("   ✗ CUDA not available!")
    sys.exit(1)

gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9

print(f"   ✓ GPU: {gpu_name}")
print(f"   ✓ VRAM: {gpu_mem:.1f} GB")
print(f"   ✓ CUDA: {torch.version.cuda}")

if "4090" not in gpu_name:
    print(f"\n   ⚠ Warning: Expected RTX 4090 but found {gpu_name}")
    print("   Continuing anyway...")

print("\n   ✓ All checks passed!")
PYTHON_CHECK

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Installation verification failed. Please check errors above."
    exit 1
fi

echo ""
echo "⚙️  Step 5/5: Optimizing config for RTX 4090..."

# Update training script for RTX 4090
python3 << 'PYTHON_OPTIMIZE'
with open('train_phishing_llm_grpo.py', 'r') as f:
    content = f.read()

# RTX 4090 optimizations
updates = {
    'MAX_SEQ_LENGTH = 2048': 'MAX_SEQ_LENGTH = 4096',
    'LORA_RANK = 32': 'LORA_RANK = 64',
    'PRE_FINETUNE_SAMPLES = 100': 'PRE_FINETUNE_SAMPLES = 200',
    'GRPO_MAX_STEPS = 500': 'GRPO_MAX_STEPS = 1000',
}

for old, new in updates.items():
    if old in content:
        content = content.replace(old, new)
        print(f"   ✓ Updated: {new}")

# Update batch sizes
content = content.replace(
    'per_device_train_batch_size = 1,',
    'per_device_train_batch_size = 2,'
)
content = content.replace(
    'gradient_accumulation_steps = 2,',
    'gradient_accumulation_steps = 4,',
)
content = content.replace(
    'num_train_epochs = 2,',
    'num_train_epochs = 3,',
)
content = content.replace(
    'num_generations = 4,',
    'num_generations = 8,',
)

with open('train_phishing_llm_grpo.py', 'w') as f:
    f.write(content)

print("\n   ✓ Configuration optimized for RTX 4090!")
print("\n   Settings:")
print("     • Max Sequence Length: 4096 tokens")
print("     • LoRA Rank: 64 (high quality)")
print("     • Pre-finetune Samples: 200")
print("     • GRPO Steps: 1000")
print("     • Generations per Prompt: 8")
print("     • Batch Size: 2")
print("     • Effective Batch Size: 8")
PYTHON_OPTIMIZE

echo ""
echo "✅ Setup complete!"
echo ""

cat << "EOF"
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║                         READY TO TRAIN!                                ║
║                                                                        ║
║  Expected timeline with RTX 4090:                                      ║
║    • Pre-finetuning: 15-20 minutes                                    ║
║    • GRPO Training: 2-4 hours                                         ║
║    • Total: ~2.5-5 hours                                              ║
║                                                                        ║
║  Monitor GPU usage in another terminal:                               ║
║    watch -n 1 nvidia-smi                                              ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
EOF

echo ""
read -p "Start training now? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Training cancelled. To start later:"
    echo "  source .venv/bin/activate"
    echo "  python3 train_phishing_llm_grpo.py"
    echo ""
    exit 0
fi

echo ""
echo "🚀 Starting training..."
echo ""

python3 train_phishing_llm_grpo.py

# Check if training succeeded
if [ $? -eq 0 ]; then
    cat << "EOF"

╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║                   ✓ TRAINING COMPLETED!                                ║
║                                                                        ║
║  Model saved to: phishing_grpo_lora/                                   ║
║                                                                        ║
║  Test your model:                                                      ║
║    python3 predict_phishing_llm.py --mode single                       ║
║                                                                        ║
║  Evaluate performance:                                                 ║
║    python3 predict_phishing_llm.py --mode evaluate \                   ║
║      --file Enron.csv --max_samples 500                                ║
║                                                                        ║
║  Compare with other models:                                            ║
║    python3 compare_all_models.py                                       ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝

EOF
else
    echo ""
    echo "❌ Training failed. Check error messages above."
    echo ""
    exit 1
fi
