#!/bin/bash
# Complete setup and training script for WSL with RTX 4090
# This script will:
# 1. Create virtual environment with UV
# 2. Install all dependencies
# 3. Verify GPU setup
# 4. Run training

set -e  # Exit on error

echo "=============================================================================="
echo "PHISHING DETECTION LLM - RTX 4090 SETUP & TRAINING"
echo "=============================================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# Step 1: Create Virtual Environment with UV
# ============================================================================

echo -e "\n${GREEN}[1/8] Creating virtual environment with UV...${NC}"

if [ -d ".venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Removing...${NC}"
    rm -rf .venv
fi

uv venv .venv
echo -e "${GREEN}✓ Virtual environment created${NC}"

# Activate virtual environment
source .venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# ============================================================================
# Step 2: Check CUDA and GPU
# ============================================================================

echo -e "\n${GREEN}[2/8] Checking CUDA and GPU availability...${NC}"

# Check if nvidia-smi works
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ nvidia-smi found${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${RED}✗ nvidia-smi not found. Please install NVIDIA drivers.${NC}"
    exit 1
fi

# Get CUDA version
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo -e "${GREEN}✓ CUDA version: ${CUDA_VERSION}${NC}"
else
    echo -e "${YELLOW}⚠ nvcc not found, but nvidia-smi works. Continuing...${NC}"
fi

# ============================================================================
# Step 3: Install PyTorch with CUDA Support
# ============================================================================

echo -e "\n${GREEN}[3/8] Installing PyTorch with CUDA 12.1 support...${NC}"

# For RTX 4090, we need CUDA 12.x
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ PyTorch installed with CUDA support${NC}"
else
    echo -e "${RED}✗ PyTorch installation failed${NC}"
    exit 1
fi

# ============================================================================
# Step 4: Install Unsloth and Dependencies
# ============================================================================

echo -e "\n${GREEN}[4/8] Installing Unsloth and dependencies...${NC}"

# Install main dependencies
uv pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
uv pip install --upgrade unsloth

# Install other requirements
uv pip install \
    transformers==4.56.2 \
    trl==0.22.2 \
    peft>=0.7.0 \
    bitsandbytes>=0.41.0 \
    xformers>=0.0.23 \
    datasets>=2.14.0 \
    pandas>=1.5.0 \
    numpy>=1.24.0 \
    scikit-learn>=1.3.0 \
    tqdm>=4.65.0 \
    safetensors>=0.3.0

# Install vLLM (important for fast inference)
echo -e "${YELLOW}Installing vLLM (this may take a few minutes)...${NC}"
uv pip install vllm

# Install triton for RTX 4090 optimization
uv pip install triton

echo -e "${GREEN}✓ All dependencies installed${NC}"

# ============================================================================
# Step 5: Verify Installation
# ============================================================================

echo -e "\n${GREEN}[5/8] Verifying installation...${NC}"

python3 << 'PYTHON_VERIFY'
import sys

packages = {
    'torch': 'PyTorch',
    'unsloth': 'Unsloth',
    'transformers': 'Transformers',
    'trl': 'TRL',
    'datasets': 'Datasets',
    'pandas': 'Pandas',
    'sklearn': 'Scikit-learn',
    'vllm': 'vLLM',
}

print("\nPackage verification:")
print("-" * 50)

all_ok = True
for package, name in packages.items():
    try:
        __import__(package)
        print(f"✓ {name}")
    except ImportError:
        print(f"✗ {name} - MISSING")
        all_ok = False

if not all_ok:
    print("\n❌ Some packages are missing!")
    sys.exit(1)

# Check GPU
import torch
print("\n" + "=" * 50)
print("GPU Information:")
print("=" * 50)
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("❌ GPU not detected!")
    sys.exit(1)

print("=" * 50)
PYTHON_VERIFY

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All packages verified${NC}"
else
    echo -e "${RED}✗ Package verification failed${NC}"
    exit 1
fi

# ============================================================================
# Step 6: Check Dataset
# ============================================================================

echo -e "\n${GREEN}[6/8] Checking Enron dataset...${NC}"

if [ -f "Enron.csv" ]; then
    echo -e "${GREEN}✓ Enron.csv found${NC}"

    # Show dataset info
    python3 << 'PYTHON_DATASET'
import pandas as pd
df = pd.read_csv("Enron.csv")
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {', '.join(df.columns.tolist()[:5])}...")
print(f"\nFirst few rows:")
print(df.head(2))
PYTHON_DATASET

else
    echo -e "${RED}✗ Enron.csv not found!${NC}"
    echo -e "${YELLOW}Please place Enron.csv in the current directory${NC}"
    exit 1
fi

# ============================================================================
# Step 7: Optimize Configuration for RTX 4090
# ============================================================================

echo -e "\n${GREEN}[7/8] Optimizing training config for RTX 4090 (24GB)...${NC}"

# Create optimized config for RTX 4090
cat > train_config_4090.py << 'EOF'
# Optimized configuration for RTX 4090 (24GB VRAM)

# High-performance settings
MAX_SEQ_LENGTH = 4096  # Increased from 2048 - use full context
LORA_RANK = 64  # Increased from 32 - higher quality
DATASET_PATH = "./Enron.csv"
PRE_FINETUNE_SAMPLES = 200  # Increased from 100 - better format learning
GRPO_MAX_STEPS = 1000  # Increased from 500 - better training
OUTPUT_DIR = "phishing_llm_outputs"
LORA_SAVE_PATH = "phishing_grpo_lora"

BASE_MODEL = "unsloth/Qwen3-4B-Base"

REASONING_START = "<start_analysis>"
REASONING_END = "<end_analysis>"
SOLUTION_START = "<CLASSIFICATION>"
SOLUTION_END = "</CLASSIFICATION>"

SYSTEM_PROMPT = f"""You are an expert cybersecurity analyst specializing in phishing email detection.
Analyze the given email carefully and provide your reasoning.
Place your analysis between {REASONING_START} and {REASONING_END}.
Identify phishing indicators such as:
- Suspicious sender addresses or domains
- Urgent or threatening language
- Requests for sensitive information
- Unusual URLs or links
- Grammar and spelling errors
- Spoofed headers or authentication failures
Then, provide your classification between {SOLUTION_START}{SOLUTION_END}.
Respond with either "PHISHING" or "LEGITIMATE"."""

# Training optimizations for RTX 4090
TRAINING_CONFIG = {
    'sft': {
        'per_device_train_batch_size': 2,  # Increased from 1
        'gradient_accumulation_steps': 4,  # Increased for stability
        'num_train_epochs': 3,  # Increased from 2
        'learning_rate': 2e-4,
    },
    'grpo': {
        'per_device_train_batch_size': 2,  # Increased from 1
        'gradient_accumulation_steps': 4,  # Increased from 4
        'num_generations': 8,  # Increased from 4 - better sampling
        'learning_rate': 5e-6,
        'max_steps': GRPO_MAX_STEPS,
    }
}

print("=" * 80)
print("RTX 4090 OPTIMIZED CONFIGURATION")
print("=" * 80)
print(f"Max Sequence Length: {MAX_SEQ_LENGTH}")
print(f"LoRA Rank: {LORA_RANK}")
print(f"Pre-finetune Samples: {PRE_FINETUNE_SAMPLES}")
print(f"GRPO Steps: {GRPO_MAX_STEPS}")
print(f"Num Generations per Prompt: {TRAINING_CONFIG['grpo']['num_generations']}")
print(f"Effective Batch Size (SFT): {TRAINING_CONFIG['sft']['per_device_train_batch_size'] * TRAINING_CONFIG['sft']['gradient_accumulation_steps']}")
print(f"Effective Batch Size (GRPO): {TRAINING_CONFIG['grpo']['per_device_train_batch_size'] * TRAINING_CONFIG['grpo']['gradient_accumulation_steps']}")
print("=" * 80)
print("\nThese settings are optimized for:")
print("  ✓ RTX 4090 (24GB VRAM)")
print("  ✓ Higher quality reasoning")
print("  ✓ Longer context emails")
print("  ✓ Better generalization")
print("=" * 80)
EOF

python3 train_config_4090.py

echo -e "${GREEN}✓ Configuration optimized for RTX 4090${NC}"

# ============================================================================
# Step 8: Start Training
# ============================================================================

echo -e "\n${GREEN}[8/8] Starting training...${NC}"

echo -e "\n${YELLOW}════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}  TRAINING WILL NOW BEGIN                                        ${NC}"
echo -e "${YELLOW}════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}  Expected timeline with RTX 4090:                               ${NC}"
echo -e "${YELLOW}    • Pre-finetuning: 15-20 minutes (200 samples, 3 epochs)      ${NC}"
echo -e "${YELLOW}    • GRPO Training: 2-4 hours (1000 steps, 8 generations)       ${NC}"
echo -e "${YELLOW}    • Total: ~2.5-5 hours                                         ${NC}"
echo -e "${YELLOW}                                                                  ${NC}"
echo -e "${YELLOW}  You can monitor progress via the output below.                 ${NC}"
echo -e "${YELLOW}  Press Ctrl+C to stop training at any time.                     ${NC}"
echo -e "${YELLOW}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Ask for confirmation
read -p "Start training now? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Training cancelled. To start later, run:${NC}"
    echo -e "${GREEN}source .venv/bin/activate${NC}"
    echo -e "${GREEN}python3 train_phishing_llm_grpo.py${NC}"
    exit 0
fi

# Update the training script with optimized settings
python3 << 'PYTHON_UPDATE_CONFIG'
import re

# Read the training script
with open('train_phishing_llm_grpo.py', 'r') as f:
    content = f.read()

# Update configurations for RTX 4090
replacements = {
    'MAX_SEQ_LENGTH = 2048': 'MAX_SEQ_LENGTH = 4096',
    'LORA_RANK = 32': 'LORA_RANK = 64',
    'PRE_FINETUNE_SAMPLES = 100': 'PRE_FINETUNE_SAMPLES = 200',
    'GRPO_MAX_STEPS = 500': 'GRPO_MAX_STEPS = 1000',
    'per_device_train_batch_size = 1,': 'per_device_train_batch_size = 2,',
    'gradient_accumulation_steps = 2,': 'gradient_accumulation_steps = 4,',
    'num_train_epochs = 2,': 'num_train_epochs = 3,',
    'num_generations = 4,': 'num_generations = 8,',
}

for old, new in replacements.items():
    content = content.replace(old, new)

# Write back
with open('train_phishing_llm_grpo.py', 'w') as f:
    f.write(content)

print("✓ Training script updated with RTX 4090 optimizations")
PYTHON_UPDATE_CONFIG

# Run training
echo -e "\n${GREEN}Starting Python training script...${NC}\n"
python3 train_phishing_llm_grpo.py

# ============================================================================
# Training Complete
# ============================================================================

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  ✓ TRAINING COMPLETED SUCCESSFULLY!                             ${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "\n${GREEN}Model saved to: phishing_grpo_lora/${NC}"
    echo -e "\n${GREEN}Next steps:${NC}"
    echo -e "  1. Test the model:"
    echo -e "     ${YELLOW}python3 predict_phishing_llm.py --mode single${NC}"
    echo -e "\n  2. Evaluate on dataset:"
    echo -e "     ${YELLOW}python3 predict_phishing_llm.py --mode evaluate --file Enron.csv --max_samples 500${NC}"
    echo -e "\n  3. Compare with other models:"
    echo -e "     ${YELLOW}python3 compare_all_models.py${NC}"
else
    echo -e "\n${RED}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}  ✗ TRAINING FAILED                                               ${NC}"
    echo -e "${RED}════════════════════════════════════════════════════════════════${NC}"
    echo -e "\n${YELLOW}Check the error messages above for details.${NC}"
    echo -e "${YELLOW}Common issues:${NC}"
    echo -e "  • Out of memory: Reduce MAX_SEQ_LENGTH or LORA_RANK"
    echo -e "  • Dataset error: Check Enron.csv format"
    echo -e "  • CUDA error: Verify GPU drivers"
    exit 1
fi
