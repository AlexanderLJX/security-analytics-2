# -*- coding: utf-8 -*-
"""
Phishing Detection LLM with GRPO (Group Relative Policy Optimization)
Based on Qwen3-4B-Base model
Training on Enron dataset for phishing email classification
"""

import os
os.environ["HF_DATASETS_DISABLE_MP"] = "1"

# CRITICAL: Import unsloth FIRST before trl, transformers, peft
from unsloth import FastLanguageModel

import re
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import TextStreamer
from trl import SFTTrainer, SFTConfig, GRPOConfig, GRPOTrainer
from vllm import SamplingParams
import gc

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model and training parameters
MAX_SEQ_LENGTH = 4096  # Reduced from 4096 to save memory
LORA_RANK = 32  # Reduced from 64 to save memory
DATASET_PATH = "./Enron.csv"
PRE_FINETUNE_SAMPLES = 2000  # Reduced from 200 to save time
GRPO_MAX_STEPS = 1000  # Reduced from 1000 for faster testing
OUTPUT_DIR = "phishing_llm_outputs"
LORA_SAVE_PATH = "phishing_grpo_lora"

# Model Selection (choose one):
# Option 1: Base model (recommended for GRPO from scratch)
BASE_MODEL = "unsloth/Qwen3-4B-Base"

# Option 2: Thinking model (has pre-trained reasoning capabilities)
# BASE_MODEL = "unsloth/Qwen3-4B-Thinking-2507"

# Note: If using Thinking model, reduce PRE_FINETUNE_SAMPLES to 50
# as it already has some reasoning capabilities

# Custom tokens for reasoning and solution
REASONING_START = "<start_analysis>"
REASONING_END = "<end_analysis>"
SOLUTION_START = "<CLASSIFICATION>"
SOLUTION_END = "</CLASSIFICATION>"

# System prompt for phishing detection
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

print("="*80)
print("PHISHING DETECTION LLM WITH GRPO")
print("="*80)
print(f"Dataset: {DATASET_PATH}")
print(f"Max Sequence Length: {MAX_SEQ_LENGTH}")
print(f"LoRA Rank: {LORA_RANK}")
print("="*80)

# ============================================================================
# 1. INSTALL AND SETUP
# ============================================================================

print("\n[1/8] Setting up environment...")

# Check if running on GPU
if not torch.cuda.is_available():
    print("WARNING: No GPU detected! Training will be very slow.")
else:
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# 2. LOAD BASE MODEL
# ============================================================================

print("\n[2/8] Loading Qwen3-4B-Base model...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = BASE_MODEL,  # Use configured base model
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit = False,  # False for LoRA 16bit
    fast_inference = False,  # Disable vLLM for SFT to avoid pickling errors
    # max_lora_rank = LORA_RANK,  # Not needed when fast_inference=False
    # gpu_memory_utilization = 0.9,  # Not needed when fast_inference=False
)

print(f"Using model: {BASE_MODEL}")

model = FastLanguageModel.get_peft_model(
    model,
    r = LORA_RANK,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = LORA_RANK * 2,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

print("Model loaded successfully!")

# ============================================================================
# 3. SETUP CHAT TEMPLATE
# ============================================================================

print("\n[3/8] Setting up chat template...")

chat_template = \
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + eos_token }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ '{system_prompt}' + eos_token }}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ message['content'] }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ message['content'] + eos_token }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
    "{% endif %}"

chat_template = chat_template\
    .replace("'{system_prompt}'", f"'{SYSTEM_PROMPT}'")\
    .replace("'{reasoning_start}'", f"'{REASONING_START}'")

tokenizer.chat_template = chat_template

# Ensure EOS token is properly set for TRL/SFTTrainer
# Qwen3 tokenizer should have eos_token, but explicitly ensure it for TRL compatibility
if tokenizer.eos_token is None:
    tokenizer.eos_token = "<|endoftext|>"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# TRL's SFTTrainer expects <EOS_TOKEN> and <PAD_TOKEN> placeholders in vocabulary
# Add them to avoid validation errors
special_tokens_to_add = []
if "<EOS_TOKEN>" not in tokenizer.get_vocab():
    special_tokens_to_add.append("<EOS_TOKEN>")
if "<PAD_TOKEN>" not in tokenizer.get_vocab():
    special_tokens_to_add.append("<PAD_TOKEN>")

if special_tokens_to_add:
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})
    print(f"Added TRL placeholder tokens: {special_tokens_to_add}")

print(f"Tokenizer EOS token: {tokenizer.eos_token}")
print(f"Tokenizer PAD token: {tokenizer.pad_token}")

# Test chat template
test_messages = [
    {"role": "user", "content": "Is this email phishing?"},
    {"role": "assistant", "content": f"{REASONING_START}Analyzing email...{REASONING_END}{SOLUTION_START}PHISHING{SOLUTION_END}"},
]
print("\nChat template test:")
print(tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=False)[:200] + "...")

# ============================================================================
# 4. LOAD AND PREPARE ENRON DATASET
# ============================================================================

print("\n[4/8] Loading and preparing Enron dataset...")

# Load Enron dataset
try:
    enron_df = pd.read_csv(DATASET_PATH)
    print(f"Loaded {len(enron_df)} emails from Enron dataset")
    print(f"Columns: {enron_df.columns.tolist()}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure Enron.csv exists in the current directory")
    raise

# Inspect the dataset structure
print("\nDataset info:")
print(enron_df.head())
print(f"\nDataset shape: {enron_df.shape}")

# Determine label column (common names: 'label', 'Label', 'Spam/Ham', 'class', etc.)
label_col = None
for col in ['label', 'Label', 'Spam/Ham', 'class', 'Class', 'type', 'Type']:
    if col in enron_df.columns:
        label_col = col
        break

if label_col is None:
    # Try to find it by looking at unique values
    for col in enron_df.columns:
        unique_vals = enron_df[col].unique()
        if len(unique_vals) <= 5:  # Likely a label column
            print(f"Potential label column '{col}': {unique_vals}")
            label_col = col
            break

if label_col is None:
    raise ValueError("Could not find label column. Please specify manually.")

print(f"\nUsing label column: '{label_col}'")
print(f"Label distribution:\n{enron_df[label_col].value_counts()}")

# Determine content column (common names: 'text', 'email', 'body', 'message', etc.)
content_col = None
for col in ['text', 'Text', 'email', 'Email', 'body', 'Body', 'message', 'Message', 'content', 'Content']:
    if col in enron_df.columns:
        content_col = col
        break

if content_col is None:
    # Find the longest text column
    text_cols = enron_df.select_dtypes(include=['object']).columns
    if len(text_cols) > 0:
        avg_lengths = {col: enron_df[col].astype(str).str.len().mean() for col in text_cols if col != label_col}
        content_col = max(avg_lengths, key=avg_lengths.get)

if content_col is None:
    raise ValueError("Could not find content column. Please specify manually.")

print(f"Using content column: '{content_col}'")

# Standardize labels to PHISHING/LEGITIMATE
def standardize_label(label):
    label_str = str(label).lower().strip()
    # Phishing indicators
    if any(word in label_str for word in ['phish', 'spam', '1', 'true', 'yes', 'malicious']):
        return "PHISHING"
    # Legitimate indicators
    elif any(word in label_str for word in ['ham', 'legit', '0', 'false', 'no', 'normal']):
        return "LEGITIMATE"
    else:
        return None  # Unknown label

enron_df['standard_label'] = enron_df[label_col].apply(standardize_label)

# Remove unknown labels
before_count = len(enron_df)
enron_df = enron_df[enron_df['standard_label'].notna()].copy()
after_count = len(enron_df)
if before_count != after_count:
    print(f"Removed {before_count - after_count} emails with unknown labels")

print(f"\nStandardized label distribution:")
print(enron_df['standard_label'].value_counts())

# Clean email content
def clean_email_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Truncate very long emails
    if len(text) > 10000:
        text = text[:10000] + "... [truncated]"
    return text.strip()

enron_df['clean_content'] = enron_df[content_col].apply(clean_email_text)

# Remove empty emails
enron_df = enron_df[enron_df['clean_content'].str.len() > 20].copy()

print(f"\nFinal dataset size: {len(enron_df)} emails")

# ============================================================================
# 5. PRE-FINETUNING FOR FORMAT LEARNING
# ============================================================================

print(f"\n[5/8] Pre-finetuning on {PRE_FINETUNE_SAMPLES} samples for format learning...")

# Create pre-finetuning dataset with manual annotations
pre_finetune_df = enron_df.sample(n=min(PRE_FINETUNE_SAMPLES, len(enron_df)), random_state=42).copy()

def create_sample_reasoning(email_text, label):
    """Create sample reasoning for pre-finetuning"""
    reasoning = "Let me analyze this email for phishing indicators. "

    if label == "PHISHING":
        # Add some generic phishing reasoning
        if "urgent" in email_text.lower() or "immediately" in email_text.lower():
            reasoning += "The email uses urgent language to pressure the recipient. "
        if "@" in email_text and "http" in email_text:
            reasoning += "The email contains suspicious links. "
        if "password" in email_text.lower() or "account" in email_text.lower():
            reasoning += "The email requests sensitive account information. "
        reasoning += "Based on these indicators, this appears to be a phishing attempt."
    else:
        reasoning += "The email appears to be from a legitimate sender with normal business communication. "
        reasoning += "No suspicious links or urgent requests for sensitive information detected."

    return reasoning

def format_pre_finetune_sample(row):
    """Format sample for pre-finetuning"""
    email_text = row['clean_content']
    label = row['standard_label']
    reasoning = create_sample_reasoning(email_text, label)

    # Create formatted response
    formatted_response = (
        f"{REASONING_START}{reasoning}{REASONING_END}"
        f"{SOLUTION_START}{label}{SOLUTION_END}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze this email:\n\n{email_text}"},
        {"role": "assistant", "content": formatted_response},
    ]

pre_finetune_df['Messages'] = pre_finetune_df.apply(format_pre_finetune_sample, axis=1)

# Tokenize and filter by length
pre_finetune_df['text'] = tokenizer.apply_chat_template(
    pre_finetune_df['Messages'].values.tolist(),
    tokenize=False
)
pre_finetune_df['token_count'] = pre_finetune_df['text'].apply(
    lambda x: len(tokenizer.encode(x))
)

# Keep only samples that fit in max_seq_length/2
pre_finetune_df = pre_finetune_df[
    pre_finetune_df['token_count'] <= MAX_SEQ_LENGTH / 2
].copy()

print(f"Pre-finetuning dataset: {len(pre_finetune_df)} samples")

# Convert to HuggingFace Dataset
pre_finetune_dataset = Dataset.from_pandas(pre_finetune_df[['text']])

# Run pre-finetuning
sft_trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=pre_finetune_dataset,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=5,
        num_train_epochs=2,
        learning_rate=2e-4,
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=f"{OUTPUT_DIR}/sft",
        report_to="none",
    ),
)

print("Starting pre-finetuning...")
sft_trainer.train()

# Test the model after pre-finetuning
print("\nTesting model after pre-finetuning:")
test_email = pre_finetune_df.iloc[0]['clean_content'][:500]
test_text = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze this email:\n\n{test_email}"},
    ],
    tokenize=False,
    add_generation_prompt=True,
)

test_output = model.generate(
    **tokenizer(test_text, return_tensors="pt").to("cuda"),
    temperature=0.7,
    max_new_tokens=512,
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)

# Clean up
del pre_finetune_dataset, sft_trainer
torch.cuda.empty_cache()
gc.collect()

print("\n[5.5/8] Model ready for GRPO (keeping fast_inference=False)...")

# ============================================================================
# 6. PREPARE GRPO TRAINING DATASET
# ============================================================================

print("\n[6/8] Preparing GRPO training dataset...")

# Use remaining samples for GRPO (excluding pre-finetune samples)
grpo_df = enron_df[~enron_df.index.isin(pre_finetune_df.index)].copy()

# Format for GRPO
def format_grpo_sample(row):
    """Format sample for GRPO training"""
    email_text = row['clean_content']
    label = row['standard_label']

    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this email:\n\n{email_text}"},
        ],
        "answer": label,
    }

grpo_dataset = grpo_df.apply(format_grpo_sample, axis=1).tolist()
grpo_dataset = Dataset.from_list(grpo_dataset)

print(f"GRPO dataset: {len(grpo_dataset)} samples")

# Filter by token length
tokenized = grpo_dataset.map(
    lambda x: {
        "tokens": tokenizer.apply_chat_template(
            x["prompt"],
            add_generation_prompt=True,
            tokenize=True
        )
    },
    batched=True,
    num_proc=1,
)
tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})

maximum_prompt_length = int(np.quantile(tokenized["L"], 0.9))
print(f"Maximum prompt length (90th percentile): {maximum_prompt_length}")

grpo_dataset = grpo_dataset.select(
    np.where(np.array(tokenized["L"]) <= maximum_prompt_length)[0]
)

print(f"Filtered GRPO dataset: {len(grpo_dataset)} samples")

# ============================================================================
# 7. DEFINE REWARD FUNCTIONS
# ============================================================================

print("\n[7/8] Setting up reward functions...")

# Regex to match expected format
solution_end_regex = r"</CLASSIFICATION>[\s]{0,}" + \
    "(?:" + re.escape(tokenizer.eos_token) + ")?"

match_format = re.compile(
    rf"{REASONING_END}.*?"
    rf"{SOLUTION_START}(.+?){solution_end_regex}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL
)

def reward_format_exact(completions, **kwargs):
    """Reward for exact format matching"""
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        if match_format.search(response) is not None:
            score += 3.0
        scores.append(score)
    return scores

def reward_format_approximate(completions, **kwargs):
    """Reward for approximate format matching"""
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]

        # Count tokens (penalize if too many or missing)
        score += 0.5 if response.count(REASONING_END) == 1 else -1.0
        score += 0.5 if response.count(SOLUTION_START) == 1 else -1.0
        score += 0.5 if response.count(SOLUTION_END) == 1 else -1.0

        scores.append(score)
    return scores

def reward_classification_accuracy(prompts, completions, answer, **kwargs):
    """Reward for correct classification"""
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1).strip()
        if (guess := match_format.search(r)) is not None else None
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(-3.0)  # Heavy penalty for no answer
            continue

        # Exact match
        if guess.upper() == true_answer.upper():
            score += 5.0
        # Partial match
        elif guess.upper() in true_answer.upper() or true_answer.upper() in guess.upper():
            score += 2.0
        else:
            # Wrong classification
            # False negative (missing phishing) is worse than false positive
            if true_answer == "PHISHING":
                score -= 5.0  # Heavy penalty for missing phishing
            else:
                score -= 2.0  # Lesser penalty for false positive

        scores.append(score)
    return scores

# Global counters for logging
PRINTED_TIMES = 0
PRINT_EVERY_STEPS = 5

def reward_with_logging(prompts, completions, answer, **kwargs):
    """Logging reward function"""
    global PRINTED_TIMES, PRINT_EVERY_STEPS

    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        question = prompts[0][-1]["content"][:200]
        response = completions[0][0]["content"]
        true_answer = answer[0]

        # Extract prediction
        extracted = match_format.search(response)
        prediction = extracted.group(1).strip() if extracted else "None"

        print("\n" + "="*80)
        print(f"Email Sample:\n{question}...")
        print(f"\nTrue Label: {true_answer}")
        print(f"Predicted: {prediction}")
        print(f"\nFull Response:\n{response[:300]}...")
        print("="*80)

    PRINTED_TIMES += 1
    return [0] * len(completions)  # No reward, just logging

# ============================================================================
# 8. GRPO TRAINING
# ============================================================================

print("\n[8/8] Starting GRPO training...")

max_prompt_length = maximum_prompt_length + 1
max_completion_length = MAX_SEQ_LENGTH - max_prompt_length

# GRPO training configuration (no vLLM needed)
training_args = GRPOConfig(
    temperature=1.0,
    learning_rate=5e-6,
    weight_decay=0.001,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,  # Reduced from 4 to save memory
    num_generations=2,  # Reduced from 4 to save memory
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    max_steps=GRPO_MAX_STEPS,
    save_steps=50,
    report_to="none",
    output_dir=OUTPUT_DIR,
)

# Create GRPO trainer
grpo_trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        reward_format_exact,
        reward_format_approximate,
        reward_classification_accuracy,
        reward_with_logging,
    ],
    args=training_args,
    train_dataset=grpo_dataset,
)

print("Starting GRPO training...")
print(f"Training steps: {GRPO_MAX_STEPS}")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

grpo_trainer.train()

# ============================================================================
# 9. SAVE MODEL
# ============================================================================

print("\n[9/9] Saving trained model...")

# Save LoRA adapters (use PEFT method since fast_inference=False)
model.save_pretrained(LORA_SAVE_PATH)
print(f"LoRA adapters saved to: {LORA_SAVE_PATH}")

# Save tokenizer
tokenizer.save_pretrained(LORA_SAVE_PATH)
print(f"Tokenizer saved to: {LORA_SAVE_PATH}")

# Test final model
print("\n" + "="*80)
print("TESTING FINAL MODEL")
print("="*80)

test_samples = grpo_df.sample(n=3, random_state=42)

for idx, row in test_samples.iterrows():
    email_text = row['clean_content'][:500]
    true_label = row['standard_label']

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze this email:\n\n{email_text}"},
    ]

    # Use standard generation (not vLLM) since fast_inference=False
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        top_k=50,
    )

    output = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

    print(f"\n{'='*80}")
    print(f"Email (truncated):\n{email_text}...")
    print(f"\nTrue Label: {true_label}")
    print(f"\nModel Response:\n{output}")
    print(f"{'='*80}\n")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"Model saved to: {LORA_SAVE_PATH}")
print(f"\nTo use the model:")
print(f"1. Load the base model: unsloth/Qwen3-4B-Base")
print(f"2. Load LoRA adapters from: {LORA_SAVE_PATH}")
print(f"3. Use the chat template defined in this script")
print("="*80)
