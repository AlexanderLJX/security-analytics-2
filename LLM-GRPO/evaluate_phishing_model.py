#!/usr/bin/env python3
"""
Evaluate the trained phishing detection model
"""

import os
os.environ["HF_DATASETS_DISABLE_MP"] = "1"

from unsloth import FastLanguageModel
import torch
import pandas as pd
import re
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

# Configuration
BASE_MODEL = "unsloth/Qwen3-4B-Base"
LORA_PATH = "phishing_grpo_lora"
DATASET_PATH = "./Enron.csv"
MAX_SEQ_LENGTH = 2048
EVAL_SAMPLES = 500  # Number of samples to evaluate

# Custom tokens
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

print("="*80)
print("PHISHING DETECTION MODEL EVALUATION")
print("="*80)

# Load model
print("\n[1/4] Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
    fast_inference=False,
)

# Setup chat template
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

# Load LoRA weights
from peft import PeftModel
model = PeftModel.from_pretrained(model, LORA_PATH)
print(f"Loaded LoRA from: {LORA_PATH}")

# Load dataset
print("\n[2/4] Loading dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"Loaded {len(df)} emails")

# Standardize labels
def standardize_label(label):
    label_str = str(label).lower().strip()
    if any(word in label_str for word in ['phish', 'spam', '1', 'true', 'yes', 'malicious']):
        return "PHISHING"
    elif any(word in label_str for word in ['ham', 'legit', '0', 'false', 'no', 'normal']):
        return "LEGITIMATE"
    return None

df['standard_label'] = df['label'].apply(standardize_label)
df = df[df['standard_label'].notna()].copy()

# Clean text
def clean_email_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'\s+', ' ', text)
    if len(text) > 5000:
        text = text[:5000]
    return text.strip()

df['clean_content'] = df['body'].apply(clean_email_text)
df = df[df['clean_content'].str.len() > 20].copy()

# Sample for evaluation
eval_df = df.sample(n=min(EVAL_SAMPLES, len(df)), random_state=42)
print(f"Evaluating on {len(eval_df)} samples")

# Evaluate
print("\n[3/4] Running evaluation...")
predictions = []
true_labels = []

match_format = re.compile(
    rf"{SOLUTION_START}(.+?){SOLUTION_END}",
    flags=re.MULTILINE | re.DOTALL
)

model.eval()
with torch.no_grad():
    for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
        email_text = row['clean_content'][:1000]  # Truncate for speed
        true_label = row['standard_label']

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this email:\n\n{email_text}"},
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        ).to("cuda")

        outputs = model.generate(
            inputs,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True,
            top_k=50,
        )

        output_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

        # Extract prediction
        match = match_format.search(output_text)
        if match:
            pred = match.group(1).strip().upper()
            if "PHISHING" in pred:
                predicted_label = "PHISHING"
            elif "LEGITIMATE" in pred:
                predicted_label = "LEGITIMATE"
            else:
                predicted_label = "UNKNOWN"
        else:
            predicted_label = "UNKNOWN"

        predictions.append(predicted_label)
        true_labels.append(true_label)

# Calculate metrics
print("\n[4/4] Computing metrics...")
print("="*80)
print("EVALUATION RESULTS")
print("="*80)

# Filter out UNKNOWN predictions
valid_mask = [p != "UNKNOWN" for p in predictions]
valid_predictions = [p for p, v in zip(predictions, valid_mask) if v]
valid_true_labels = [t for t, v in zip(true_labels, valid_mask) if v]

if len(valid_predictions) > 0:
    accuracy = accuracy_score(valid_true_labels, valid_predictions)
    precision = precision_score(valid_true_labels, valid_predictions, pos_label="PHISHING", zero_division=0)
    recall = recall_score(valid_true_labels, valid_predictions, pos_label="PHISHING", zero_division=0)
    f1 = f1_score(valid_true_labels, valid_predictions, pos_label="PHISHING", zero_division=0)

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    print(f"\nParsing Success Rate: {len(valid_predictions)}/{len(predictions)} ({len(valid_predictions)/len(predictions)*100:.1f}%)")

    print("\nClassification Report:")
    print(classification_report(valid_true_labels, valid_predictions, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(valid_true_labels, valid_predictions, labels=["LEGITIMATE", "PHISHING"])
    print("                Predicted")
    print("                LEGIT  PHISH")
    print(f"Actual LEGIT    {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"       PHISH    {cm[1][0]:5d}  {cm[1][1]:5d}")
else:
    print("ERROR: No valid predictions generated!")

print("="*80)
