# -*- coding: utf-8 -*-
"""
Phishing Detection Inference using trained GRPO LLM
Loads the trained Qwen3-4B model with LoRA adapters for phishing detection
"""

import torch
import pandas as pd
from vllm import SamplingParams
import re
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================

# Must match training configuration
MAX_SEQ_LENGTH = 2048
LORA_RANK = 32
LORA_PATH = "phishing_grpo_lora"

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

# ============================================================================
# LOAD MODEL
# ============================================================================

def load_model():
    """Load the trained phishing detection model"""
    print("Loading Qwen3-4B-Base model with trained LoRA adapters...")

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-4B-Base",
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.9,
    )

    # Setup chat template (same as training)
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

    print(f"Model loaded successfully!")
    print(f"Loading LoRA adapters from: {LORA_PATH}")

    return model, tokenizer

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_single_email(model, tokenizer, email_text, temperature=0.7, max_tokens=1024):
    """
    Predict whether a single email is phishing or legitimate

    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        email_text: The email content to analyze
        temperature: Sampling temperature (lower = more deterministic)
        max_tokens: Maximum tokens to generate

    Returns:
        dict with 'prediction', 'reasoning', 'raw_output', 'confidence'
    """

    # Prepare messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze this email:\n\n{email_text}"},
    ]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    # Generate response
    sampling_params = SamplingParams(
        temperature=temperature,
        top_k=50,
        max_tokens=max_tokens,
    )

    output = model.fast_generate(
        text,
        sampling_params=sampling_params,
        lora_request=model.load_lora(LORA_PATH),
    )[0].outputs[0].text

    # Parse output
    result = parse_model_output(output)
    result['raw_output'] = output

    return result

def parse_model_output(output):
    """
    Parse the model's output to extract prediction and reasoning

    Returns:
        dict with 'prediction', 'reasoning', 'confidence'
    """

    result = {
        'prediction': 'UNKNOWN',
        'reasoning': '',
        'confidence': 0.0,
    }

    # Extract reasoning
    reasoning_pattern = re.compile(
        rf"{REASONING_START}(.*?){REASONING_END}",
        flags=re.DOTALL
    )
    reasoning_match = reasoning_pattern.search(output)
    if reasoning_match:
        result['reasoning'] = reasoning_match.group(1).strip()

    # Extract classification
    classification_pattern = re.compile(
        rf"{SOLUTION_START}(.*?){SOLUTION_END}",
        flags=re.DOTALL
    )
    classification_match = classification_pattern.search(output)
    if classification_match:
        prediction = classification_match.group(1).strip().upper()

        # Standardize prediction
        if 'PHISHING' in prediction or 'PHISH' in prediction:
            result['prediction'] = 'PHISHING'
            result['confidence'] = 0.9
        elif 'LEGITIMATE' in prediction or 'LEGIT' in prediction or 'HAM' in prediction:
            result['prediction'] = 'LEGITIMATE'
            result['confidence'] = 0.9
        else:
            result['prediction'] = prediction
            result['confidence'] = 0.5

    return result

def predict_batch(model, tokenizer, emails, batch_size=4, temperature=0.7):
    """
    Predict on a batch of emails

    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        emails: List of email texts
        batch_size: Number of emails to process at once
        temperature: Sampling temperature

    Returns:
        List of prediction dicts
    """

    results = []

    for i in range(0, len(emails), batch_size):
        batch = emails[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(emails)-1)//batch_size + 1}...")

        for email_text in batch:
            result = predict_single_email(model, tokenizer, email_text, temperature)
            results.append(result)

    return results

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_on_dataset(model, tokenizer, csv_path, content_col='text', label_col='label', max_samples=None):
    """
    Evaluate the model on a labeled dataset

    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        csv_path: Path to CSV file with emails and labels
        content_col: Name of column containing email text
        label_col: Name of column containing labels
        max_samples: Maximum number of samples to evaluate (None = all)

    Returns:
        dict with evaluation metrics and predictions
    """

    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=42)

    print(f"Evaluating on {len(df)} samples...")

    # Standardize labels
    def standardize_label(label):
        label_str = str(label).lower().strip()
        if any(word in label_str for word in ['phish', 'spam', '1', 'true', 'yes']):
            return "PHISHING"
        elif any(word in label_str for word in ['ham', 'legit', '0', 'false', 'no']):
            return "LEGITIMATE"
        return None

    df['true_label'] = df[label_col].apply(standardize_label)
    df = df[df['true_label'].notna()].copy()

    # Make predictions
    predictions = predict_batch(model, tokenizer, df[content_col].tolist())

    # Add predictions to dataframe
    df['predicted_label'] = [p['prediction'] for p in predictions]
    df['reasoning'] = [p['reasoning'] for p in predictions]
    df['confidence'] = [p['confidence'] for p in predictions]

    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    metrics = {
        'accuracy': accuracy_score(df['true_label'], df['predicted_label']),
        'confusion_matrix': confusion_matrix(df['true_label'], df['predicted_label']),
        'classification_report': classification_report(df['true_label'], df['predicted_label']),
    }

    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print(f"\nClassification Report:")
    print(metrics['classification_report'])
    print("="*80)

    return {
        'metrics': metrics,
        'predictions': df,
    }

# ============================================================================
# MAIN INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Phishing Detection using LLM with GRPO')
    parser.add_argument('--mode', type=str, choices=['single', 'batch', 'evaluate'],
                        default='single', help='Prediction mode')
    parser.add_argument('--email', type=str, help='Email text for single prediction')
    parser.add_argument('--file', type=str, help='CSV file for batch prediction or evaluation')
    parser.add_argument('--output', type=str, help='Output CSV file for batch predictions')
    parser.add_argument('--content_col', type=str, default='text', help='Column name for email content')
    parser.add_argument('--label_col', type=str, default='label', help='Column name for labels (evaluation mode)')
    parser.add_argument('--max_samples', type=int, help='Maximum samples to process')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model()

    if args.mode == 'single':
        # Single email prediction
        if not args.email:
            # Use example email
            example_email = """
            Subject: URGENT: Account Verification Required

            Dear Valued Customer,

            Your account has been flagged for suspicious activity. Please verify your identity
            immediately by clicking the link below and entering your password:

            http://secure-bank-verify.com/login

            Failure to verify within 24 hours will result in account suspension.

            Thank you,
            Security Team
            """
            print("No email provided. Using example phishing email:")
            print(example_email)
            email_text = example_email
        else:
            email_text = args.email

        result = predict_single_email(model, tokenizer, email_text, temperature=args.temperature)

        print("\n" + "="*80)
        print("PHISHING DETECTION RESULT")
        print("="*80)
        print(f"\nEmail:\n{email_text}")
        print(f"\nPrediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"\nReasoning:\n{result['reasoning']}")
        print("="*80)

    elif args.mode == 'batch':
        # Batch prediction
        if not args.file:
            print("Error: --file required for batch mode")
            return

        df = pd.read_csv(args.file)
        print(f"Loaded {len(df)} emails from {args.file}")

        if args.max_samples:
            df = df.head(args.max_samples)

        predictions = predict_batch(model, tokenizer, df[args.content_col].tolist(),
                                     temperature=args.temperature)

        df['predicted_label'] = [p['prediction'] for p in predictions]
        df['reasoning'] = [p['reasoning'] for p in predictions]
        df['confidence'] = [p['confidence'] for p in predictions]

        output_file = args.output or args.file.replace('.csv', '_predictions.csv')
        df.to_csv(output_file, index=False)
        print(f"\nPredictions saved to: {output_file}")

        # Print summary
        print("\n" + "="*80)
        print("PREDICTION SUMMARY")
        print("="*80)
        print(f"Total emails: {len(df)}")
        print(f"\nPrediction distribution:")
        print(df['predicted_label'].value_counts())
        print("="*80)

    elif args.mode == 'evaluate':
        # Evaluation mode
        if not args.file:
            print("Error: --file required for evaluation mode")
            return

        results = evaluate_on_dataset(
            model, tokenizer, args.file,
            content_col=args.content_col,
            label_col=args.label_col,
            max_samples=args.max_samples
        )

        if args.output:
            results['predictions'].to_csv(args.output, index=False)
            print(f"\nDetailed results saved to: {args.output}")

if __name__ == "__main__":
    main()
