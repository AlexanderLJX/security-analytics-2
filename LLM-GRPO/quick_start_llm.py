#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick Start Script for Phishing Detection LLM with GRPO

This script provides an interactive menu to:
1. Train the model
2. Test predictions
3. Evaluate performance
4. Compare with other models
"""

import os
import sys

def print_banner():
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║          PHISHING DETECTION LLM WITH GRPO (Qwen3-4B)                 ║
    ║                                                                       ║
    ║     Explainable AI for Email Security using Reinforcement Learning   ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)

def check_requirements():
    """Check if required packages are installed"""
    print("\n[1/3] Checking requirements...")

    required_packages = {
        'torch': 'PyTorch',
        'unsloth': 'Unsloth',
        'transformers': 'Transformers',
        'trl': 'TRL',
        'datasets': 'Datasets',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
    }

    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - MISSING")
            missing.append(package)

    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements_llm.txt")
        return False

    return True

def check_gpu():
    """Check if GPU is available"""
    print("\n[2/3] Checking GPU availability...")

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✓ GPU: {gpu_name}")
            print(f"  ✓ Memory: {gpu_memory:.2f} GB")
            return True
        else:
            print("  ✗ No GPU detected")
            print("  ⚠ Training will be very slow on CPU")
            response = input("\nContinue anyway? (y/n): ")
            return response.lower() == 'y'
    except Exception as e:
        print(f"  ✗ Error checking GPU: {e}")
        return False

def check_dataset():
    """Check if dataset exists"""
    print("\n[3/3] Checking dataset...")

    dataset_path = "./Enron.csv"

    if os.path.exists(dataset_path):
        import pandas as pd
        try:
            df = pd.read_csv(dataset_path)
            print(f"  ✓ Dataset found: {len(df)} rows")
            print(f"  ✓ Columns: {', '.join(df.columns.tolist()[:5])}...")
            return True
        except Exception as e:
            print(f"  ✗ Error reading dataset: {e}")
            return False
    else:
        print(f"  ✗ Dataset not found: {dataset_path}")
        print("\n  Please place your Enron.csv file in the current directory")
        return False

def train_model():
    """Run training script"""
    print("\n" + "="*75)
    print("STARTING TRAINING")
    print("="*75)
    print("\nThis will:")
    print("  1. Load Qwen3-4B-Base model")
    print("  2. Pre-finetune on sample data (~10 mins)")
    print("  3. Run GRPO training (~1-3 hours)")
    print("  4. Save trained model")

    response = input("\nContinue with training? (y/n): ")
    if response.lower() != 'y':
        return

    print("\nStarting training...")
    os.system("python train_phishing_llm_grpo.py")

def test_prediction():
    """Test model with example email"""
    print("\n" + "="*75)
    print("TEST PREDICTION")
    print("="*75)

    example_phishing = """
Subject: URGENT: Verify Your Account Now

Dear Customer,

Your account has been flagged for suspicious activity. Please verify your
identity immediately by clicking the link below and entering your password:

http://secure-verify-account.net/login

Failure to verify within 24 hours will result in permanent account suspension.

Thank you,
Security Department
    """

    example_legitimate = """
Subject: Team Meeting Tomorrow

Hi everyone,

Just a reminder that we have our weekly team meeting tomorrow at 2 PM in
Conference Room B. Please review the attached agenda before the meeting.

Let me know if you have any topics to add.

Best,
Sarah
    """

    print("\n1. Test with phishing example")
    print("2. Test with legitimate example")
    print("3. Enter custom email")

    choice = input("\nSelect option (1-3): ")

    if choice == '1':
        email = example_phishing
        print("\n[Using phishing example]")
    elif choice == '2':
        email = example_legitimate
        print("\n[Using legitimate example]")
    elif choice == '3':
        print("\nEnter email text (press Ctrl+D or Ctrl+Z when done):")
        email = sys.stdin.read()
    else:
        print("Invalid choice")
        return

    print(f"\nEmail:\n{email[:200]}...")
    print("\nRunning prediction...")

    os.system(f'python predict_phishing_llm.py --mode single --email "{email}"')

def evaluate_model():
    """Evaluate model on dataset"""
    print("\n" + "="*75)
    print("EVALUATE MODEL")
    print("="*75)

    dataset = input("\nDataset path (default: ./Enron.csv): ").strip()
    if not dataset:
        dataset = "./Enron.csv"

    max_samples = input("Maximum samples to evaluate (default: 100): ").strip()
    if not max_samples:
        max_samples = "100"

    print(f"\nEvaluating on {max_samples} samples from {dataset}...")

    os.system(
        f"python predict_phishing_llm.py "
        f"--mode evaluate "
        f"--file {dataset} "
        f"--max_samples {max_samples} "
        f"--output evaluation_results.csv"
    )

def compare_models():
    """Compare LLM with XGBoost/Random Forest"""
    print("\n" + "="*75)
    print("COMPARE MODELS")
    print("="*75)
    print("\nComparing LLM-GRPO vs XGBoost vs Random Forest")

    # Check if other models exist
    if not os.path.exists("predict_phishing.py"):
        print("\n⚠ XGBoost/Random Forest scripts not found")
        print("This feature requires the other models to be trained first")
        return

    print("\nSelect test email:")
    print("1. Phishing example")
    print("2. Legitimate example")
    print("3. Custom email")

    choice = input("\nChoice (1-3): ")

    example_email = """
Subject: Account Security Alert

Your account requires immediate verification. Click here to confirm:
http://suspicious-link.com/verify
    """

    if choice == '1' or choice == '2':
        email_text = example_email
    else:
        print("\nEnter email:")
        email_text = sys.stdin.read()

    print("\n" + "-"*75)
    print("RESULTS")
    print("-"*75)

    # Run XGBoost
    print("\n[1] XGBoost Prediction:")
    os.system(f'python predict_phishing.py "{email_text}"')

    # Run LLM
    print("\n[2] LLM-GRPO Prediction:")
    os.system(f'python predict_phishing_llm.py --mode single --email "{email_text}"')

    print("\n" + "-"*75)

def main_menu():
    """Display main menu"""
    while True:
        print("\n" + "="*75)
        print("MAIN MENU")
        print("="*75)
        print("\n1. Train Model (GRPO + Enron dataset)")
        print("2. Test Prediction (single email)")
        print("3. Evaluate Model (on dataset)")
        print("4. Compare Models (LLM vs XGBoost/RF)")
        print("5. View Documentation")
        print("6. Exit")

        choice = input("\nSelect option (1-6): ").strip()

        if choice == '1':
            train_model()
        elif choice == '2':
            test_prediction()
        elif choice == '3':
            evaluate_model()
        elif choice == '4':
            compare_models()
        elif choice == '5':
            print("\nOpening README_LLM_GRPO.md...")
            if os.name == 'nt':  # Windows
                os.system("start README_LLM_GRPO.md")
            else:  # Linux/Mac
                os.system("xdg-open README_LLM_GRPO.md || open README_LLM_GRPO.md")
        elif choice == '6':
            print("\nExiting...")
            break
        else:
            print("\n⚠ Invalid option. Please select 1-6.")

def main():
    """Main entry point"""
    print_banner()

    # System checks
    print("\nRunning system checks...\n")

    if not check_requirements():
        print("\n❌ Please install required packages first")
        print("Run: pip install -r requirements_llm.txt")
        sys.exit(1)

    if not check_gpu():
        print("\n❌ GPU check failed")
        sys.exit(1)

    if not check_dataset():
        print("\n⚠ Dataset not found, but you can continue with other options")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    print("\n" + "="*75)
    print("✓ All checks passed!")
    print("="*75)

    # Show main menu
    main_menu()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
