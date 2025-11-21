#!/usr/bin/env python3
"""
Installation Verification Script for ICT3214 Phishing Detection Demo
Run this script after installation to verify all packages are correctly installed.
"""

import sys
import subprocess

def check_package(package_name, import_name=None, min_version=None):
    """Check if a package is installed and optionally verify version."""
    if import_name is None:
        import_name = package_name

    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')

        if min_version and version != 'unknown':
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                print(f"❌ {package_name}: v{version} (need >= {min_version})")
                return False

        print(f"✅ {package_name}: v{version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: NOT INSTALLED")
        return False

def check_gpu():
    """Check if GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f} GB)")

            # Check if it's Tesla T4
            if "T4" in gpu_name:
                print("   ℹ️  Tesla T4 detected - optimal for this notebook")
            else:
                print("   ℹ️  Non-T4 GPU - should still work")

            return True
        else:
            print("❌ GPU: Not available (CPU only)")
            print("   ⚠️  LLM training requires GPU")
            return False
    except ImportError:
        print("❌ GPU: Cannot check (torch not installed)")
        return False

def main():
    print("="*70)
    print("INSTALLATION VERIFICATION - ICT3214 Phishing Detection Demo")
    print("="*70)

    print("\n[1] BASIC PACKAGES (Required for all models)")
    print("-"*70)
    basic_packages = [
        ('numpy', None),
        ('pandas', None),
        ('scikit-learn', 'sklearn'),
        ('xgboost', 'xgboost'),
        ('matplotlib', None),
        ('seaborn', None),
        ('tqdm', None),
    ]

    basic_ok = all(check_package(pkg, imp) for pkg, imp in basic_packages)

    print("\n[2] GPU AVAILABILITY (Required for LLM)")
    print("-"*70)
    gpu_ok = check_gpu()

    print("\n[3] LLM PACKAGES (Required for LLM-GRPO training)")
    print("-"*70)
    llm_packages = [
        ('torch', 'torch', '2.0.0'),
        ('transformers', 'transformers', '4.56.0'),
        ('trl', 'trl', '0.22.0'),
        ('peft', 'peft'),
        ('bitsandbytes', 'bitsandbytes'),
    ]

    llm_ok = all(check_package(pkg, imp, ver) if ver else check_package(pkg, imp)
                 for pkg, imp, *ver_tuple in llm_packages
                 for ver in (ver_tuple + [None]))

    # Check Unsloth
    print("\n[4] UNSLOTH FRAMEWORK (Required for LLM-GRPO)")
    print("-"*70)
    try:
        import unsloth
        print(f"✅ unsloth: installed")
        unsloth_ok = True
    except ImportError:
        print("❌ unsloth: NOT INSTALLED")
        print("   Run: pip install unsloth")
        unsloth_ok = False

    # Check VLLM
    try:
        import vllm
        version = getattr(vllm, '__version__', 'unknown')
        print(f"✅ vllm: v{version}")

        if version == '0.9.2':
            print("   ℹ️  Version 0.9.2 - optimized for Tesla T4")
        elif version == '0.10.2':
            print("   ℹ️  Version 0.10.2 - for newer GPUs")

        vllm_ok = True
    except ImportError:
        print("❌ vllm: NOT INSTALLED")
        vllm_ok = False

    # Check environment variable
    print("\n[5] ENVIRONMENT SETTINGS")
    print("-"*70)
    import os
    standby = os.environ.get("UNSLOTH_VLLM_STANDBY", "not set")
    if standby == "1":
        print("✅ UNSLOTH_VLLM_STANDBY: enabled (30% more context)")
    else:
        print("⚠️  UNSLOTH_VLLM_STANDBY: not set")
        print("   Set with: os.environ['UNSLOTH_VLLM_STANDBY'] = '1'")

    # Final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if basic_ok:
        print("✅ Basic ML models (Random Forest, XGBoost): READY")
    else:
        print("❌ Basic ML models: MISSING PACKAGES")
        print("   Run: pip install pandas numpy scikit-learn xgboost matplotlib seaborn")

    if llm_ok and unsloth_ok and vllm_ok and gpu_ok:
        print("✅ LLM-GRPO training: READY")
        print("   You can run full LLM training (~1-2 hours on T4)")
    elif llm_ok and unsloth_ok and vllm_ok and not gpu_ok:
        print("⚠️  LLM-GRPO training: PACKAGES OK, BUT NO GPU")
        print("   Enable GPU in Colab: Runtime → Change runtime type → GPU")
    else:
        print("❌ LLM-GRPO training: MISSING PACKAGES")
        print("   Run cell 5 in the notebook to install LLM packages")

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)

    if basic_ok and llm_ok and unsloth_ok and vllm_ok and gpu_ok:
        print("🎉 All checks passed! You can run the full notebook including LLM training.")
        print("\nTo start:")
        print("1. Upload your Enron.csv dataset when prompted")
        print("2. Run all cells in order")
        print("3. Wait ~1-2 hours for LLM training to complete")
    elif basic_ok:
        print("✅ You can run Random Forest and XGBoost models.")
        print("❌ LLM training requires additional setup (see above).")
        print("\nTo run LLM training:")
        print("1. Enable GPU: Runtime → Change runtime type → GPU → T4")
        print("2. Run cell 5 in the notebook to install LLM packages")
        print("3. Restart runtime and run all cells")
    else:
        print("⚠️  Install basic packages first:")
        print("   pip install -r requirements.txt")
        print("   OR run cell 4 in the notebook")

    print("\n" + "="*70)

if __name__ == "__main__":
    main()
