---
title: Nimo's Coder Agent v3
emoji: 🔒
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Nimo's Coder Agent v3 - Security Enhanced

A fine-tuned LLM for code generation and **security vulnerability detection**.

## What's New in v3

- **Security vulnerability detection** - Identifies command injection, SQL injection
- **Trained on 25k+ examples** - CodeAlpaca + Security DPO + CrossVul datasets
- **81% token accuracy** - Improved from 77% in v2

## Try It

1. Paste vulnerable code in the "Code to Review" box
2. Ask "Is this code safe?" or "Review this code for security vulnerabilities"
3. Get security analysis and suggestions

## Example

**Input:**
```python
import os
user_input = input("Enter filename: ")
os.system(f"cat {user_input}")
```

**Ask:** "Is this code safe?"

**v3 Response:** Detects command injection vulnerability and suggests secure alternative.

## Links

- [Model on HuggingFace](https://huggingface.co/CaptainNimo/nimos-coder-agent-v3)
- [GitHub Repository](https://github.com/CaptainNimo/nimos-coder-v3-security)
- [v2 (Previous Version)](https://huggingface.co/CaptainNimo/nimos-coder-agent-v2)

## Training

- **Base Model:** Qwen2.5-Coder-0.5B-Instruct
- **Method:** QLoRA (4-bit quantization + LoRA)
- **Training Time:** 2.8 hours on Google Colab T4 GPU
- **Datasets:** CodeAlpaca-20k, Security DPO, CrossVul
