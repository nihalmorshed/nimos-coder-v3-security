# Nimo's Coder Agent v3 - Security Enhanced

A security-focused fine-tuned LLM for code generation, vulnerability detection, and secure coding practices.

**This is v3** - an improved version of [nimos-personal-coder-agent](https://github.com/CaptainNimo/nimos-personal-coder-agent) with enhanced security awareness.

## What's New in v3

| Capability | v2 Score | v3 Target | Improvement |
|------------|----------|-----------|-------------|
| Security Review | 40% | 75%+ | Vulnerability detection |
| Error Handling | <10% | 60%+ | Try-catch in generated code |
| General Coding | 80% | 80%+ | Maintained |

## Key Improvements

### 1. Security Vulnerability Detection
```python
# v2 said this was "safe" (WRONG!)
# v3 correctly identifies command injection:

import os
user_input = input("Enter filename: ")
os.system(f"cat {user_input}")  # v3: "VULNERABILITY DETECTED!"
```

### 2. Proper Error Handling
```python
# v2 generated code without error handling
# v3 includes try-catch blocks:

def read_file(path):
    try:
        with open(path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except PermissionError:
        raise PermissionError(f"Cannot read: {path}")
```

### 3. SQL Injection Awareness
```python
# v3 warns about SQL injection:
query = f"SELECT * FROM users WHERE id = {user_id}"  # UNSAFE!

# And suggests parameterized queries:
cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
```

## Training Data

| Dataset | Size | Purpose |
|---------|------|---------|
| CodeAlpaca-20k | 20,000 | General coding ability |
| Code_Vulnerability_Security_DPO | ~5,000 | Secure vs vulnerable pairs |
| CrossVul | ~6,000 | Real vulnerability fixes |
| Custom Error Handling | 50+ | Error handling patterns |

**Total: ~31,000 examples**

## Model Details

- **Base Model:** Qwen2.5-Coder-0.5B-Instruct
- **Fine-tuning:** QLoRA (4-bit quantization + LoRA)
- **Training:** 1 epoch, ~3.5 hours on T4 GPU
- **Parameters:** 494M total, 8.4M trainable (1.7%)

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load model
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
model = PeftModel.from_pretrained(base, "CaptainNimo/nimos-coder-agent-v3")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")

# Generate
prompt = """### Instruction:
Review this code for security vulnerabilities

### Input:
os.system(f"rm {user_input}")

### Response:
"""
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

## Training Approach

### Why Train from Base Model (not continue from v2)?

| Continue from v2 | Train from Base |
|------------------|-----------------|
| Risk of catastrophic forgetting | Balanced learning |
| LoRA stacking complexity | Clean single adapter |
| May overfit to security data | Proper data mixing |

Since security data (~10k) is 50% of original dataset, we train fresh with mixed data.

## Project Structure

```
nimos-coder-v3-security/
├── README.md
├── notebooks/
│   └── train_v3.ipynb       # Training notebook (Colab)
├── src/
│   └── inference.py         # Inference utilities
├── evaluation/
│   ├── test_security.py     # Security test suite
│   └── results.md           # Evaluation results
└── requirements.txt
```

## Evaluation Results

*To be updated after training*

| Test | Status | Notes |
|------|--------|-------|
| Command Injection Detection | Pending | |
| SQL Injection Detection | Pending | |
| XSS Detection | Pending | |
| Error Handling Generation | Pending | |
| General Coding | Pending | |

## Links

- **Model:** [HuggingFace](https://huggingface.co/CaptainNimo/nimos-coder-agent-v3)
- **Demo:** [HuggingFace Spaces](https://huggingface.co/spaces/CaptainNimo/nimos-coder-v3)
- **v2 Project:** [GitHub](https://github.com/CaptainNimo/nimos-personal-coder-agent)

## Training Datasets

- [CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k)
- [CyberNative/Code_Vulnerability_Security_DPO](https://huggingface.co/datasets/CyberNative/Code_Vulnerability_Security_DPO)
- [hitoshura25/crossvul](https://huggingface.co/datasets/hitoshura25/crossvul)

## License

MIT License

## Author

**Nimo** - [GitHub](https://github.com/CaptainNimo) | [HuggingFace](https://huggingface.co/CaptainNimo)

---

*Fine-tuned using QLoRA on Google Colab (free T4 GPU)*
