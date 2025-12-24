# Complete Guide: Fine-Tuning LLMs from Zero to Deployment

A practical, step-by-step guide for fine-tuning language models using free resources. Follow this guide to replicate or adapt the process for your own projects.

---

## Table of Contents

1. [Overview: The Complete Pipeline](#1-overview-the-complete-pipeline)
2. [Prerequisites and Setup](#2-prerequisites-and-setup)
3. [Phase 1: Planning Your Project](#3-phase-1-planning-your-project)
4. [Phase 2: Dataset Selection and Preparation](#4-phase-2-dataset-selection-and-preparation)
5. [Phase 3: Model Selection](#5-phase-3-model-selection)
6. [Phase 4: Setting Up the Training Environment](#6-phase-4-setting-up-the-training-environment)
7. [Phase 5: Writing the Training Code](#7-phase-5-writing-the-training-code)
8. [Phase 6: Running Training](#8-phase-6-running-training)
9. [Phase 7: Testing Your Model](#9-phase-7-testing-your-model)
10. [Phase 8: Uploading to HuggingFace Hub](#10-phase-8-uploading-to-huggingface-hub)
11. [Phase 9: Deploying to HuggingFace Spaces](#11-phase-9-deploying-to-huggingface-spaces)
12. [Troubleshooting Common Issues](#12-troubleshooting-common-issues)
13. [Decision Frameworks](#13-decision-frameworks)
14. [Checklists](#14-checklists)
15. [Resource Links](#15-resource-links)

---

## 1. Overview: The Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     LLM FINE-TUNING PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PLANNING                                                               │
│  ├── Define your goal (what should the model do?)                       │
│  ├── Identify constraints (GPU, time, budget)                           │
│  └── Choose approach (fine-tune vs prompt engineering)                  │
│           │                                                             │
│           ▼                                                             │
│  DATASET PREPARATION                                                    │
│  ├── Find existing datasets OR create your own                          │
│  ├── Clean and format data                                              │
│  └── Convert to training format (Alpaca, ChatML, etc.)                  │
│           │                                                             │
│           ▼                                                             │
│  MODEL SELECTION                                                        │
│  ├── Choose base model (size, license, capabilities)                    │
│  ├── Decide on fine-tuning method (Full, LoRA, QLoRA)                   │
│  └── Verify it fits your hardware                                       │
│           │                                                             │
│           ▼                                                             │
│  TRAINING                                                               │
│  ├── Set up environment (Colab, Kaggle, local)                          │
│  ├── Configure hyperparameters                                          │
│  ├── Run training with checkpointing                                    │
│  └── Monitor for issues                                                 │
│           │                                                             │
│           ▼                                                             │
│  EVALUATION                                                             │
│  ├── Test on held-out examples                                          │
│  ├── Check for common failure modes                                     │
│  └── Iterate if needed                                                  │
│           │                                                             │
│           ▼                                                             │
│  DEPLOYMENT                                                             │
│  ├── Upload model to HuggingFace Hub                                    │
│  ├── Create demo on HuggingFace Spaces                                  │
│  └── Document and share                                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Time estimates for this project:**
- Planning: 1-2 hours
- Dataset preparation: 2-4 hours
- Training setup: 1-2 hours
- Training execution: 2-5 hours (depends on dataset size)
- Evaluation: 1-2 hours
- Deployment: 1-2 hours
- **Total: 1-2 days**

---

## 2. Prerequisites and Setup

### 2.1 Accounts You Need (All Free)

| Account | Purpose | Sign Up Link |
|---------|---------|--------------|
| **HuggingFace** | Model hosting, datasets, Spaces | https://huggingface.co/join |
| **Google Account** | Google Colab for free GPU | https://accounts.google.com |
| **GitHub** | Code hosting (optional) | https://github.com/join |

### 2.2 Setting Up HuggingFace

1. **Create account** at https://huggingface.co/join

2. **Create access token:**
   ```
   Go to: https://huggingface.co/settings/tokens
   Click: "New token"
   Name: "colab-training"
   Type: "Write" (important - need write access!)
   Copy the token and save it somewhere safe
   ```

3. **Accept model licenses** (for gated models):
   ```
   Some models require you to accept terms first.
   Visit the model page and click "Agree" if prompted.
   Example: https://huggingface.co/meta-llama/Llama-2-7b
   ```

### 2.3 Google Colab Setup

1. **Go to** https://colab.research.google.com

2. **Enable GPU:**
   ```
   Menu: Runtime → Change runtime type
   Hardware accelerator: T4 GPU (free tier)
   Click: Save
   ```

3. **Verify GPU:**
   ```python
   !nvidia-smi
   # Should show "Tesla T4" with 15GB memory
   ```

### 2.4 Understanding Free Tier Limits

```
┌─────────────────────────────────────────────────────────────┐
│              GOOGLE COLAB FREE TIER LIMITS                  │
├─────────────────────────────────────────────────────────────┤
│  GPU:              Tesla T4 (16GB VRAM)                     │
│  Session Length:   ~4 hours maximum                         │
│  Idle Timeout:     ~90 minutes                              │
│  Daily Limit:      ~12 hours total GPU time                 │
│  Storage:          Temporary (lost on disconnect)           │
├─────────────────────────────────────────────────────────────┤
│  IMPLICATIONS:                                              │
│  • Training must complete in <4 hours                       │
│  • Save checkpoints frequently to HuggingFace               │
│  • Don't leave notebook idle                                │
│  • Plan for disconnections                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Phase 1: Planning Your Project

### 3.1 Define Your Goal

Ask yourself these questions:

```
1. WHAT should the model do?
   □ Generate code in specific language(s)?
   □ Answer questions about a domain?
   □ Follow specific instructions?
   □ Detect/classify something?

2. WHO is the target user?
   □ Yourself (learning project)?
   □ Developers (tool)?
   □ General public (demo)?

3. WHAT makes it different from base model?
   □ Specialized knowledge?
   □ Specific output format?
   □ Particular behavior (e.g., security awareness)?
```

**Example for this project:**
```
Goal: Code generation with security vulnerability detection
Target: Developers, portfolio showcase
Difference: Identifies security issues that base model misses
```

### 3.2 Assess Your Constraints

```
HARDWARE CONSTRAINTS:
┌─────────────────────────────────────────────────────────────┐
│  Your GPU        │  Max Model Size (QLoRA)  │  Training Time │
├──────────────────┼──────────────────────────┼────────────────┤
│  T4 (16GB)       │  7B parameters           │  2-8 hours     │
│  A100 (40GB)     │  33B parameters          │  1-4 hours     │
│  RTX 3090 (24GB) │  13B parameters          │  2-6 hours     │
│  RTX 4090 (24GB) │  13B parameters          │  1-4 hours     │
└──────────────────┴──────────────────────────┴────────────────┘

TIME CONSTRAINTS:
• Colab free: Train must complete in ~4 hours
• Colab Pro: Up to 24 hours
• Local GPU: No limit

BUDGET CONSTRAINTS:
• Free: T4 on Colab, small models (0.5B-7B)
• Low ($10-50/month): Colab Pro, larger models
• Medium ($50-200): Cloud GPU rental, any model
```

### 3.3 Decide: Fine-Tune or Not?

```
┌─────────────────────────────────────────────────────────────┐
│          DO YOU NEED TO FINE-TUNE?                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  TRY PROMPT ENGINEERING FIRST IF:                           │
│  • Task can be described in instructions                    │
│  • Few-shot examples work well                              │
│  • You don't have training data                             │
│  • Quick iteration is important                             │
│                                                             │
│  FINE-TUNE IF:                                              │
│  • Prompt engineering isn't working                         │
│  • You need consistent specific behavior                    │
│  • You have quality training data                           │
│  • You want to reduce prompt length                         │
│  • You need the model to learn new knowledge                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Phase 2: Dataset Selection and Preparation

### 4.1 Finding Existing Datasets

**Where to look:**

| Source | URL | Best For |
|--------|-----|----------|
| HuggingFace Datasets | https://huggingface.co/datasets | General purpose |
| Kaggle | https://kaggle.com/datasets | Structured data |
| GitHub | Search "dataset" + your topic | Specialized |
| Papers With Code | https://paperswithcode.com/datasets | Research datasets |

**Search strategy:**
```
1. Go to https://huggingface.co/datasets
2. Search for keywords related to your task
3. Filter by:
   - Task type (text-generation, question-answering, etc.)
   - Size (start small for testing)
   - License (make sure you can use it)
4. Check the dataset card for format and quality
```

### 4.2 Dataset Formats

**Common formats you'll encounter:**

```python
# Format 1: Alpaca (Instruction-Input-Output)
{
    "instruction": "Write a function to add two numbers",
    "input": "",  # Optional context
    "output": "def add(a, b):\n    return a + b"
}

# Format 2: ChatML (Messages)
{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a function to add two numbers"},
        {"role": "assistant", "content": "def add(a, b):\n    return a + b"}
    ]
}

# Format 3: Prompt-Completion
{
    "prompt": "Write a function to add two numbers\n",
    "completion": "def add(a, b):\n    return a + b"
}

# Format 4: DPO (Preference pairs)
{
    "prompt": "How do I run a shell command?",
    "chosen": "subprocess.run(['cmd'], shell=False)",  # Good
    "rejected": "os.system(user_input)"  # Bad
}
```

### 4.3 Loading Datasets

```python
from datasets import load_dataset

# Load from HuggingFace Hub
dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")

# Load specific split
train = load_dataset("dataset_name", split="train")
test = load_dataset("dataset_name", split="test")

# Load with streaming (for large datasets)
dataset = load_dataset("bigcode/the-stack", streaming=True)

# Load from local file
dataset = load_dataset("json", data_files="my_data.json")

# Load from CSV
dataset = load_dataset("csv", data_files="my_data.csv")
```

### 4.4 Inspecting Your Dataset

```python
# Basic info
print(f"Size: {len(dataset)}")
print(f"Columns: {dataset.column_names}")
print(f"Features: {dataset.features}")

# View samples
print(dataset[0])  # First example
print(dataset[:5])  # First 5 examples

# Check for issues
print(f"Null values: {dataset.filter(lambda x: x['output'] is None)}")

# Length distribution
lengths = [len(x['output']) for x in dataset]
print(f"Min length: {min(lengths)}")
print(f"Max length: {max(lengths)}")
print(f"Avg length: {sum(lengths)/len(lengths):.0f}")
```

### 4.5 Converting to Training Format

```python
def format_alpaca(example):
    """Convert to Alpaca format string."""
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '')

    if input_text and input_text.strip():
        text = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
    else:
        text = f"""### Instruction:
{instruction}

### Response:
{output}"""

    return {"text": text}

# Apply formatting
formatted_dataset = dataset.map(format_alpaca)

# Verify
print(formatted_dataset[0]['text'])
```

### 4.6 Combining Multiple Datasets

```python
from datasets import concatenate_datasets, Dataset

# Load multiple datasets
dataset1 = load_dataset("dataset1", split="train")
dataset2 = load_dataset("dataset2", split="train")

# Convert both to same format
formatted1 = dataset1.map(format_function1)
formatted2 = dataset2.map(format_function2)

# Combine
combined = concatenate_datasets([formatted1, formatted2])

# Shuffle (IMPORTANT!)
combined = combined.shuffle(seed=42)

print(f"Combined size: {len(combined)}")
```

### 4.7 Creating Your Own Dataset

If you need custom data:

```python
# Option 1: Create from list
my_data = [
    {
        "instruction": "Your instruction",
        "input": "Optional input",
        "output": "Expected output"
    },
    # Add more examples...
]
dataset = Dataset.from_list(my_data)

# Option 2: Create from JSON file
# my_data.json:
# [{"instruction": "...", "input": "...", "output": "..."}, ...]
dataset = load_dataset("json", data_files="my_data.json", split="train")

# Option 3: Create from CSV
# my_data.csv:
# instruction,input,output
# "Do X","","Result"
dataset = load_dataset("csv", data_files="my_data.csv", split="train")
```

### 4.8 Dataset Size Guidelines

```
┌─────────────────────────────────────────────────────────────┐
│              DATASET SIZE RECOMMENDATIONS                   │
├─────────────────────────────────────────────────────────────┤
│  Goal                          │  Recommended Size          │
├────────────────────────────────┼────────────────────────────┤
│  Style/format change           │  100-500 examples          │
│  Domain adaptation             │  1,000-5,000 examples      │
│  New capability                │  5,000-20,000 examples     │
│  Major behavior change         │  20,000+ examples          │
├────────────────────────────────┴────────────────────────────┤
│  RULE OF THUMB:                                             │
│  • Need behavior X in output?                               │
│  • Include examples of X in at least 5-20% of your data     │
│  • Less than 1% = signal too weak, won't learn              │
└─────────────────────────────────────────────────────────────┘

This project:
• Security behavior: 20% of data → Worked well
• Error handling: 0.05% of data → Did not work
```

---

## 5. Phase 3: Model Selection

### 5.1 Choosing a Base Model

**Decision framework:**

```
Step 1: What's your task?
├── Code generation → Choose code-specialized model
├── Chat/conversation → Choose chat-tuned model
├── Domain-specific → Choose model pretrained on that domain
└── General → Choose general-purpose model

Step 2: What size fits your hardware?
├── T4 (16GB) with QLoRA → Up to 7B parameters
├── A100 (40GB) with QLoRA → Up to 33B parameters
└── Consumer GPU (8-24GB) → Up to 13B with QLoRA

Step 3: What license do you need?
├── Commercial use → Apache 2.0, MIT
├── Research only → Check model license
└── Personal → Usually any license OK
```

### 5.2 Recommended Models by Task

**For Code Generation:**

| Model | Size | License | Notes |
|-------|------|---------|-------|
| Qwen2.5-Coder-0.5B | 494M | Apache 2.0 | Great for free tier |
| Qwen2.5-Coder-1.5B | 1.5B | Apache 2.0 | Better quality |
| Qwen2.5-Coder-7B | 7B | Apache 2.0 | Best quality, fits T4 |
| DeepSeek-Coder-6.7B | 6.7B | MIT | Competitive with 7B |
| CodeLlama-7B | 7B | Custom | Meta's code model |
| StarCoder2-7B | 7B | OpenRAIL | Good for many languages |

**For General Chat:**

| Model | Size | License | Notes |
|-------|------|---------|-------|
| Qwen2.5-0.5B-Instruct | 494M | Apache 2.0 | Tiny but capable |
| Llama-3.2-1B | 1B | Llama license | Good balance |
| Mistral-7B-Instruct | 7B | Apache 2.0 | Very capable |
| Llama-3.1-8B-Instruct | 8B | Llama license | State of the art |

### 5.3 Instruct vs Base Models

```
BASE MODEL:
• Pretrained on raw text
• Completes text (not follows instructions)
• Example: "def add(" → "(a, b): return a + b"
• Use when: You have lots of data and want full control

INSTRUCT MODEL:
• Further trained to follow instructions
• Already understands instruction format
• Example: "Write add function" → "def add(a, b):..."
• Use when: You want to build on existing instruction-following
• RECOMMENDED for most fine-tuning projects
```

**Our choice:** `Qwen2.5-Coder-0.5B-Instruct`
- "-Instruct" = already instruction-tuned
- "-Coder" = specialized for code
- "0.5B" = fits easily in free tier

### 5.4 Verifying Model Fits Your GPU

```python
# Quick memory estimate for QLoRA:
def estimate_memory_gb(num_params_billions):
    """Estimate GPU memory needed for QLoRA training."""
    # 4-bit quantized model
    model_memory = num_params_billions * 0.5  # ~0.5 GB per billion params

    # LoRA adapters, gradients, optimizer
    training_overhead = 0.5  # Roughly 0.5 GB

    # Activations (depends on batch size and seq length)
    activations = 1.0  # Rough estimate for batch_size=2

    total = model_memory + training_overhead + activations
    return total

# Examples:
print(f"0.5B model: {estimate_memory_gb(0.5):.1f} GB")  # ~1.5 GB
print(f"7B model: {estimate_memory_gb(7):.1f} GB")      # ~5.0 GB
print(f"13B model: {estimate_memory_gb(13):.1f} GB")    # ~8.5 GB

# T4 has 16GB, so 7B is comfortable, 13B is tight
```

---

## 6. Phase 4: Setting Up the Training Environment

### 6.1 Google Colab Setup

**Step 1: Create new notebook**
```
Go to: https://colab.research.google.com
Click: "New notebook"
```

**Step 2: Enable GPU**
```
Menu: Runtime → Change runtime type
Hardware accelerator: T4 GPU
Click: Save
```

**Step 3: Install dependencies**
```python
# Run this cell first
!pip install -q transformers datasets peft bitsandbytes accelerate trl huggingface_hub
```

**Step 4: Verify setup**
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

**Step 5: Login to HuggingFace**
```python
from huggingface_hub import login

# This will prompt for your token
login()

# Or directly with token (less secure but convenient)
# login(token="hf_xxxxxxxxxxxxxxxxxxxxx")
```

### 6.2 Understanding the Libraries

```
┌─────────────────────────────────────────────────────────────┐
│                    LIBRARY PURPOSES                         │
├─────────────────────────────────────────────────────────────┤
│  transformers     Core library for loading/using models    │
│  datasets         Loading and processing datasets          │
│  peft             LoRA and other efficient fine-tuning     │
│  bitsandbytes     4-bit and 8-bit quantization             │
│  accelerate       Multi-GPU and mixed precision training   │
│  trl              Training utilities (SFTTrainer, etc.)    │
│  huggingface_hub  Upload/download from HuggingFace         │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 Project Configuration

Create a configuration cell at the top of your notebook:

```python
# ============================================================
# CONFIGURATION - MODIFY THESE FOR YOUR PROJECT
# ============================================================

# HuggingFace username (change this!)
HF_USERNAME = "YourUsername"

# Model settings
BASE_MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
OUTPUT_MODEL = "my-finetuned-model"

# Dataset
DATASET_NAME = "sahil2801/CodeAlpaca-20k"

# Training settings
NUM_EPOCHS = 1
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 1024

# LoRA settings
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
```

---

## 7. Phase 5: Writing the Training Code

### 7.1 Complete Training Script Template

Here's a complete, copy-paste ready training script:

```python
# ============================================================
# CELL 1: CONFIGURATION
# ============================================================

HF_USERNAME = "YourUsername"  # CHANGE THIS
BASE_MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
OUTPUT_MODEL = "my-model-name"

# ============================================================
# CELL 2: IMPORTS AND SETUP
# ============================================================

import torch
import time
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login

# Check GPU
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

start_time = time.time()

# ============================================================
# CELL 3: LOGIN
# ============================================================

login()  # Enter your HuggingFace token when prompted

# ============================================================
# CELL 4: LOAD AND PREPARE DATASET
# ============================================================

print("Loading dataset...")
dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
print(f"Dataset size: {len(dataset)}")

def format_example(example):
    """Convert to training format."""
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '')

    if input_text and input_text.strip():
        text = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
    else:
        text = f"""### Instruction:
{instruction}

### Response:
{output}"""

    return {"text": text}

# Format dataset
dataset = dataset.map(format_example)
dataset = dataset.shuffle(seed=42)

print(f"Sample:\n{dataset[0]['text'][:200]}...")

# ============================================================
# CELL 5: LOAD MODEL WITH QUANTIZATION
# ============================================================

print("Loading model...")

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model = prepare_model_for_kbit_training(model)
print(f"Model loaded. Memory: {model.get_memory_footprint() / 1e9:.2f} GB")

# ============================================================
# CELL 6: CONFIGURE LoRA
# ============================================================

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

model = get_peft_model(model, lora_config)

# Print trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# ============================================================
# CELL 7: CONFIGURE TRAINING
# ============================================================

training_args = SFTConfig(
    output_dir=f"./{OUTPUT_MODEL}",

    # Training
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,

    # Learning rate
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",

    # Memory optimization
    gradient_checkpointing=True,
    bf16=True,
    fp16=False,

    # Saving (important for Colab!)
    logging_steps=25,
    save_strategy="steps",
    save_steps=300,
    save_total_limit=3,

    # Push to Hub (saves your work!)
    push_to_hub=True,
    hub_model_id=f"{HF_USERNAME}/{OUTPUT_MODEL}",
    hub_strategy="every_save",

    # Other
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    dataset_text_field="text",
    max_seq_length=1024,
)

# ============================================================
# CELL 8: INITIALIZE TRAINER
# ============================================================

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

print(f"Ready to train on {len(dataset)} examples")
print(f"Estimated steps: {len(dataset) // (2 * 4)}")

# ============================================================
# CELL 9: TRAIN!
# ============================================================

trainer.train()

# ============================================================
# CELL 10: SAVE FINAL MODEL
# ============================================================

print("Saving final model...")
trainer.save_model()
trainer.push_to_hub()

total_time = (time.time() - start_time) / 3600
print(f"\n✓ Training complete in {total_time:.2f} hours")
print(f"Model saved to: huggingface.co/{HF_USERNAME}/{OUTPUT_MODEL}")

# ============================================================
# CELL 11: TEST THE MODEL
# ============================================================

# Switch to inference mode
model.eval()
model.config.use_cache = True

def generate(prompt, max_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test
test_prompt = """### Instruction:
Write a Python function to check if a number is prime

### Response:
"""

print(generate(test_prompt))
```

### 7.2 Key Configuration Explained

```python
# QUANTIZATION CONFIG
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,           # Use 4-bit precision (saves memory)
    bnb_4bit_quant_type="nf4",   # NormalFloat4 (optimized for neural nets)
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in BFloat16
    bnb_4bit_use_double_quant=True,  # Quantize the quantization constants
)
# WHY: Reduces 494M param model from 2GB to 0.3GB

# LORA CONFIG
lora_config = LoraConfig(
    r=16,              # Rank - higher = more capacity, more memory
    lora_alpha=32,     # Scaling factor - usually 2x r
    lora_dropout=0.05, # Regularization
    target_modules=[...],  # Which layers to adapt
)
# WHY: Only train 2-3% of parameters instead of 100%

# TRAINING CONFIG
training_args = SFTConfig(
    num_train_epochs=1,           # Full passes through data
    per_device_train_batch_size=2,  # Examples per forward pass
    gradient_accumulation_steps=4,  # Accumulate before update
    # Effective batch size = 2 * 4 = 8

    learning_rate=2e-4,    # How big each update is
    warmup_ratio=0.03,     # Gradual LR increase at start

    gradient_checkpointing=True,  # Trade compute for memory
    bf16=True,  # Use BFloat16 precision

    push_to_hub=True,           # Save to HuggingFace
    hub_strategy="every_save",  # Upload at each checkpoint
)
# WHY: Optimized for T4 GPU with 4-hour time limit
```

---

## 8. Phase 6: Running Training

### 8.1 Pre-Flight Checklist

Before starting training, verify:

```
□ GPU is enabled (Runtime → Change runtime type → T4)
□ All cells above trainer.train() run without errors
□ Dataset loaded correctly (check sample output)
□ Model loaded (check memory usage ~0.7GB)
□ Logged into HuggingFace (no token errors)
□ Output model name is unique (won't overwrite existing)
□ Estimated time is under 4 hours
```

### 8.2 During Training

**What you'll see:**
```
{'loss': 1.234, 'learning_rate': 0.0002, 'epoch': 0.1}
{'loss': 1.123, 'learning_rate': 0.00019, 'epoch': 0.2}
...
```

**What to monitor:**
```
HEALTHY TRAINING:
• Loss decreasing over time (1.2 → 0.6)
• No NaN or Inf values
• Memory usage stable
• Regular checkpoint saves

WARNING SIGNS:
• Loss not decreasing after 100+ steps
• Loss suddenly spikes to very high values
• "CUDA out of memory" errors
• Loss becomes NaN
```

### 8.3 If Colab Disconnects

**Don't panic!** If you configured `hub_strategy="every_save"`:

1. Your checkpoints are saved to HuggingFace Hub
2. Start a new session
3. Resume from checkpoint:

```python
# Load your saved checkpoint
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

# Load your fine-tuned adapter
model = PeftModel.from_pretrained(
    base_model,
    f"{HF_USERNAME}/{OUTPUT_MODEL}"
)

# Now you can use the model even if training didn't finish!
```

### 8.4 Interpreting Final Results

```python
TrainOutput(
    global_step=3146,           # Total training steps
    training_loss=0.6127,       # Final loss (lower is better)
    metrics={
        'train_runtime': 9891.8,        # Seconds
        'train_samples_per_second': 2.5, # Throughput
        'train_loss': 0.6127,           # Same as above
        'epoch': 1.0                    # Completed epochs
    }
)
```

**Good loss values:**
- Starting loss: 1.0-2.0
- Final loss: 0.3-0.8
- If loss < 0.1: Possible overfitting
- If loss > 2.0: Model not learning well

---

## 9. Phase 7: Testing Your Model

### 9.1 Quick Sanity Tests

```python
# Prepare for inference
model.eval()
model.config.use_cache = True
if hasattr(model, 'gradient_checkpointing_disable'):
    model.gradient_checkpointing_disable()

def generate(instruction, input_text="", max_tokens=256):
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    return response

# Test cases
tests = [
    ("Write a Python function to check if a number is prime", ""),
    ("Explain this code", "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)"),
    ("Fix the bug", "def divide(a, b):\n    return a / b"),
]

for instruction, input_text in tests:
    print(f"\n{'='*60}")
    print(f"Instruction: {instruction}")
    if input_text:
        print(f"Input: {input_text}")
    print(f"{'='*60}")
    print(generate(instruction, input_text))
```

### 9.2 Evaluation Checklist

```
□ Does it follow instructions correctly?
□ Does output format match training data?
□ Does it handle edge cases?
□ Is output coherent and complete?
□ Does it maintain the new behavior you trained for?
□ Does it still have base capabilities?
```

### 9.3 Common Issues and Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| Repetitive output | Temperature too low | Increase temperature to 0.8-1.0 |
| Cuts off mid-sentence | max_tokens too low | Increase max_new_tokens |
| Ignores input | Wrong prompt format | Match training format exactly |
| Generic responses | Undertrained | Train longer or add more data |
| Nonsense output | Dtype mismatch | Check bf16 consistency |

---

## 10. Phase 8: Uploading to HuggingFace Hub

### 10.1 Automatic Upload (Recommended)

If you used `push_to_hub=True` in training config, your model is already uploaded! Check:
```
https://huggingface.co/{HF_USERNAME}/{OUTPUT_MODEL}
```

### 10.2 Manual Upload

If you need to upload manually:

```python
# Save locally first
model.save_pretrained(f"./{OUTPUT_MODEL}")
tokenizer.save_pretrained(f"./{OUTPUT_MODEL}")

# Push to Hub
from huggingface_hub import HfApi

api = HfApi()

# Create repo if it doesn't exist
api.create_repo(
    repo_id=f"{HF_USERNAME}/{OUTPUT_MODEL}",
    repo_type="model",
    exist_ok=True
)

# Upload all files
api.upload_folder(
    folder_path=f"./{OUTPUT_MODEL}",
    repo_id=f"{HF_USERNAME}/{OUTPUT_MODEL}",
    repo_type="model"
)

print(f"Uploaded to: huggingface.co/{HF_USERNAME}/{OUTPUT_MODEL}")
```

### 10.3 Creating a Model Card

Create a `README.md` in your model repo:

```markdown
---
language:
- en
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen2.5-Coder-0.5B-Instruct
tags:
- code
- fine-tuned
---

# My Fine-Tuned Model

## Description
A fine-tuned version of Qwen2.5-Coder for [your task].

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
model = PeftModel.from_pretrained(base, "YourUsername/your-model")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")

# Generate
prompt = "### Instruction:\nYour instruction\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

## Training Details
- Base model: Qwen2.5-Coder-0.5B-Instruct
- Method: QLoRA (4-bit quantization + LoRA)
- Dataset: [Your dataset]
- Training time: X hours on T4 GPU

## Limitations
[Describe any limitations]
```

---

## 11. Phase 9: Deploying to HuggingFace Spaces

### 11.1 What is HuggingFace Spaces?

```
HuggingFace Spaces = Free hosting for ML demos

Features:
• Free CPU tier (2 vCPU, 16GB RAM)
• Paid GPU tiers available
• Supports Gradio, Streamlit, Docker
• Automatic HTTPS
• Persistent storage
```

### 11.2 Creating a Space

**Step 1: Go to Spaces**
```
https://huggingface.co/spaces
Click: "Create new Space"
```

**Step 2: Configure**
```
Owner: Your username
Space name: your-model-demo
License: MIT (or your choice)
SDK: Gradio
Hardware: CPU Basic (free)
```

**Step 3: Clone the Space**
```bash
git clone https://huggingface.co/spaces/YourUsername/your-model-demo
cd your-model-demo
```

### 11.3 Create the App Files

**app.py:**
```python
"""
Your Model Demo - HuggingFace Spaces
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Configuration
MODEL_ID = "YourUsername/your-model"
BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

# Global variables
model = None
tokenizer = None


def load_model():
    """Load the model."""
    global model, tokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    # CPU loading for free tier
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    model = PeftModel.from_pretrained(base_model, MODEL_ID)
    model.eval()
    print("Model loaded!")


def generate(instruction, context="", max_tokens=256, temperature=0.7):
    """Generate response."""
    global model, tokenizer

    if model is None:
        return "Model is loading..."

    if context.strip():
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    return response


# Load model at startup
load_model()

# Create interface
demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Instruction", placeholder="What do you want?"),
        gr.Textbox(label="Context (optional)", placeholder="Additional context..."),
        gr.Slider(64, 512, value=256, label="Max Length"),
        gr.Slider(0.1, 1.5, value=0.7, label="Temperature"),
    ],
    outputs=gr.Textbox(label="Response"),
    title="Your Model Demo",
    description="A fine-tuned model for [your task].",
    examples=[
        ["Write a function to add two numbers", "", 256, 0.7],
        ["Explain this code", "print('hello')", 256, 0.7],
    ],
)

demo.launch()
```

**requirements.txt:**
```
transformers>=4.36.0
peft>=0.7.0
torch>=2.0.0
gradio>=4.0.0
accelerate>=0.25.0
```

**README.md (for the Space):**
```markdown
---
title: Your Model Demo
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Your Model Demo

A demo of my fine-tuned model.
```

### 11.4 Deploy

```bash
# Add files
git add app.py requirements.txt README.md

# Commit
git commit -m "Initial deployment"

# Push (this triggers build)
git push
```

### 11.5 Monitor Deployment

```
Go to: https://huggingface.co/spaces/YourUsername/your-model-demo

Check:
• Build logs (click "Logs" tab)
• App status (should show "Running")
• Try the interface once it's ready
```

### 11.6 Troubleshooting Spaces

| Issue | Cause | Fix |
|-------|-------|-----|
| Build fails | Missing dependency | Add to requirements.txt |
| "Out of memory" | Model too large | Use smaller model or request GPU |
| Slow loading | CPU inference | Normal for CPU, consider GPU tier |
| Timeout on startup | Model loading slow | Add loading message to UI |

---

## 12. Troubleshooting Common Issues

### 12.1 Training Issues

**Error: CUDA out of memory**
```python
# Solutions:
1. Reduce batch_size (2 → 1)
2. Reduce max_seq_length (1024 → 512)
3. Use smaller model
4. Enable gradient_checkpointing=True
```

**Error: BFloat16 not implemented**
```python
# Cause: Mixing bf16 and fp16
# Fix: Use bf16=True, fp16=False in training config
# AND: Use bnb_4bit_compute_dtype=torch.bfloat16
```

**Error: Token not found / Login required**
```python
# Fix: Run login again
from huggingface_hub import login
login()  # Enter write token
```

**Model outputs garbage**
```python
# Cause: Dtype mismatch between quantization and training
# Fix: Ensure consistency:
bnb_config = BitsAndBytesConfig(
    bnb_4bit_compute_dtype=torch.bfloat16,  # Must match...
)
training_args = SFTConfig(
    bf16=True,   # ...this setting
    fp16=False,
)
```

**Loss not decreasing**
```python
# Possible causes:
1. Learning rate too low → Increase to 2e-4 or 5e-4
2. Dataset too small → Add more data
3. Data format wrong → Check prompt template
4. Model already knows this → Try different task
```

### 12.2 Inference Issues

**Output is repetitive**
```python
# Fix: Adjust generation parameters
outputs = model.generate(
    temperature=0.8,      # Increase randomness
    repetition_penalty=1.1,  # Penalize repetition
    no_repeat_ngram_size=3,  # Block repeated phrases
)
```

**Output cuts off**
```python
# Fix: Increase max tokens
outputs = model.generate(
    max_new_tokens=512,  # Increase from 256
)
```

**Wrong format**
```python
# Fix: Match training prompt format exactly
# If trained with:
#   "### Instruction:\n...\n\n### Response:\n"
# Use same format at inference!
```

### 12.3 Deployment Issues

**Space build fails**
```
# Check logs for specific error
# Common fixes:
1. Add missing packages to requirements.txt
2. Pin package versions
3. Check Python version compatibility
```

**Model too slow on CPU**
```python
# Options:
1. Accept slower speed (10-30 seconds per response)
2. Upgrade to GPU Space ($0.60/hour for T4)
3. Use smaller model
4. Reduce max_new_tokens
```

---

## 13. Decision Frameworks

### 13.1 Choosing Dataset Size

```
┌─────────────────────────────────────────────────────────────┐
│             HOW MUCH DATA DO I NEED?                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Question: How different is your task from base model?      │
│                                                             │
│  MINOR ADJUSTMENT (format, style):                          │
│  └── 100-500 examples                                       │
│                                                             │
│  MODERATE CHANGE (new domain):                              │
│  └── 1,000-5,000 examples                                   │
│                                                             │
│  MAJOR CHANGE (new capability):                             │
│  └── 5,000-50,000 examples                                  │
│                                                             │
│  Question: What percentage for new behavior?                │
│                                                             │
│  Want behavior in 80%+ of outputs:                          │
│  └── Include in 20-50% of training data                     │
│                                                             │
│  Want behavior in 50% of outputs:                           │
│  └── Include in 10-20% of training data                     │
│                                                             │
│  Want behavior occasionally:                                │
│  └── Include in 5-10% of training data                      │
│                                                             │
│  BELOW 1% = Will NOT learn the behavior!                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 13.2 Choosing Model Size

```
┌─────────────────────────────────────────────────────────────┐
│           WHICH MODEL SIZE SHOULD I USE?                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  YOUR CONSTRAINTS:                                          │
│                                                             │
│  Free Colab (T4, 16GB):                                     │
│  └── 0.5B - 7B models with QLoRA                            │
│  └── Recommended: Start with 0.5B-1.5B                      │
│                                                             │
│  Colab Pro (A100, 40GB):                                    │
│  └── Up to 33B models with QLoRA                            │
│  └── Recommended: 7B-13B models                             │
│                                                             │
│  Local RTX 3090/4090 (24GB):                                │
│  └── Up to 13B models with QLoRA                            │
│                                                             │
│  QUALITY vs EFFICIENCY:                                     │
│                                                             │
│  Smaller (0.5B-1.5B):                                       │
│  ✓ Fast training (1-2 hours)                                │
│  ✓ Fast inference                                           │
│  ✗ Less capable                                             │
│  Best for: Learning, simple tasks                           │
│                                                             │
│  Medium (3B-7B):                                            │
│  ✓ Good balance                                             │
│  ✓ Most tasks work well                                     │
│  Best for: Production demos                                 │
│                                                             │
│  Large (13B+):                                              │
│  ✓ Best quality                                             │
│  ✗ Slow training                                            │
│  ✗ Expensive inference                                      │
│  Best for: When quality is critical                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 13.3 Choosing Training Duration

```
┌─────────────────────────────────────────────────────────────┐
│            HOW LONG SHOULD I TRAIN?                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  EPOCHS (full passes through data):                         │
│                                                             │
│  1 epoch:                                                   │
│  └── Quick training                                         │
│  └── Less risk of overfitting                               │
│  └── Good for large datasets (>10k examples)                │
│  └── Fits in Colab free tier                                │
│                                                             │
│  2-3 epochs:                                                │
│  └── Better learning                                        │
│  └── Some overfitting risk                                  │
│  └── Good for medium datasets (1k-10k)                      │
│                                                             │
│  5+ epochs:                                                 │
│  └── Maximum learning                                       │
│  └── High overfitting risk                                  │
│  └── Only for small, high-quality datasets                  │
│                                                             │
│  TIME ESTIMATION:                                           │
│                                                             │
│  time_hours = (dataset_size × epochs) / (steps_per_hour)    │
│                                                             │
│  For T4 GPU with QLoRA:                                     │
│  └── ~1,100-1,500 steps per hour                            │
│  └── steps = dataset_size / effective_batch_size            │
│  └── effective_batch_size = batch_size × grad_accum         │
│                                                             │
│  Example (this project):                                    │
│  └── 25,162 examples × 1 epoch                              │
│  └── effective_batch = 2 × 4 = 8                            │
│  └── steps = 25,162 / 8 = 3,145 steps                       │
│  └── time = 3,145 / 1,100 = 2.86 hours ✓                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 14. Checklists

### 14.1 Pre-Training Checklist

```
ACCOUNTS & ACCESS:
□ HuggingFace account created
□ Write access token generated
□ Logged into Colab with Google account
□ Accepted any required model licenses

DATASET:
□ Dataset identified and loadable
□ Dataset format understood
□ Conversion function written
□ Sample output verified
□ Dataset size appropriate (not too small/large)

MODEL:
□ Base model selected
□ Model fits GPU memory (verified with estimate)
□ Model license compatible with your use

CONFIGURATION:
□ HF_USERNAME correct
□ OUTPUT_MODEL name unique
□ Training time estimated (<4 hours for Colab free)
□ Checkpointing enabled (save_steps set)
□ push_to_hub=True configured
```

### 14.2 During Training Checklist

```
MONITORING:
□ Loss is decreasing (check every 100 steps)
□ No NaN or Inf in loss
□ No CUDA memory errors
□ Checkpoints being saved
□ Hub uploads succeeding

TIMING:
□ Tracking elapsed time
□ On pace to finish before Colab timeout
□ Browser tab active (prevent idle timeout)
```

### 14.3 Post-Training Checklist

```
VERIFICATION:
□ Final model saved to Hub
□ Model card created
□ Quick inference test passed
□ Multiple test cases work

DEPLOYMENT:
□ Space created
□ app.py uploaded
□ requirements.txt complete
□ Space builds successfully
□ Demo works in browser
```

---

## 15. Resource Links

### 15.1 Documentation

| Resource | URL |
|----------|-----|
| HuggingFace Transformers | https://huggingface.co/docs/transformers |
| PEFT/LoRA | https://huggingface.co/docs/peft |
| TRL (Training) | https://huggingface.co/docs/trl |
| Datasets | https://huggingface.co/docs/datasets |
| HuggingFace Hub | https://huggingface.co/docs/hub |
| Gradio | https://gradio.app/docs |
| bitsandbytes | https://github.com/TimDettmers/bitsandbytes |

### 15.2 Tutorials & Guides

| Topic | URL |
|-------|-----|
| QLoRA Fine-tuning | https://huggingface.co/blog/4bit-transformers-bitsandbytes |
| SFT Training | https://huggingface.co/docs/trl/sft_trainer |
| Deploying to Spaces | https://huggingface.co/docs/hub/spaces |

### 15.3 Model Hubs

| Hub | Best For |
|-----|----------|
| HuggingFace Models | https://huggingface.co/models |
| HuggingFace Datasets | https://huggingface.co/datasets |

### 15.4 Community

| Platform | Link |
|----------|------|
| HuggingFace Discord | https://discord.gg/huggingface |
| HuggingFace Forums | https://discuss.huggingface.co |
| r/LocalLLaMA | https://reddit.com/r/LocalLLaMA |

---

## Appendix A: Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│              QUICK REFERENCE: FINE-TUNING LLMs              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  MEMORY FORMULA (QLoRA):                                    │
│  GPU_GB ≈ (Params_B × 0.5) + 1.5                            │
│  Example: 7B model → 7×0.5 + 1.5 = 5GB                      │
│                                                             │
│  TIME FORMULA:                                              │
│  Hours ≈ (Dataset × Epochs) / (Batch × GradAccum × 1100)    │
│                                                             │
│  ESSENTIAL SETTINGS:                                        │
│  • bnb_4bit_compute_dtype = torch.bfloat16                  │
│  • bf16=True, fp16=False                                    │
│  • gradient_checkpointing=True                              │
│  • push_to_hub=True, hub_strategy="every_save"              │
│                                                             │
│  DATA RULES:                                                │
│  • Want behavior? Include 5-20% examples of it              │
│  • <1% of data = won't learn                                │
│  • Always shuffle combined datasets                         │
│                                                             │
│  DEBUGGING:                                                 │
│  • Garbage output? Check dtype consistency                  │
│  • OOM? Reduce batch_size or seq_length                     │
│  • Not learning? Increase learning_rate or data             │
│                                                             │
│  COLAB SURVIVAL:                                            │
│  • save_steps=300 (every ~20 min)                           │
│  • hub_strategy="every_save"                                │
│  • Keep browser tab active                                  │
│  • Training <4 hours                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Appendix B: Template Files

### Minimal Training Notebook

Copy this to start any new project:

```python
# ===== CONFIG =====
HF_USERNAME = "YOUR_USERNAME"
BASE_MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
OUTPUT_MODEL = "my-model"
DATASET = "sahil2801/CodeAlpaca-20k"

# ===== SETUP =====
!pip install -q transformers datasets peft bitsandbytes accelerate trl
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login
login()

# ===== DATA =====
dataset = load_dataset(DATASET, split="train")
def fmt(x): return {"text": f"### Instruction:\n{x['instruction']}\n\n### Response:\n{x['output']}"}
dataset = dataset.map(fmt).shuffle(seed=42)

# ===== MODEL =====
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto")
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))

# ===== TRAIN =====
args = SFTConfig(output_dir=f"./{OUTPUT_MODEL}", num_train_epochs=1, per_device_train_batch_size=2,
    gradient_accumulation_steps=4, learning_rate=2e-4, bf16=True, gradient_checkpointing=True,
    save_steps=300, push_to_hub=True, hub_model_id=f"{HF_USERNAME}/{OUTPUT_MODEL}", hub_strategy="every_save",
    dataset_text_field="text", max_seq_length=1024)
SFTTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer).train()
```

---

*Guide version: 1.0*
*Last updated: December 2024*
*Author: Nimo*

**Remember:** The best way to learn is by doing. Start with a small project, make mistakes, and iterate. Each training run teaches you something new!
