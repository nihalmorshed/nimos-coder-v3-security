# Technical Documentation: Nimo's Coder Agent v3

A comprehensive deep-dive into the theory, architecture, and engineering decisions behind training a security-enhanced code generation model.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Transformer Architecture](#2-transformer-architecture)
3. [Attention Mechanisms](#3-attention-mechanisms)
4. [The Base Model: Qwen2.5-Coder](#4-the-base-model-qwen25-coder)
5. [Parameter-Efficient Fine-Tuning (PEFT)](#5-parameter-efficient-fine-tuning-peft)
6. [Quantization Deep Dive](#6-quantization-deep-dive)
7. [QLoRA: Combining Quantization and LoRA](#7-qlora-combining-quantization-and-lora)
8. [Training Strategy: From Scratch vs Continue](#8-training-strategy-from-scratch-vs-continue)
9. [Dataset Engineering](#9-dataset-engineering)
10. [Security-Focused Training](#10-security-focused-training)
11. [Training Dynamics](#11-training-dynamics)
12. [Inference Optimization](#12-inference-optimization)
13. [Evaluation Methodology](#13-evaluation-methodology)
14. [Lessons Learned: v2 to v3](#14-lessons-learned-v2-to-v3)
15. [Future Directions](#15-future-directions)
16. [References](#16-references)

---

## 1. Executive Summary

This project fine-tunes a 494M parameter language model for code generation with enhanced security vulnerability detection. Key technical decisions:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Base Model | Qwen2.5-Coder-0.5B | Code-specialized, efficient size |
| Fine-tuning Method | QLoRA | 97% memory reduction |
| Quantization | NF4 + Double Quantization | Optimal for normal distributions |
| Training Dtype | BFloat16 | Stability + range |
| Dataset Strategy | Mixed from scratch | Avoid catastrophic forgetting |
| Security Data | 20% of total | Sufficient signal for behavior change |

**Results:**
- Training time: 2.81 hours (T4 GPU)
- Token accuracy: 81.26%
- Security detection: Significantly improved
- Memory usage: 0.72 GB (vs ~2GB full precision)

---

## 2. Transformer Architecture

### 2.1 The Foundation

Transformers, introduced in "Attention Is All You Need" (Vaswani et al., 2017), revolutionized NLP by replacing recurrent architectures with self-attention.

```
┌─────────────────────────────────────────────────────────────┐
│                    TRANSFORMER BLOCK                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input Embeddings                                           │
│        │                                                    │
│        ▼                                                    │
│  ┌─────────────────┐                                        │
│  │ Multi-Head      │◄─── Query, Key, Value projections     │
│  │ Self-Attention  │                                        │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │ Add & LayerNorm │◄─── Residual connection               │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │ Feed-Forward    │◄─── Two linear layers + activation    │
│  │ Network (FFN)   │                                        │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │ Add & LayerNorm │◄─── Residual connection               │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│      Output                                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Why Transformers Work for Code

Code has properties that make transformers particularly effective:

1. **Long-range dependencies**: A variable defined 100 lines ago affects current code
2. **Hierarchical structure**: Functions contain blocks contain statements
3. **Pattern repetition**: Similar constructs appear throughout codebases
4. **Precise syntax**: Small changes have large semantic effects

Self-attention allows the model to "look at" any position when processing each token, crucial for understanding code context.

### 2.3 Decoder-Only Architecture

Qwen2.5-Coder uses a decoder-only architecture (like GPT), not encoder-decoder (like T5):

```
Encoder-Decoder (T5):          Decoder-Only (GPT/Qwen):

Input → [Encoder] → Context    Input → [Decoder] → Output
                ↓                         ↑
         [Decoder] → Output          (Causal mask)
```

**Why decoder-only for code generation?**
- Simpler architecture, fewer parameters
- Natural fit for autoregressive generation
- Causal masking prevents "looking ahead"
- Efficient for long sequence generation

---

## 3. Attention Mechanisms

### 3.1 Self-Attention Mathematics

Self-attention computes relationships between all positions in a sequence:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

Where:
- **Q** (Query): "What am I looking for?"
- **K** (Key): "What do I contain?"
- **V** (Value): "What information do I provide?"
- **d_k**: Dimension of keys (scaling factor)

**Step-by-step example:**

```python
# Input: "def add(a, b):"
# Token positions: [def, add, (, a, ,, b, ), :]

# 1. Each token creates Q, K, V vectors
Q_def = W_q @ embedding("def")  # What is 'def' looking for?
K_add = W_k @ embedding("add")  # What does 'add' represent?
V_add = W_v @ embedding("add")  # What info does 'add' provide?

# 2. Compute attention scores
score_def_to_add = Q_def @ K_add.T / sqrt(d_k)

# 3. Softmax normalizes scores to probabilities
attention_weights = softmax(all_scores)

# 4. Weighted sum of values
output = attention_weights @ V
```

### 3.2 Multi-Head Attention

Instead of one attention function, we use multiple "heads" that attend to different aspects:

```
┌──────────────────────────────────────────────────────────┐
│                  MULTI-HEAD ATTENTION                    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Input                                                   │
│    │                                                     │
│    ├──► Head 1: Syntax patterns (def, class, if)        │
│    ├──► Head 2: Variable relationships (a → a)          │
│    ├──► Head 3: Scope boundaries (indentation)          │
│    ├──► Head 4: Type patterns (int, str, list)          │
│    │    ...                                              │
│    ├──► Head N: Other patterns                          │
│    │                                                     │
│    ▼                                                     │
│  Concatenate all heads                                   │
│    │                                                     │
│    ▼                                                     │
│  Linear projection → Output                              │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**Qwen2.5-Coder-0.5B configuration:**
- 14 attention heads
- 64 dimensions per head
- 896 total hidden dimension (14 × 64)

### 3.3 Grouped Query Attention (GQA)

Qwen2.5 uses GQA, an optimization between Multi-Head Attention (MHA) and Multi-Query Attention (MQA):

```
MHA:  Q₁ K₁ V₁   Q₂ K₂ V₂   Q₃ K₃ V₃   Q₄ K₄ V₄
      (Each head has its own K, V)

MQA:  Q₁ ─┐      Q₂ ─┐      Q₃ ─┐      Q₄ ─┐
          ├─ K V     ├─ K V     ├─ K V     ├─ K V
      (All heads share one K, V)

GQA:  Q₁ Q₂ ─┐      Q₃ Q₄ ─┐
             ├─ K₁ V₁       ├─ K₂ V₂
      (Groups of heads share K, V)
```

**Benefits of GQA:**
- Reduces KV-cache memory during inference
- Maintains most of MHA's expressiveness
- Faster decoding for long sequences

### 3.4 Rotary Position Embeddings (RoPE)

Unlike absolute position embeddings, RoPE encodes position through rotation:

```python
# Traditional absolute position:
embedding = token_embedding + position_embedding[pos]

# RoPE: Rotate query/key vectors based on position
def rope(x, position):
    # Split into pairs and rotate
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos(position * theta)
    sin = sin(position * theta)
    return concat(x1*cos - x2*sin, x1*sin + x2*cos)
```

**Why RoPE for code?**
- Relative position awareness (distance between tokens matters)
- Extrapolates to longer sequences than seen during training
- Efficient computation (applied only to Q, K)

---

## 4. The Base Model: Qwen2.5-Coder

### 4.1 Model Architecture

```
Qwen2.5-Coder-0.5B-Instruct
├── Embedding Layer: 151,936 vocab × 896 dim
├── 24 Transformer Layers
│   ├── Multi-Head Attention (14 heads, GQA)
│   ├── RMSNorm (instead of LayerNorm)
│   ├── SwiGLU Feed-Forward Network
│   └── Residual Connections
├── Final RMSNorm
└── Language Model Head: 896 → 151,936

Total Parameters: 494,032,768
```

### 4.2 Why Qwen2.5-Coder?

| Feature | Benefit |
|---------|---------|
| Code-specific pretraining | Better code understanding out-of-box |
| Instruction-tuned variant | Already understands instruction format |
| 0.5B size | Fits in free tier GPUs with quantization |
| Apache 2.0 license | Commercially usable |
| Recent (2024) | State-of-the-art techniques |

### 4.3 SwiGLU Activation

Qwen uses SwiGLU instead of ReLU/GELU in feed-forward layers:

```python
# Traditional FFN:
FFN(x) = W_2 @ ReLU(W_1 @ x)

# SwiGLU FFN:
FFN(x) = W_2 @ (Swish(W_gate @ x) * (W_up @ x))

# Where Swish(x) = x * sigmoid(x)
```

**Why SwiGLU?**
- Smoother gradients than ReLU
- Gating mechanism allows selective information flow
- Empirically better performance on language tasks

### 4.4 RMSNorm vs LayerNorm

```python
# LayerNorm:
y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta

# RMSNorm (Root Mean Square Normalization):
y = x / sqrt(mean(x²) + eps) * gamma
```

**Why RMSNorm?**
- Simpler computation (no mean subtraction)
- ~10% faster training
- Equivalent performance in practice

---

## 5. Parameter-Efficient Fine-Tuning (PEFT)

### 5.1 The Problem with Full Fine-Tuning

Full fine-tuning updates all 494M parameters:

```
Memory Required = Parameters × Bytes per Parameter × Overhead

For Adam optimizer with fp32:
- Parameters: 494M × 4 bytes = 1.98 GB
- Gradients: 494M × 4 bytes = 1.98 GB
- Optimizer states (m, v): 494M × 4 × 2 = 3.95 GB
- Activations: Variable, often 2-10 GB

Total: ~10-20 GB minimum
```

This exceeds free tier GPU memory (T4: 16GB, but shared with OS/CUDA).

### 5.2 Low-Rank Adaptation (LoRA)

LoRA's key insight: Weight updates during fine-tuning have low "intrinsic rank."

**Mathematical Foundation:**

Instead of updating weight matrix W directly:
```
W_new = W_original + ΔW
```

LoRA decomposes ΔW into two smaller matrices:
```
ΔW = B × A

Where:
- W_original: d × k (frozen, not updated)
- A: r × k (trainable)
- B: d × r (trainable)
- r << min(d, k) (the "rank")
```

**Visual representation:**

```
Full Fine-Tuning:              LoRA:

W (896 × 896)                  W (896 × 896) [FROZEN]
= 802,816 params                      │
All trainable                         │
                                      ▼
                               ┌──────────────┐
                               │   B (896×16) │ = 14,336 params
                               │   A (16×896) │ = 14,336 params
                               └──────────────┘
                               Total: 28,672 trainable
                               (3.6% of original)
```

### 5.3 Why Low Rank Works

Research shows that weight updates during fine-tuning lie in a low-dimensional subspace:

```
Original Weight Space: 896 × 896 = 802,816 dimensions
Effective Update Space: ~16-64 dimensions (empirically)

The "intrinsic dimensionality" of fine-tuning is small.
```

This makes intuitive sense for domain adaptation:
- The base model already "knows" language/code
- Fine-tuning adjusts behavior, not fundamentals
- Small adjustments in the right directions suffice

### 5.4 LoRA Configuration Explained

```python
lora_config = LoraConfig(
    r=16,                # Rank of adaptation matrices
    lora_alpha=32,       # Scaling factor
    lora_dropout=0.05,   # Regularization
    target_modules=[     # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)
```

**Parameter: r (rank)**
```
r=8:  Minimal adaptation, fastest training
r=16: Good balance (our choice)
r=32: More capacity, slower training
r=64: Near full fine-tuning capacity
```

**Parameter: lora_alpha**

Scaling factor for LoRA updates:
```python
output = original_output + (lora_alpha / r) * lora_output
```

With r=16 and alpha=32, the scaling is 32/16 = 2.0

Higher alpha → stronger adaptation signal
Lower alpha → more conservative updates

**Parameter: target_modules**

We target all attention projections AND feed-forward layers:
```
Attention: q_proj, k_proj, v_proj, o_proj
FFN: gate_proj, up_proj, down_proj

Why all? Security detection requires changes to:
- How the model attends to code patterns (attention)
- How it processes and transforms representations (FFN)
```

### 5.5 Trainable Parameters Calculation

```python
# For each target module in each layer:
params_per_module = r * input_dim + r * output_dim

# Qwen2.5-Coder-0.5B dimensions:
hidden_size = 896
intermediate_size = 4864
num_layers = 24
r = 16

# Attention projections (q, k, v, o):
attention_params = 4 * (r * hidden_size + r * hidden_size) * num_layers
# = 4 * (16 * 896 + 16 * 896) * 24 = 2,752,512

# FFN projections (gate, up, down):
# gate/up: hidden → intermediate
# down: intermediate → hidden
ffn_params = 2 * (r * hidden_size + r * intermediate_size) * num_layers
ffn_params += (r * intermediate_size + r * hidden_size) * num_layers
# = 2 * (16*896 + 16*4864) * 24 + (16*4864 + 16*896) * 24
# = 6,045,696

Total ≈ 8,798,208 trainable parameters (2.72% of model)
```

---

## 6. Quantization Deep Dive

### 6.1 Why Quantize?

Neural networks are typically trained in FP32 (32-bit floating point):

```
FP32: 1 sign bit + 8 exponent bits + 23 mantissa bits
Range: ±3.4 × 10^38
Precision: ~7 decimal digits

Memory per parameter: 4 bytes
494M params × 4 bytes = 1.98 GB
```

Quantization reduces precision to save memory:

```
FP16: 1 + 5 + 10 bits → 2 bytes → 0.99 GB
INT8: 8 bits total → 1 byte → 0.49 GB
INT4: 4 bits total → 0.5 byte → 0.25 GB
```

### 6.2 The Challenge of 4-bit Quantization

Naive 4-bit quantization fails because:

```
4 bits = 16 possible values
Neural network weights follow roughly normal distribution
Equal-width bins waste precision:

      │
      │    ████
      │   ██████
      │  ████████
      │ ██████████
      │████████████
      └────────────────
        Bin: [1][2][3][4][5][6]...

Most weights cluster near zero, but uniform bins
allocate equal precision everywhere.
```

### 6.3 NF4: Normal Float 4-bit

NF4 (Normal Float 4-bit) uses non-uniform quantization levels:

```python
# NF4 quantization levels (normalized):
nf4_levels = [
    -1.0, -0.6962, -0.5251, -0.3949,
    -0.2844, -0.1848, -0.0911, 0.0,
    0.0796, 0.1609, 0.2461, 0.3379,
    0.4407, 0.5626, 0.7230, 1.0
]

# These levels are optimized for normal distribution:
# More levels near zero where weights cluster
# Fewer levels at extremes where weights are rare
```

**Visual comparison:**

```
Uniform INT4:     │ ─ │ ─ │ ─ │ ─ │ ─ │ ─ │ ─ │ ─ │
                  -8      -4       0       4       8

NF4:              │─│─│─│──│───│───│──│─│─│─│
                  -1    -0.5      0      0.5     1
                  (Dense near zero, sparse at extremes)
```

### 6.4 Block-wise Quantization

Weights are quantized in blocks, each with its own scale:

```python
# Block size: 64 weights per block
block_size = 64

# For each block:
# 1. Find absolute maximum
block_max = max(abs(weights[block]))

# 2. Scale weights to [-1, 1]
normalized = weights[block] / block_max

# 3. Quantize to nearest NF4 level
quantized = find_nearest_nf4(normalized)

# 4. Store: quantized values (4-bit) + scale (FP32/FP16)
```

**Memory calculation:**
```
Per block: 64 weights × 4 bits + 1 scale × 16 bits
         = 256 bits + 16 bits = 272 bits
Per weight: 272 / 64 = 4.25 bits (overhead from scales)
```

### 6.5 Double Quantization

Double quantization quantizes the quantization scales themselves:

```
Standard:          Weights (4-bit) + Scales (FP16)
Double Quantized:  Weights (4-bit) + Scales (8-bit) + Super-scales (FP16)

Memory savings: ~0.5 bits per weight
```

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # Use NF4 levels
    bnb_4bit_use_double_quant=True,      # Quantize scales too
    bnb_4bit_compute_dtype=torch.bfloat16  # Compute in BF16
)
```

### 6.6 BFloat16 vs Float16

```
Float16 (FP16):  1 sign + 5 exponent + 10 mantissa
                 Range: ±65,504
                 Precision: ~3.3 decimal digits

BFloat16 (BF16): 1 sign + 8 exponent + 7 mantissa
                 Range: ±3.4 × 10^38 (same as FP32!)
                 Precision: ~2.4 decimal digits
```

**Why BFloat16 for training?**

```
Problem with FP16:
- Limited range causes overflow/underflow in gradients
- Loss scaling tricks required
- Can cause training instability

BFloat16 advantages:
- Same range as FP32 (no overflow issues)
- Hardware support on modern GPUs (T4, A100)
- No loss scaling needed
- Slightly less precision, but sufficient for neural networks
```

---

## 7. QLoRA: Combining Quantization and LoRA

### 7.1 The QLoRA Innovation

QLoRA (Quantized LoRA) combines:
1. 4-bit NF4 quantized base model (frozen)
2. FP16/BF16 LoRA adapters (trainable)
3. Paged optimizers for memory efficiency

```
┌─────────────────────────────────────────────────────────────┐
│                      QLoRA ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input                                                     │
│     │                                                       │
│     ▼                                                       │
│   ┌─────────────────────────────────────────┐               │
│   │         BASE MODEL WEIGHTS              │               │
│   │         (4-bit NF4 Quantized)           │               │
│   │         FROZEN - No gradients           │               │
│   └─────────────────┬───────────────────────┘               │
│                     │                                       │
│                     │ Dequantize to BF16 for computation    │
│                     ▼                                       │
│   ┌─────────────────────────────────────────┐               │
│   │    Computation in BFloat16              │               │
│   │    (Mixed with LoRA outputs)            │               │
│   └─────────────────┬───────────────────────┘               │
│                     │                                       │
│     ┌───────────────┴───────────────┐                       │
│     │                               │                       │
│     ▼                               ▼                       │
│   ┌───────────┐               ┌───────────┐                 │
│   │  LoRA A   │               │  LoRA B   │                 │
│   │  (BF16)   │──────────────►│  (BF16)   │                 │
│   │ Trainable │               │ Trainable │                 │
│   └───────────┘               └───────────┘                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Memory Breakdown

```
Component                          Memory (494M param model)
─────────────────────────────────────────────────────────────
Base model (4-bit quantized)       ~0.31 GB
LoRA adapters (BF16)               ~0.017 GB (8.8M params)
Gradients (only for LoRA)          ~0.017 GB
Optimizer states (only for LoRA)   ~0.07 GB
Activations (gradient checkpointing) ~0.3 GB
─────────────────────────────────────────────────────────────
Total                              ~0.72 GB

vs Full Fine-Tuning:               ~10-15 GB
Memory Reduction:                  ~95%
```

### 7.3 Gradient Flow in QLoRA

```python
# Forward pass:
def forward(x):
    # 1. Dequantize frozen weights
    W_dequant = dequantize_nf4(W_quantized)  # 4-bit → BF16

    # 2. Compute base output
    base_output = W_dequant @ x

    # 3. Compute LoRA output
    lora_output = B @ (A @ x)  # Both A, B are BF16

    # 4. Combine with scaling
    return base_output + (alpha / r) * lora_output

# Backward pass:
# Gradients only flow through LoRA path (A, B matrices)
# Base weights remain frozen (no gradients computed)
```

### 7.4 Paged Optimizers

AdamW optimizer stores two states per parameter (momentum, variance):

```python
# Standard AdamW memory:
optimizer_memory = 2 * num_params * bytes_per_param

# For 8.8M LoRA params in FP32:
# = 2 * 8.8M * 4 = 70.4 MB
```

Paged optimizers use CPU RAM as overflow:
```python
optim="paged_adamw_8bit"  # 8-bit quantized optimizer states
                          # Pages to CPU if GPU memory pressure
```

---

## 8. Training Strategy: From Scratch vs Continue

### 8.1 The Decision

For v3, we trained from the base model, not continuing from v2:

```
Option A: Continue from v2        Option B: Train from base (CHOSEN)

v2 weights                        Base Qwen weights
    │                                  │
    ▼                                  ▼
Train on security data            Train on MIXED data
    │                             (CodeAlpaca + Security)
    ▼                                  │
v3 weights                             ▼
                                  v3 weights
```

### 8.2 Why Not Continue from v2?

**Catastrophic Forgetting:**

When fine-tuning on new data, models can "forget" previously learned capabilities:

```
Training Step:     0      1000    2000    3000
────────────────────────────────────────────────
General Coding:   100%    95%     85%     70%  ← Decreasing!
Security:          40%    55%     70%     80%  ← Increasing

The model trades off old skills for new ones.
```

**Data Proportion Analysis:**

```
v2 training data:     20,000 examples (CodeAlpaca)
New security data:     5,000 examples
Ratio:                25% new data

Rule of thumb:
- <10% new data: Safe to continue fine-tuning
- 10-30% new data: Risk of forgetting, consider mixing
- >30% new data: Strongly recommend training from scratch with mix
```

### 8.3 The Mixing Strategy

We combined all data and trained from the base model:

```
Final Dataset Composition:
───────────────────────────────────
CodeAlpaca:     20,022 (79.6%)  ← General coding
Security DPO:    5,000 (19.9%)  ← Vulnerability pairs
CrossVul:          128 (0.5%)   ← Real bug fixes
Error Handling:     12 (0.05%)  ← Hand-crafted
───────────────────────────────────
Total:          25,162 (100%)
```

**Why this ratio works:**

```
Signal Strength for Behavior Change:

20% security data is enough to:
✓ Learn to identify vulnerabilities
✓ Generate security warnings
✓ Suggest secure alternatives

But NOT enough to:
✗ Change fundamental code generation patterns
✗ Add error handling consistently (0.05% was too weak)
```

### 8.4 Shuffling Importance

Data was shuffled before training:

```python
combined_dataset = combined_dataset.shuffle(seed=42)
```

**Why shuffling matters:**

```
Without shuffling:
Batch 1-1000:    All CodeAlpaca
Batch 1001-1500: All Security
→ Model "mode switches" between domains

With shuffling:
Each batch: Mixed CodeAlpaca + Security
→ Model learns to handle both simultaneously
→ Better generalization
```

---

## 9. Dataset Engineering

### 9.1 Dataset Sources

#### CodeAlpaca-20k
```
Source: sahil2801/CodeAlpaca-20k
Size: 20,022 examples
Format: Instruction-Input-Output triples

Example:
{
  "instruction": "Write a function to sort a list",
  "input": "",
  "output": "def sort_list(lst):\n    return sorted(lst)"
}
```

#### Security DPO (Direct Preference Optimization)
```
Source: CyberNative/Code_Vulnerability_Security_DPO
Size: 4,656 examples (we used 2,500)
Format: Chosen (secure) vs Rejected (vulnerable)

Example:
{
  "question": "How to execute a command?",
  "chosen": "subprocess.run(['cmd'], shell=False)",
  "rejected": "os.system(user_input)"
}
```

#### CrossVul
```
Source: hitoshura25/crossvul
Size: 9,313 examples (we used 128 after filtering)
Format: Before (vulnerable) and After (fixed) code pairs

Example:
{
  "cwe_id": "CWE-78",
  "vulnerable_code": "os.system(f'rm {file}')",
  "fixed_code": "os.remove(file)",
  "language": "Python"
}

Note: Many examples exceeded our 2000 char limit,
resulting in only 128 usable examples.
```

### 9.2 Data Conversion Pipeline

```python
def convert_security_dpo(item):
    """Convert DPO format to SFT format."""

    # Create vulnerability detection example
    detection_example = {
        "instruction": "Review this code for security vulnerabilities. Is it safe?",
        "input": item["rejected"],  # Vulnerable code
        "output": f"""NO - SECURITY VULNERABILITY DETECTED!

This code contains security issues that could be exploited.

**Secure Version:**
```
{item["chosen"]}
```

Always validate inputs and follow secure coding practices."""
    }

    # Create fix example
    fix_example = {
        "instruction": "Fix the security vulnerabilities in this code",
        "input": item["rejected"],
        "output": item["chosen"]
    }

    return [detection_example, fix_example]
```

### 9.3 Prompt Template

All examples follow the Alpaca format:

```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

**Why Alpaca format?**
- Well-established, widely used
- Clear separation of components
- Model easily learns the pattern
- Compatible with many inference frameworks

### 9.4 Data Quality Considerations

**What we filtered:**
```python
# Length filtering for CrossVul
if len(vuln_code) > 2000 or len(fixed_code) > 2000:
    continue  # Skip very long examples

# This reduced CrossVul from 9,313 to 128 usable examples
```

**What we should have done (lessons learned):**
```python
# More error handling examples needed
# 12 examples = 0.05% of data = too weak signal

# Recommendation: At least 2-5% of data for behavior change
min_examples = total_data * 0.02  # ~500 examples
```

---

## 10. Security-Focused Training

### 10.1 Types of Vulnerabilities Covered

```
┌─────────────────────────────────────────────────────────────┐
│              SECURITY VULNERABILITIES IN TRAINING           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  INJECTION ATTACKS                                          │
│  ├── Command Injection (os.system, subprocess)              │
│  ├── SQL Injection (string formatting in queries)           │
│  └── Code Injection (eval, exec)                            │
│                                                             │
│  INPUT VALIDATION                                           │
│  ├── Path Traversal (../ in file paths)                     │
│  ├── Unvalidated Redirects                                  │
│  └── Type Confusion                                         │
│                                                             │
│  AUTHENTICATION/AUTHORIZATION                               │
│  ├── Hardcoded Credentials                                  │
│  ├── Weak Password Handling                                 │
│  └── Missing Access Controls                                │
│                                                             │
│  DATA EXPOSURE                                              │
│  ├── Information Leakage                                    │
│  ├── Sensitive Data in Logs                                 │
│  └── Insecure Data Storage                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 10.2 Teaching Vulnerability Detection

The training data teaches two complementary skills:

**Skill 1: Identify vulnerabilities**
```
Input: "Is this code safe?"
       os.system(f"cat {user_input}")

Expected Output: "NO - SECURITY VULNERABILITY!
                 Command Injection detected..."
```

**Skill 2: Fix vulnerabilities**
```
Input: "Fix the security issues"
       os.system(f"cat {user_input}")

Expected Output: subprocess.run(["cat", user_input], shell=False)
```

### 10.3 Why Security Training Worked

The improvement from v2 to v3 was dramatic for security:

```
v2 Security Test (Command Injection):
Input: os.system(f"cat {user_input}")
Output: "This code looks fine."  ← WRONG!

v3 Security Test (Command Injection):
Input: os.system(f"cat {user_input}")
Output: "NO - SECURITY VULNERABILITY DETECTED!
        This code contains security issues..."  ← CORRECT!
```

**Why it worked:**
1. **Clear pattern**: Security examples have consistent format
2. **Sufficient volume**: 5,000 security examples = 20% of data
3. **Diverse coverage**: Multiple vulnerability types
4. **Explicit labeling**: "NO - SECURITY VULNERABILITY" is unambiguous

### 10.4 Why Error Handling Training Failed

```
Error Handling Test:
Input: "Write a function to read a file"
Output: def read_file(f):
            with open(f) as file:
                return file.read()
        ← No try-except!

Expected (from training data):
def read_file(f):
    try:
        with open(f) as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Not found: {f}")
```

**Why it failed:**
1. **Too few examples**: 12 examples = 0.05% of data
2. **Overwhelmed by CodeAlpaca**: 20k examples mostly without error handling
3. **Pattern not reinforced**: Model saw error handling rarely
4. **No negative examples**: We didn't show "code without error handling is bad"

---

## 11. Training Dynamics

### 11.1 Training Configuration

```python
SFTConfig(
    num_train_epochs=1,           # Single pass through data
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    bf16=True,
    max_seq_length=1024,
)
```

### 11.2 Effective Batch Size

```
Effective batch size = batch_size × gradient_accumulation × num_GPUs
                     = 2 × 4 × 1
                     = 8 examples per optimization step

Why gradient accumulation?
- GPU memory can only fit batch_size=2
- Accumulate gradients over 4 forward passes
- Update weights once with accumulated gradients
- Simulates larger batch without more memory
```

### 11.3 Learning Rate Schedule

```
                     Learning Rate Over Training
    │
2e-4│         ┌─────────────────────────────────┐
    │        /                                   \
    │       /                                     \
    │      /                                       \
    │     /                                         \
    │    /                                           \
  0 │___/                                             \___
    └─────────────────────────────────────────────────────
         Warmup            Cosine Decay           End
         (3%)              (94%)                  (3%)
```

**Warmup phase (3%):**
```
Steps 0-94: LR gradually increases from 0 to 2e-4
Why: Prevents large gradient updates early when model is unstable
```

**Cosine decay:**
```
Steps 94-3146: LR follows cosine curve down to ~0
Why: Smoother than step decay, better final convergence
Formula: lr = lr_max * 0.5 * (1 + cos(π * progress))
```

### 11.4 Training Metrics

```
Metric              Start       End         Interpretation
────────────────────────────────────────────────────────────
train_loss          ~1.2        0.613       Model is learning
token_accuracy      ~75%        81.26%      Better predictions
entropy             ~0.8        0.673       More confident outputs
```

**Understanding the metrics:**

```
Loss (Cross-Entropy):
- Measures how wrong predictions are
- Lower is better
- 0.613 is good for a fine-tuned model

Token Accuracy:
- % of next tokens predicted correctly
- 81.26% means 4 out of 5 tokens correct
- Reasonable for code generation

Entropy:
- Measures prediction uncertainty
- Lower = more confident
- 0.673 shows model is decisive
```

### 11.5 Training Timeline

```
Total Training Time: 2.81 hours
Steps: 3,146
Time per step: ~3.2 seconds

Breakdown:
- Data loading/tokenization: 3.6 minutes
- Forward pass: ~1.5 seconds/step
- Backward pass: ~1.2 seconds/step
- Optimizer step: ~0.3 seconds/step
- Logging/checkpointing: ~0.2 seconds/step
```

### 11.6 Checkpoint Strategy

```python
save_strategy="steps",
save_steps=300,
save_total_limit=3,
hub_strategy="every_save",
```

**Checkpoint timeline:**
```
Step 300:  Checkpoint 1 saved, pushed to Hub
Step 600:  Checkpoint 2 saved, pushed to Hub
Step 900:  Checkpoint 3 saved, pushed to Hub
Step 1200: Checkpoint 4 saved, Checkpoint 1 deleted
...
Step 3000: Checkpoint 10 saved, only last 3 kept
Step 3146: Final model saved
```

**Why frequent saves?**
- Colab can disconnect at any time
- `hub_strategy="every_save"` uploads immediately
- Even if disconnected at step 2800, we have step 2700 checkpoint
- This saved v2 training when Colab disconnected at step 3341/3381!

---

## 12. Inference Optimization

### 12.1 Switching to Inference Mode

```python
# After training:
model.eval()                           # Disable dropout
model.config.use_cache = True          # Enable KV caching
model.gradient_checkpointing_disable() # Disable activation checkpointing
```

**Why each change matters:**

| Setting | Training | Inference | Why |
|---------|----------|-----------|-----|
| model.train()/eval() | Dropout ON | Dropout OFF | Deterministic outputs |
| use_cache | False | True | KV caching for speed |
| gradient_checkpointing | True | False | No need to save activations |

### 12.2 KV Caching Explained

Without KV cache (slow):
```
Generating: "def add(a, b):\n    return"
Step 1: Process ["def"]                     → predict "add"
Step 2: Process ["def", "add"]              → predict "("
Step 3: Process ["def", "add", "("]         → predict "a"
...
Step N: Process all previous tokens         → predict next

Complexity: O(n²) - each step processes all previous tokens
```

With KV cache (fast):
```
Step 1: Process ["def"], cache K₁,V₁        → predict "add"
Step 2: Process ["add"], use K₁,V₁, cache K₂,V₂  → predict "("
Step 3: Process ["("], use K₁,V₁,K₂,V₂, cache K₃,V₃  → predict "a"
...

Complexity: O(n) - each step only processes new token
Speedup: ~10-50x for long sequences
```

### 12.3 Generation Parameters

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=256,      # Maximum tokens to generate
    temperature=0.7,          # Randomness control
    top_p=0.9,               # Nucleus sampling
    do_sample=True,          # Enable sampling (vs greedy)
    pad_token_id=tokenizer.eos_token_id,
)
```

**Temperature:**
```
temperature = 0.1: Very deterministic, repetitive
temperature = 0.7: Balanced (our choice)
temperature = 1.0: As trained
temperature = 1.5: Creative but potentially incoherent

Math: probs = softmax(logits / temperature)
Lower temp → sharper distribution → more deterministic
```

**Top-p (Nucleus Sampling):**
```
top_p = 0.9 means:
1. Sort tokens by probability
2. Keep tokens until cumulative prob >= 0.9
3. Renormalize and sample from this subset

Example:
Token A: 50% ─┐
Token B: 30%  ├─ Sum = 90%, keep these
Token C: 10% ─┘
Token D: 5%  ─── Excluded
Token E: 5%  ─── Excluded

Prevents sampling very unlikely tokens
```

---

## 13. Evaluation Methodology

### 13.1 Test Categories

```
┌─────────────────────────────────────────────────────────────┐
│                    EVALUATION FRAMEWORK                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  SECURITY TESTS                                             │
│  ├── Command Injection Detection                            │
│  │   Input: os.system(f"cat {user_input}")                  │
│  │   Pass: Response contains "vulnerability/unsafe/no"      │
│  │                                                          │
│  ├── SQL Injection Detection                                │
│  │   Input: f"SELECT * FROM users WHERE id = {user_id}"     │
│  │   Pass: Response mentions "SQL injection/parameterized"  │
│  │                                                          │
│  └── Safe Code Recognition                                  │
│      Input: cursor.execute("...WHERE id = ?", (id,))        │
│      Pass: Response confirms safety                         │
│                                                             │
│  ERROR HANDLING TESTS                                       │
│  ├── File Reading Function                                  │
│  │   Pass: Output contains try/except                       │
│  │                                                          │
│  └── API Fetch Function                                     │
│      Pass: Output contains try/catch and response.ok        │
│                                                             │
│  GENERAL CODING TESTS                                       │
│  └── Prime Number Function                                  │
│      Pass: Generates correct algorithm                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 13.2 Automated Evaluation

```python
def evaluate_security_detection(response, expected_keywords):
    """Check if model detected vulnerability."""
    response_lower = response.lower()
    found = [kw for kw in expected_keywords if kw in response_lower]
    return len(found) >= 2  # At least 2 keywords

def evaluate_error_handling(response):
    """Check if generated code has error handling."""
    has_try = 'try' in response.lower()
    has_except = 'except' in response.lower() or 'catch' in response.lower()
    return has_try and has_except
```

### 13.3 Results Analysis

```
Test Results Summary:
─────────────────────────────────────────────
Test                        v2      v3      Δ
─────────────────────────────────────────────
Command Injection           FAIL    PASS    +
SQL Injection              FAIL    PASS    +
Error Handling (Python)    FAIL    FAIL    =
Error Handling (JS)        FAIL    FAIL    =
General Coding             PASS    PASS    =
─────────────────────────────────────────────
Security Detection:        0/2     2/2     +100%
Error Handling:            0/2     0/2     No change
General:                   1/1     1/1     Maintained
```

---

## 14. Lessons Learned: v2 to v3

### 14.1 What Worked

```
✓ Security training with 20% of data
  - Clear improvement in vulnerability detection
  - Model learned the pattern consistently

✓ Training from scratch with mixed data
  - No catastrophic forgetting
  - Maintained general coding ability

✓ Frequent checkpointing
  - Saved training when Colab disconnected
  - peace of mind during long runs

✓ BFloat16 consistency
  - Fixed the dtype mismatch from v2
  - No more garbage output
```

### 14.2 What Didn't Work

```
✗ Error handling with 0.05% of data
  - Signal too weak
  - Overwhelmed by examples without error handling
  - Need at least 2-5% for behavior change

✗ CrossVul dataset utilization
  - Only 128/9313 examples usable
  - Length filtering too aggressive
  - Should chunk long examples instead

✗ Single epoch limitations
  - Less exposure to each example
  - Might benefit from 2 epochs with smaller dataset
```

### 14.3 Recommendations for Future

```
For Error Handling Improvement:
1. Create 500+ error handling examples
2. Include "bad" examples (code without error handling marked as incomplete)
3. Consider separate fine-tuning pass focused on error handling

For Better Dataset Utilization:
1. Chunk long code examples instead of filtering
2. Balance dataset more carefully
3. Use curriculum learning (easy → hard)

For Training Stability:
1. Always validate dtype consistency before training
2. Run quick sanity check (100 steps) before full training
3. Monitor loss curve for anomalies
```

---

## 15. Future Directions

### 15.1 Model Scaling

```
Current: Qwen2.5-Coder-0.5B (494M params)

Potential upgrades:
┌─────────────────────────────────────────────────────────────┐
│  Model               Params    Memory(4-bit)  Expected Gain │
├─────────────────────────────────────────────────────────────┤
│  Qwen2.5-Coder-1.5B  1.5B     ~1.2 GB        +5-10%        │
│  Qwen2.5-Coder-3B    3B       ~2.5 GB        +10-15%       │
│  Qwen2.5-Coder-7B    7B       ~5.5 GB        +15-20%       │
│  DeepSeek-Coder-6.7B 6.7B     ~5.2 GB        +15-20%       │
└─────────────────────────────────────────────────────────────┘

Trade-off: Larger models need more memory, longer training
T4 GPU (16GB) can handle up to 7B with QLoRA
```

### 15.2 Advanced Techniques

```
Potential improvements:

1. DPO (Direct Preference Optimization)
   - Train on preference pairs directly
   - Better for "secure vs insecure" classification

2. RLHF (Reinforcement Learning from Human Feedback)
   - Use security expert feedback
   - More nuanced security understanding

3. Retrieval Augmented Generation (RAG)
   - Retrieve security documentation
   - Access to up-to-date vulnerability info

4. Multi-task Learning
   - Train on multiple objectives simultaneously
   - Code generation + security review + error handling
```

### 15.3 Evaluation Improvements

```
Current: Keyword matching in responses
Limited: Can't verify code correctness

Improvements:
1. Execute generated code in sandbox
2. Run security scanners on output
3. Use LLM-as-judge for evaluation
4. Human evaluation for edge cases
```

---

## 16. References

### Papers

1. **Attention Is All You Need** (Vaswani et al., 2017)
   - Original transformer architecture
   - https://arxiv.org/abs/1706.03762

2. **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2021)
   - Foundation of parameter-efficient fine-tuning
   - https://arxiv.org/abs/2106.09685

3. **QLoRA: Efficient Finetuning of Quantized LLMs** (Dettmers et al., 2023)
   - Combining quantization with LoRA
   - https://arxiv.org/abs/2305.14314

4. **LLM.int8(): 8-bit Matrix Multiplication** (Dettmers et al., 2022)
   - Foundation of bitsandbytes quantization
   - https://arxiv.org/abs/2208.07339

5. **RoFormer: Enhanced Transformer with Rotary Position Embedding** (Su et al., 2021)
   - RoPE positional encoding
   - https://arxiv.org/abs/2104.09864

### Libraries

- **Transformers**: https://github.com/huggingface/transformers
- **PEFT**: https://github.com/huggingface/peft
- **bitsandbytes**: https://github.com/TimDettmers/bitsandbytes
- **TRL**: https://github.com/huggingface/trl

### Datasets

- **CodeAlpaca-20k**: https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k
- **Code_Vulnerability_Security_DPO**: https://huggingface.co/datasets/CyberNative/Code_Vulnerability_Security_DPO
- **CrossVul**: https://huggingface.co/datasets/hitoshura25/crossvul

---

## Appendix A: Full Training Configuration

```python
# QLoRA Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# LoRA Configuration
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

# Training Configuration
training_args = SFTConfig(
    output_dir="./nimos-coder-agent-v3",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    bf16=True,
    fp16=False,
    logging_steps=25,
    save_strategy="steps",
    save_steps=300,
    save_total_limit=3,
    push_to_hub=True,
    hub_strategy="every_save",
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    dataset_text_field="text",
    max_seq_length=1024,
)
```

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Attention** | Mechanism to weigh importance of different input positions |
| **BFloat16** | Brain Float 16-bit format with extended range |
| **Catastrophic Forgetting** | Loss of previously learned knowledge during fine-tuning |
| **DPO** | Direct Preference Optimization - training from preference pairs |
| **FFN** | Feed-Forward Network - MLP layers in transformer |
| **GQA** | Grouped Query Attention - memory-efficient attention variant |
| **KV Cache** | Key-Value cache for efficient autoregressive generation |
| **LoRA** | Low-Rank Adaptation - parameter-efficient fine-tuning method |
| **NF4** | Normal Float 4-bit - quantization format for normal distributions |
| **PEFT** | Parameter-Efficient Fine-Tuning |
| **QLoRA** | Quantized LoRA - combines 4-bit quantization with LoRA |
| **RMSNorm** | Root Mean Square Normalization |
| **RoPE** | Rotary Position Embedding |
| **SFT** | Supervised Fine-Tuning |
| **SwiGLU** | Swish-Gated Linear Unit activation function |

---

*Document version: 1.0*
*Last updated: December 2024*
*Author: Nimo*
