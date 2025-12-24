"""
Nimo's Coder Agent v3 - Security Enhanced

A fine-tuned LLM for code generation and security vulnerability detection.
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Configuration - V3 Security Enhanced
MODEL_ID = "CaptainNimo/nimos-coder-agent-v3"
BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

# Global variables
model = None
tokenizer = None


def load_model():
    """Load the fine-tuned model."""
    global model, tokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )

    print("Loading fine-tuned adapter...")
    model = PeftModel.from_pretrained(base_model, MODEL_ID)
    model.eval()

    print("Model loaded successfully!")
    return model, tokenizer


def generate_code(instruction: str, context: str = "", max_tokens: int = 256, temperature: float = 0.7):
    """Generate code from instruction."""
    global model, tokenizer

    if model is None:
        return "Model is loading, please wait..."

    # Build prompt
    if context.strip():
        prompt = f"""### Instruction:
{instruction}

### Input:
{context}

### Response:
"""
    else:
        prompt = f"""### Instruction:
{instruction}

### Response:
"""

    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    return response


# Example prompts - including security examples
EXAMPLES = [
    # Security review examples (NEW in v3!)
    ["Review this code for security vulnerabilities. Is it safe?", "import os\nuser_input = input('Enter filename: ')\nos.system(f'cat {user_input}')"],
    ["Is this code secure?", 'query = f"SELECT * FROM users WHERE id = {user_id}"'],
    ["Fix the security vulnerabilities in this code", "import os\nos.system(f'rm {filename}')"],
    # General coding
    ["Write a Python function to check if a number is prime", ""],
    ["Create a JavaScript function to debounce API calls", ""],
    ["Write a SQL query to find the top 5 customers by sales", ""],
    # Code improvement
    ["Add error handling to this function", "def divide(a, b):\n    return a / b"],
]

# Load model at startup
print("Initializing Nimo's Coder Agent v3 - Security Enhanced...")
load_model()

# Create interface
with gr.Blocks(title="Nimo's Coder Agent v3", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Nimo's Coder Agent v3 - Security Enhanced

        A fine-tuned LLM for **code generation** and **security vulnerability detection**.

        **What's new in v3:**
        - Detects command injection, SQL injection vulnerabilities
        - Trained on 25k+ examples including security datasets
        - 81% token accuracy

        **Model**: Qwen2.5-Coder-0.5B + QLoRA | **Training**: CodeAlpaca + Security DPO + CrossVul

        [GitHub](https://github.com/CaptainNimo/nimos-coder-v3-security) |
        [Model](https://huggingface.co/CaptainNimo/nimos-coder-agent-v3) |
        [v2 (Previous)](https://huggingface.co/CaptainNimo/nimos-coder-agent-v2)

        ---
        **Try the security review!** Paste vulnerable code and ask "Is this code safe?"
        """
    )

    with gr.Row():
        with gr.Column():
            instruction = gr.Textbox(
                label="What do you need?",
                placeholder="e.g., Review this code for security vulnerabilities...",
                lines=2
            )
            context = gr.Textbox(
                label="Code to Review/Context (optional)",
                placeholder="Paste code here for security review, debugging, or refactoring...",
                lines=6
            )
            with gr.Row():
                max_tokens = gr.Slider(64, 512, value=256, step=32, label="Max Length")
                temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Creativity")

            btn = gr.Button("Generate / Review", variant="primary")

        with gr.Column():
            output = gr.Textbox(label="Response", lines=18)

    gr.Examples(examples=EXAMPLES, inputs=[instruction, context])

    btn.click(generate_code, inputs=[instruction, context, max_tokens, temperature], outputs=output)

    gr.Markdown(
        """
        ---
        **Note:** While v3 is better at detecting vulnerabilities than v2, always have security-critical code reviewed by experts.

        *Fine-tuned by Nimo using QLoRA on free Google Colab T4 GPU (2.8 hours)*
        """
    )

demo.launch()
