"""
Nimo's Coder Agent v3 - Inference Module

Security-enhanced code generation with vulnerability detection.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from typing import Optional


class CoderAgentV3:
    """Security-enhanced code generation agent."""

    MODEL_ID = "CaptainNimo/nimos-coder-agent-v3"
    BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

    def __init__(self, use_4bit: bool = True, device: str = "auto"):
        """
        Initialize the model.

        Args:
            use_4bit: Use 4-bit quantization (reduces memory)
            device: Device to load model on ("auto", "cuda", "cpu")
        """
        self.device = device
        self.use_4bit = use_4bit
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load the model and tokenizer."""
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.BASE_MODEL_ID,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading model...")
        if self.use_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                self.BASE_MODEL_ID,
                quantization_config=bnb_config,
                device_map=self.device,
                trust_remote_code=True,
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.BASE_MODEL_ID,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
            )

        print("Loading fine-tuned adapter...")
        self.model = PeftModel.from_pretrained(base_model, self.MODEL_ID)
        self.model.eval()

        print("Model loaded successfully!")
        return self

    def generate(
        self,
        instruction: str,
        input_text: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate code from instruction.

        Args:
            instruction: What to generate
            input_text: Optional context/code for review
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling

        Returns:
            Generated response
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Build prompt
        if input_text.strip():
            prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
        else:
            prompt = f"""### Instruction:
{instruction}

### Response:
"""

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract response
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()

        return response

    def review_security(self, code: str) -> str:
        """
        Review code for security vulnerabilities.

        Args:
            code: Code to review

        Returns:
            Security analysis
        """
        return self.generate(
            "Review this code for security vulnerabilities. Is it safe?",
            code
        )

    def fix_security(self, code: str) -> str:
        """
        Fix security vulnerabilities in code.

        Args:
            code: Vulnerable code

        Returns:
            Fixed code
        """
        return self.generate(
            "Fix the security vulnerabilities in this code",
            code
        )

    def add_error_handling(self, code: str) -> str:
        """
        Add error handling to code.

        Args:
            code: Code without error handling

        Returns:
            Code with try-catch blocks
        """
        return self.generate(
            "Add proper error handling to this code",
            code
        )


def main():
    """Demo usage."""
    agent = CoderAgentV3()
    agent.load()

    # Test 1: Security review
    print("\n" + "=" * 60)
    print("TEST: Security Review")
    print("=" * 60)

    vulnerable_code = '''import os
user_input = input("Enter filename: ")
os.system(f"cat {user_input}")'''

    result = agent.review_security(vulnerable_code)
    print(result)

    # Test 2: Generate with error handling
    print("\n" + "=" * 60)
    print("TEST: Generate Code")
    print("=" * 60)

    result = agent.generate("Write a Python function to read a JSON file")
    print(result)


if __name__ == "__main__":
    main()
