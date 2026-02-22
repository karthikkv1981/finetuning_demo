import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys

MODEL_ID = "google/gemma-3-270m-it"
ADAPTER_DIR = "gemma3-270m-email-lora-adapter"

def print_header(text):
    print("\n" + "="*50)
    print(f"🚀 {text.upper()}")
    print("="*50)

print_header("Initializing Interactive Inference")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading base model to MPS (Apple Silicon Acceleration)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32, # Safe precision for Mac
).to("mps")

try:
    print(f"Applying your fine-tuned LoRA adapters from '{ADAPTER_DIR}'...")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
except Exception as e:
    print(f"\n❌ ERROR: Could not find the trained adapter folder '{ADAPTER_DIR}'.")
    print("Please run 'python3 interactive_train.py' first to train the model!")
    sys.exit()

model.eval()

print_header("Model Ready!")
print("Type a 'blunt' email and the model will rewrite it professionally.")
print("Type 'quit' or 'exit' to stop.")

while True:
    user_input = input("\n[BLUNT EMAIL]: ").strip()
    
    if user_input.lower() in ["quit", "exit"]:
        print("Goodbye!")
        break
    
    if not user_input:
        continue

    # Format the prompt using the Gemma Chat Template
    prompt = f"Rewrite professionally: {user_input}"
    messages = [{"role": "user", "content": prompt}]
    
    # Pre-process
    inputs = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to("mps")

    # Generate
    print("Generating...")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False, # Use greedy decoding for MPS stability
        )

    # Decode
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = out[0][input_length:]
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    print(f"\n[PROFESSIONAL VERSION]:")
    print("-" * 30)
    print(output_text if output_text else "(Model produced an empty output. Try a different phrasing.)")
    print("-" * 30)
