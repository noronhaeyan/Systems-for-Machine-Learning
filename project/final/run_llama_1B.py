import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
model_path = "/home/lqliu/mlsys/huggingface_models/llama_1B"
prompt = "How can I improve my productivity while working from home?"
max_length = 50  # Maximum output length

# Load the model and tokenizer
print(f"Loading model and tokenizer from '{model_path}'...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Ensure the model runs on CPU
device = torch.device("cpu")
model.to(device)

# Tokenize the input prompt
print("Tokenizing input prompt...")
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Generate a response
print("Generating response...")
output_ids = model.generate(
    input_ids,
    max_length=len(input_ids[0]) + max_length,
    do_sample=True,  # Sampling for diversity in output
    top_k=50,        # Top-k sampling for more natural responses
    top_p=0.95,      # Nucleus sampling (top-p)
    temperature=0.7, # Lower temperature for more focused output
    num_return_sequences=1,  # Generate a single response
)

# Decode the generated response
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Response:")
print(response)
