import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

# Load the model and tokenizer
model_name = "gpt2"  # GPT-2 small model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Check if MPS (Apple's Metal backend) is available and use it if possible
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model = model.to(device)

# Define a sample prompt
prompt = "Once upon a time in a distant land, there was a"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Function to calculate latency and generate text
def generate_text(input_ids, model, tokenizer, max_length=50):
    # Start the timer
    start_time = time.time()
    
    # Generate text
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)

    # End the timer
    latency = time.time() - start_time

    # Decode the output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text, latency

# Generate text and calculate latency
generated_text, latency = generate_text(input_ids, model, tokenizer)

print("Generated Text:\n", generated_text)
print(f"Latency: {latency:.4f} seconds")
