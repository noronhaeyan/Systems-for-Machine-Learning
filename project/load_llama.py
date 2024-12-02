import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import time
import matplotlib.pyplot as plt

# Set model size
model_name = "meta-llama/Llama-1b-hf"  # Use "meta-llama/Llama-8b-hf" for 8B if you have enough memory

# Load tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# Move model to MPS if available, otherwise to CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model = model.to(device)

# Define a sample prompt
prompt = "In a quiet village nestled between mountains and rivers, a curious child found a mysterious"
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

# Generate text and calculate latency
max_length = [20, 50, 100, 150 ,200]
output_latency = []
for length in max_length:
    generated_text, latency = generate_text(input_ids, model, tokenizer, max_length=length)
    output_latency.append(latency)
    print(f"Max Length: {length}, Latency: {latency:.4f} seconds")

# Plotting the latency vs max_length
plt.figure(figsize=(10, 6))
plt.plot(max_length, output_latency, marker='o')
plt.title('Latency vs Output Token Length for %s Text Generation' % model_name)
plt.xlabel('Output Token Length')
plt.ylabel('Latency (seconds)')
plt.grid(True)
plt.show()
# print("Generated Text:\n", generated_text)
# print(f"Latency: {latency:.4f} seconds")
