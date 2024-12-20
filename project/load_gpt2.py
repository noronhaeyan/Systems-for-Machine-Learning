import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time
import matplotlib.pyplot as plt

# Load the model and tokenizer
model_name = "gpt2-xl"  # GPT-2 small model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Check if MPS (Apple's Metal backend) is available and use it if possible
if torch.backends.mps.is_available():
#if False:
    print("Metal Performance Shaders (MPS) is available.")
    device = torch.device("mps")
else:
    print("Defaulting to CPU.")
    device = torch.device("cpu")

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
