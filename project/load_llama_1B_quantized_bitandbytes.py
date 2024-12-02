import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import matplotlib.pyplot as plt


# Specify the model name
model_name = "meta-llama/Llama-3.2-1B-Instruct"

# To get access to a gated model on huggingface
# Type: "huggingface-cli login" on your teminal and 
# input your access token

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load the model with INT8 quantization using bitsandbytes
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to CPU
# Load the model with bitsandbytes in 8-bit quantization mode
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,  # Enable 8-bit quantization
    device_map="auto"   # Automatically assign devices
)

# Move model to the appropriate device (CPU or MPS if compatible)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


# Define a sample prompt
prompt = "How can I improve my productivity while working from home?"
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

# #Generate text and calculate latency

# max_length = [25, 50, 100, 150, 200]
# output_latency = []
# for length in max_length:
#     generated_text, latency = generate_text(input_ids, model, tokenizer, max_length=length)
#     output_latency.append(latency)
#     print(f"Max Length: {length}, Latency: {latency:.4f} seconds")

# # Plotting the latency vs max_length
# plt.figure(figsize=(10, 6))
# plt.plot(max_length, output_latency, marker='o')
# plt.title('Latency vs Output Token Length for %s Text Generation' % model_name)
# plt.xlabel('Output Token Length')
# plt.ylabel('Latency (seconds)')
# plt.grid(True)
# plt.show()


generated_text, latency = generate_text(input_ids, model, tokenizer, max_length=200)

print("Generated Text:\n", generated_text)
print(f"Latency: {latency:.4f} seconds")
