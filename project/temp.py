import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import matplotlib.pyplot as plt


# Generate text and calculate latency

# max_length = [13, 15, 17, 20, 25]
# output_latency = []
# for length in max_length:
#     generated_text, latency = generate_text(input_ids, model, tokenizer, max_length=length)
#     output_latency.append(latency)
#     print(f"Max Length: {length}, Latency: {latency:.4f} seconds")

# Plotting the latency vs max_length

max_length = [13, 15, 17, 20, 25]
output_latency = [222, 455, 678, 1234, 1762]
model_name = "meta-llama/Llama-3.1-8B-Instruct"

plt.figure(figsize=(10, 6))
plt.plot(max_length, output_latency, marker='o')
plt.title('Latency vs Output Token Length for %s Text Generation' % model_name)
plt.xlabel('Output Token Length')
plt.ylabel('Latency (seconds)')
plt.grid(True)
plt.show()
