from llama_cpp import Llama
import time

# Configuration
model_path = "/home/lqliu/mlsys/huggingface_models/llama_1B_q8_0.gguf"
prompt = "How can I improve my productivity while working from home?"
max_tokens = 50  # Maximum tokens in the response
temperature = 0.7  # Adjust temperature for response diversity
top_p = 0.95  # Nucleus sampling

# Load the model
print(f"Loading model from '{model_path}'...")
llm = Llama(
    model_path=model_path,
    n_gpu_layers=0)

# Run inference
print("Running inference...")
start_time = time.time()
response = llm(
    prompt,
    max_tokens=max_tokens,
    temperature=temperature,
    top_p=top_p,
    echo=False,  # Do not repeat the input prompt in the output
)
latency = time.time() - start_time
print(f"Latency: {latency:.4f} seconds")

# Output the response
print("Response:")
print(response["choices"][0]["text"].strip())
