import os
from huggingface_hub import snapshot_download

# Configuration
model_name = "meta-llama/Llama-3.1-8B-Instruct"
revision = "0e9e39f249a16976918f6564b8830bc894c89659"
output_dir = "/home/lqliu/mlsys/huggingface_models/llama_8B"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Download the snapshot
print(f"Downloading snapshot of '{model_name}' with revision '{revision}'...")
snapshot_path = snapshot_download(
    repo_id=model_name,
    revision=revision,
    cache_dir=output_dir,
    local_dir=output_dir,
    local_dir_use_symlinks=False  # Avoid symlinks for a fully self-contained copy
)

print(f"Snapshot downloaded to '{snapshot_path}'.")