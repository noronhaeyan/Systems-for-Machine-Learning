# Small enough to run on CPU for development.
# Use this just to test that your code does not crash before running on Colab GPU to reduce GPU usage.

dataset=lm_synthetic 
epochs=2
lr=0.0005
train_batch=2
input_seq_len=8
d_model=64

# Check if python or python3 is available
if command -v python3 &>/dev/null; then
    PYTHON_CMD=python3
elif command -v python &>/dev/null; then
    PYTHON_CMD=python
else
    echo "Python is not installed."
    exit 1
fi

$PYTHON_CMD src/experiments.py \
    --task ${dataset} \
    --epochs ${epochs} \
    --learning_rate ${lr} \
    --train_batch ${train_batch} \
    --input_seq_len ${input_seq_len} \
    --d_model ${d_model}