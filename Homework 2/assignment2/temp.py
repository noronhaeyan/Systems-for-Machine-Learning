# Be sure to set top_k=1, otherwise you'll pollute the RNG! (Which will make your code harder to debug.)

### YOUR CODE HERE
# use the draft_model to generate speculative tokens
idx_speculative = draft_model.generate(idx_cond, num_speculative, temperature=temperature, top_k=top_k)

# obtain the logits from the main model by passing in the idx_speculative
all_logits, _ = self(idx_speculative)
### END YOUR CODE HERE

# Step through the predictions of the main model, sampling, and check whether they match the next token. Stop upon mismatch.

### YOUR CODE HERE

# iterate from the end position of idx_cond (prefix sequence) to the end position of idx_speculative (generated sequence)
for i in range(idx_cond.size(1), idx_speculative.size(1)):
    # pluck the logits at the current position and scale by desired temperature
    logits = all_logits[:, i, :] / temperature

    # optionally crop the logits to only the top k options
    if top_k is not None:
        if top_k == 1:
            idx_next = torch.argmax(logits, dim=-1).reshape((-1, 1))
        else:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

    # Sample from the logits
    probs = torch.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1)

    # Check if the sampled token matches the next token in idx_speculative
    if idx_next != idx_speculative[:, i].unsqueeze(-1):
        break

    # Append the sampled token to the sequence
    idx_cond = torch.cat((idx_cond, idx_next), dim=1)

### END YOUR CODE HERE