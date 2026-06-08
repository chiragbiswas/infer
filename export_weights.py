"""
Generates a small MLP with random weights and saves them to weights.bin.

Format (little-endian float32):
  [num_layers: int32]
  for each layer:
    [in_features: int32] [out_features: int32]
    [W: in*out float32s, row-major]
    [b: out float32s]

Also saves a test input and expected output so we can verify C++ matches.
"""
import numpy as np
import struct

np.random.seed(41)

# Architecture must match what you build in main.cpp
layer_sizes = [784, 256, 128, 10]

layers = []
for i in range(len(layer_sizes) - 1):
    fan_in  = layer_sizes[i]
    fan_out = layer_sizes[i + 1]
    # Xavier uniform init — same as PyTorch's default Linear
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    W = np.random.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)
    b = np.zeros(fan_out, dtype=np.float32)
    layers.append((W, b))

with open("weights.bin", "wb") as f:
    f.write(struct.pack("<i", len(layers)))
    for W, b in layers:
        in_f, out_f = W.shape
        f.write(struct.pack("<ii", in_f, out_f))
        f.write(W.tobytes())
        f.write(b.tobytes())

print(f"Wrote weights.bin  ({len(layers)} layers: {layer_sizes})")

# --- Generate a test input and run numpy forward pass ---
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

# Single test input
x = np.random.randn(1, 784).astype(np.float32)

out = x
for i, (W, b) in enumerate(layers):
    out = out @ W + b
    if i < len(layers) - 1:
        out = relu(out)
softmax_out = softmax(out)

with open("test_input.bin", "wb") as f:
    f.write(x.tobytes())

# Save expected output
with open("test_output.bin", "wb") as f:
    f.write(softmax_out.tobytes())

print("Wrote test_input.bin and test_output.bin")
print("Expected output:", softmax_out[0])
print("Predicted class:", softmax_out[0].argmax())
