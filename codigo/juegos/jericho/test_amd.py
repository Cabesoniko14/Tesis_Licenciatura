import torch

# Check if ROCm is available
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Use Metal Performance Shaders for macOS
elif torch.cuda.is_available():
    device = torch.device("cuda")  # Use CUDA for NVIDIA GPUs
else:
    device = torch.device("cpu")    # Fallback to CPU

# Print device information
print(f"Using device: {device}")

# Create a simple tensor and move it to the selected device
x = torch.rand(5, 5).to(device)
print(f"Tensor on {device}:")
print(x)

# Perform a simple operation on the tensor
y = x + 2
print(f"Result after adding 2:")
print(y)
aaaaaa