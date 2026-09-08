import torch
import torch_directml

device = torch_directml.device()
print(f"Device: {device}")

x = torch.rand(3, 3).to(device)
y = torch.rand(3, 3).to(device)
z = x @ y

print("Test matrix multiply result:")
print(z)
print("DirectML is working.")