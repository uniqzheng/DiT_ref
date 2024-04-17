import torch
from diffusers.models import AutoencoderKL

# Load the model
vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").cuda()

# Calculate the total number of bytes of all parameters
total_bytes = sum(p.numel() * p.element_size() for p in vae.parameters())

# Convert bytes to megabytes
total_mb = total_bytes / (1024 ** 2)

print(f'Total size of model weights: {total_mb:.2f} MB')