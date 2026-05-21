import torch
from src.light_model.t2i_imagenet_wrapper import T2IImageNetWrapper

device = "cuda" if torch.cuda.is_available() else "cpu"

wrapper = T2IImageNetWrapper(device=device)

# Image factice normalisée dans [-1, 1]
x = torch.randn(1, 3, 256, 256).to(device).clamp(-1, 1)

x_hat = wrapper.reconstruct_image(x)
error_map = wrapper.reconstruction_error_map(x)

print("x shape:", x.shape)
print("x_hat shape:", x_hat.shape)
print("error_map shape:", error_map.shape)

print("x min/max:", x.min().item(), x.max().item())
print("x_hat min/max:", x_hat.min().item(), x_hat.max().item())
print("error_map min/max:", error_map.min().item(), error_map.max().item())