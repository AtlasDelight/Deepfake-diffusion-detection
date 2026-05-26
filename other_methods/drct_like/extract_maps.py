import torch


@torch.no_grad()
def extract_drct_like_reconstruction(
    x: torch.Tensor,
    wrapper,
) -> torch.Tensor:
    """
    Reconstruct an input image using the pretrained lightweight model.

    Args:
        x: image tensor [B, 3, H, W], normalized in [-1, 1]
        wrapper: T2IImageNetWrapper

    Returns:
        x_hat: reconstructed image [B, 3, H, W]
    """

    x = x.to(wrapper.device)
    x_hat = wrapper.reconstruct_image(x)
    x_hat = x_hat.clamp(-1, 1)

    return x_hat