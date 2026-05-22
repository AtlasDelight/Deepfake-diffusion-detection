import torch


@torch.no_grad()
def extract_dire_like_map(x: torch.Tensor, wrapper) -> torch.Tensor:
    """
    Extract a DIRE-like reconstruction error map.

    Principle:
        x -> x_hat
        M(x) = |x - x_hat|

    Args:
        x:
            Input image tensor of shape [B, 3, H, W].
            The image must be normalized in [-1, 1].

        wrapper:
            Lightweight pretrained generative model wrapper.
            It must implement reconstruct_image(x).

    Returns:
        error_map:
            Reconstruction error map of shape [B, 3, H, W].
    """

    x_hat = wrapper.reconstruct_image(x)

    # Safety clamp, because decoded VAE outputs may slightly exceed [-1, 1]
    x_hat = x_hat.clamp(-1, 1)

    error_map = torch.abs(x - x_hat)

    return error_map


@torch.no_grad()
def extract_dire_like_gray_map(x: torch.Tensor, wrapper) -> torch.Tensor:
    """
    Extract a single-channel DIRE-like map.

    This averages the RGB reconstruction error over channels.

    Returns:
        gray_error_map: [B, 1, H, W]
    """

    error_map = extract_dire_like_map(x, wrapper)
    gray_error_map = error_map.mean(dim=1, keepdim=True)

    return gray_error_map