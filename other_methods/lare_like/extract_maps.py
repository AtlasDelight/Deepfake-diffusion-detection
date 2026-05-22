import torch
import torch.nn.functional as F


@torch.no_grad()
def extract_lare_like_map(x: torch.Tensor, wrapper) -> torch.Tensor:
    """
    Extract a LaRE-like latent reconstruction error map.

    Principle:
        x -> z
        z -> x_hat
        x_hat -> z_hat

        M(x) = |z - z_hat|

    Args:
        x:
            Input image tensor [B, 3, H, W], normalized in [-1, 1].

        wrapper:
            Lightweight pretrained generative model wrapper.
            It must implement latent_reconstruction_error_map(x).

    Returns:
        latent_error_map:
            Tensor [B, C_latent, H_latent, W_latent].
    """

    return wrapper.latent_reconstruction_error_map(x)


@torch.no_grad()
def extract_lare_like_gray_map(x: torch.Tensor, wrapper) -> torch.Tensor:
    """
    Convert the latent reconstruction error map into a single-channel map.

    Returns:
        gray_map: [B, 1, H_latent, W_latent]
    """

    latent_map = extract_lare_like_map(x, wrapper)
    gray_map = latent_map.mean(dim=1, keepdim=True)

    return gray_map


@torch.no_grad()
def extract_lare_like_upsampled_map(
    x: torch.Tensor,
    wrapper,
    size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """
    Extract a single-channel LaRE-like map and upsample it to image resolution.

    Useful for visualization or for using the same CNN input size as image-level maps.

    Returns:
        upsampled_map: [B, 1, H, W]
    """

    gray_map = extract_lare_like_gray_map(x, wrapper)

    if size is None:
        size = x.shape[-2:]

    upsampled_map = F.interpolate(
        gray_map,
        size=size,
        mode="bilinear",
        align_corners=False,
    )

    return upsampled_map