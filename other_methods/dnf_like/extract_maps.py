import torch
import torch.nn.functional as F


@torch.no_grad()
def extract_dnf_like_map(
    x: torch.Tensor,
    wrapper,
    gamma: float = 0.5,
    prompt=None,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Extract a DNF-like noise response error map.

    Principle:
        x -> z
        z_t = sqrt(gamma) z + sqrt(1 - gamma) epsilon
        epsilon_hat = model_response(z_t, gamma, prompt)
        M(x) = |epsilon - epsilon_hat|

    This is a lightweight DNF-inspired signal. It is not the official DNF
    implementation.
    """

    z = wrapper.encode_image(x, sample=False)

    z_t, epsilon, gamma_tensor = wrapper.add_noise(
        z=z,
        noise=noise,
        gamma=gamma,
    )

    epsilon_hat = wrapper.predict_noise(
        z_t=z_t,
        gamma=gamma_tensor,
        prompt=prompt,
    )

    # Safety check: if the model response has a slightly different scale,
    # the map is still computed as a response error.
    noise_error_map = torch.abs(epsilon - epsilon_hat)

    return noise_error_map


@torch.no_grad()
def extract_dnf_like_gray_map(
    x: torch.Tensor,
    wrapper,
    gamma: float = 0.5,
    prompt=None,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Convert the DNF-like noise error map into a single-channel map.

    Returns:
        gray_map: [B, 1, H_latent, W_latent]
    """

    noise_map = extract_dnf_like_map(
        x=x,
        wrapper=wrapper,
        gamma=gamma,
        prompt=prompt,
        noise=noise,
    )

    gray_map = noise_map.mean(dim=1, keepdim=True)

    return gray_map


@torch.no_grad()
def extract_dnf_like_upsampled_map(
    x: torch.Tensor,
    wrapper,
    gamma: float = 0.5,
    prompt=None,
    noise: torch.Tensor | None = None,
    size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """
    Extract a DNF-like map and upsample it to image resolution.

    Returns:
        upsampled_map: [B, 1, H, W]
    """

    gray_map = extract_dnf_like_gray_map(
        x=x,
        wrapper=wrapper,
        gamma=gamma,
        prompt=prompt,
        noise=noise,
    )

    if size is None:
        size = x.shape[-2:]

    upsampled_map = F.interpolate(
        gray_map,
        size=size,
        mode="bilinear",
        align_corners=False,
    )

    return upsampled_map