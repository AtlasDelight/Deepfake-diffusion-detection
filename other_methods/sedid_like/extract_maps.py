import torch


@torch.no_grad()
def extract_sedid_like_map(
    x: torch.Tensor,
    wrapper,
    gamma_source: float = 0.5,
    gamma_target: float = 0.8,
    prompt=None,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Extract a SeDID-like error map in image space.

    Logic:
        1. Encode x into z0
        2. Build a noisier latent z_source at gamma_source
        3. Reverse path:
              z0 -> z_target_reverse
        4. Denoising path:
              z_source -> z0_hat -> z_target_denoise
        5. Decode both target states back to image space
        6. Compute the image-space error map:
              M(x) = |x_target_reverse - x_target_denoise|

    Output:
        image_error_map: [B, 3, H, W]
    """

    if gamma_source >= gamma_target:
        raise ValueError(
            "gamma_source should be smaller than gamma_target "
            "so that the denoising path starts from a noisier state."
        )

    # Clean latent
    z0 = wrapper.encode_image(x, sample=False)

    # Shared Gaussian noise
    if noise is None:
        noise = torch.randn_like(z0)
    else:
        noise = noise.to(wrapper.device)

    # Source noisy latent
    z_source, gamma_source_tensor = wrapper.diffuse_latent_with_gamma(
        z=z0,
        noise=noise,
        gamma=gamma_source,
    )

    # Reverse / diffusion path
    z_target_reverse, _ = wrapper.sedid_reverse_psi(
        z0=z0,
        noise=noise,
        gamma_target=gamma_target,
    )

    # Denoising path
    z_target_denoise, _ = wrapper.sedid_denoise_phi(
        z_source=z_source,
        gamma_source=gamma_source_tensor,
        gamma_target=gamma_target,
        prompt=prompt,
    )

    # Decode both states back to image space
    x_target_reverse = wrapper.decode_latent(z_target_reverse, clamp=True)
    x_target_denoise = wrapper.decode_latent(z_target_denoise, clamp=True)

    # Final SeDID-like image-space error map
    image_error_map = torch.abs(x_target_reverse - x_target_denoise)

    return image_error_map


@torch.no_grad()
def extract_sedid_like_gray_map(
    x: torch.Tensor,
    wrapper,
    gamma_source: float = 0.5,
    gamma_target: float = 0.8,
    prompt=None,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Convert the RGB SeDID-like image error map into a single-channel map.

    Returns:
        gray_map: [B, 1, H, W]
    """

    image_error_map = extract_sedid_like_map(
        x=x,
        wrapper=wrapper,
        gamma_source=gamma_source,
        gamma_target=gamma_target,
        prompt=prompt,
        noise=noise,
    )

    gray_map = image_error_map.mean(dim=1, keepdim=True)

    return gray_map