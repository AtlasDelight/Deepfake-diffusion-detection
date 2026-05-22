import torch


@torch.no_grad()
def extract_anl_like_features(
    x: torch.Tensor,
    wrapper,
    gamma: float = 0.5,
    prompt=None,
    noise: torch.Tensor | None = None,
):
    """
    Extract noise-related features for ANL-like detection.

    This function does NOT compute attention.
    It only returns the inputs that will be used by a learnable attention module.

    Returns:
        image_error: |x - x_hat| in image space, shape [B, 3, H, W]
        noise_response: predicted noise-like response, shape [B, 4, H_latent, W_latent]
        noise_error: |epsilon - epsilon_hat|, shape [B, 4, H_latent, W_latent]
    """

    x = x.to(wrapper.device)

    # Image reconstruction error
    x_hat = wrapper.reconstruct_image(x).clamp(-1, 1)
    image_error = torch.abs(x - x_hat)

    # Latent noise response
    z = wrapper.encode_image(x, sample=False)

    z_gamma, epsilon, gamma_tensor = wrapper.add_noise(
        z=z,
        noise=noise,
        gamma=gamma,
    )

    epsilon_hat = wrapper.predict_noise(
        z_t=z_gamma,
        gamma=gamma_tensor,
        prompt=prompt,
    )

    noise_error = torch.abs(epsilon - epsilon_hat)

    return {
        "image_error": image_error,
        "noise_response": epsilon_hat,
        "noise_error": noise_error,
    }