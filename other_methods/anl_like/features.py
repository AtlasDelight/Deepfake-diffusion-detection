import torch


@torch.no_grad()
def extract_anl_like_predicted_noise(
    x: torch.Tensor,
    wrapper,
    gamma: float = 0.5,
    prompt=None,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Extract predicted diffusion-noise features for ANL-like detection.

    This function extracts the predicted noise response from the pretrained model.

    Args:
        x:
            Input image tensor [B, 3, H, W], normalized in [-1, 1].

        wrapper:
            Lightweight pretrained generative model wrapper.

        gamma:
            Noise level.

        prompt:
            Optional text prompt.

        noise:
            Optional fixed Gaussian noise.

    Returns:
        predicted_noise:
            Tensor [B, C_latent, H_latent, W_latent].
    """

    x = x.to(wrapper.device)

    z = wrapper.encode_image(x, sample=False)

    z_gamma, _, gamma_tensor = wrapper.add_noise(
        z=z,
        noise=noise,
        gamma=gamma,
    )

    predicted_noise = wrapper.predict_noise(
        z_t=z_gamma,
        gamma=gamma_tensor,
        prompt=prompt,
    )

    return predicted_noise


@torch.no_grad()
def extract_anl_like_features(
    x: torch.Tensor,
    wrapper,
    gamma: float = 0.5,
    prompt=None,
    noise: torch.Tensor | None = None,
):
    """
    Compatibility wrapper.

    This keeps the old function name, but now returns only the signal
    used by the ANL-like classifier: the predicted noise.

    Returns:
        {
            "predicted_noise": epsilon_hat,
            "noise_response": epsilon_hat
        }
    """

    predicted_noise = extract_anl_like_predicted_noise(
        x=x,
        wrapper=wrapper,
        gamma=gamma,
        prompt=prompt,
        noise=noise,
    )

    return {
        "predicted_noise": predicted_noise,
        "noise_response": predicted_noise,
    }