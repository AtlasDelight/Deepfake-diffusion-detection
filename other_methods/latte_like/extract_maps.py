import torch


@torch.no_grad()
def extract_latte_like_sequence(
    x: torch.Tensor,
    wrapper,
    gammas: list[float] | None = None,
    prompt=None,
    shared_noise: bool = True,
) -> torch.Tensor:
    """
    Extract a LATTE-like latent trajectory sequence.

    Principle:
        x -> z0

        For each gamma:
            z_gamma = sqrt(gamma) * z0 + sqrt(1 - gamma) * epsilon
            epsilon_hat = f_theta(z_gamma, gamma)
            z0_hat_gamma = single-step reconstructed latent

        Final output:
            sequence = [z0_hat_gamma1, z0_hat_gamma2, ..., z0_hat_gammaK]

    Output shape:
        [B, K, C_latent, H_latent, W_latent]

    This is a LATTE-inspired approximation. The goal is to preserve the
    trajectory structure instead of flattening all timesteps into channels.
    """

    if gammas is None:
        gammas = [0.2, 0.4, 0.6, 0.8]

    z0 = wrapper.encode_image(x, sample=False)

    if shared_noise:
        base_noise = torch.randn_like(z0)
    else:
        base_noise = None

    sequence = []

    for gamma in gammas:
        noise = base_noise if shared_noise else torch.randn_like(z0)

        z_gamma, epsilon, gamma_tensor = wrapper.add_noise(
            z=z0,
            noise=noise,
            gamma=gamma,
        )

        epsilon_hat = wrapper.predict_noise(
            z_t=z_gamma,
            gamma=gamma_tensor,
            prompt=prompt,
        )

        z0_hat = wrapper.estimate_clean_latent_from_noise(
            z_t=z_gamma,
            epsilon_hat=epsilon_hat,
            gamma=gamma_tensor,
        )

        sequence.append(z0_hat)

    sequence = torch.stack(sequence, dim=1)

    return sequence


@torch.no_grad()
def extract_latte_like_response_sequence(
    x: torch.Tensor,
    wrapper,
    gammas: list[float] | None = None,
    prompt=None,
    shared_noise: bool = True,
) -> torch.Tensor:
    """
    Extract a sequence of model noise-like responses.

    Instead of storing reconstructed latents z0_hat, this function stores
    the predicted model responses epsilon_hat at several gamma levels.

    For each gamma:
        epsilon_hat_gamma = f_theta(z_gamma, gamma)

    Output shape:
        [B, K, C_latent, H_latent, W_latent]
    """

    if gammas is None:
        gammas = [0.2, 0.4, 0.6, 0.8]

    z0 = wrapper.encode_image(x, sample=False)

    if shared_noise:
        base_noise = torch.randn_like(z0)
    else:
        base_noise = None

    responses = []

    for gamma in gammas:
        noise = base_noise if shared_noise else torch.randn_like(z0)

        z_gamma, epsilon, gamma_tensor = wrapper.add_noise(
            z=z0,
            noise=noise,
            gamma=gamma,
        )

        epsilon_hat = wrapper.predict_noise(
            z_t=z_gamma,
            gamma=gamma_tensor,
            prompt=prompt,
        )

        responses.append(epsilon_hat)

    responses = torch.stack(responses, dim=1)

    return responses