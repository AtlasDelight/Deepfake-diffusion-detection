import torch


def _stats_per_sample(x: torch.Tensor) -> dict:
    """
    Compute scalar statistics for each sample in a batch.
    """
    x = x.detach().float()
    flat = x.flatten(start_dim=1)

    return {
        "mean": flat.mean(dim=1),
        "std": flat.std(dim=1, unbiased=False),
        "l1": flat.abs().mean(dim=1),
        "l2": torch.sqrt((flat ** 2).mean(dim=1) + 1e-12),
        "max": flat.abs().max(dim=1).values,
    }


@torch.no_grad()
def extract_noise_prediction_single(
    x: torch.Tensor,
    wrapper,
    gamma: float = 0.5,
    prompt: str = "a photo",
    noise: torch.Tensor | None = None,
    return_maps: bool = True,
) -> dict:
    """
    Single-step noise prediction signal for T2I-ImageNet / CAD-I.

    Important:
        For CAD-I, the predicted noise is a latent-space signal.
        The model does not directly predict pixel noise from x.
        It predicts noise from a noisy latent.

    Pipeline:
        image x
        -> encode_image(x)
        -> latent z
        -> add_noise(z, gamma)
        -> noisy latent z_gamma
        -> predict_noise(z_gamma, gamma, prompt)
        -> epsilon_hat
        -> noise error |epsilon - epsilon_hat|

    """
    x = x.to(wrapper.device)

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

    output = {
        "scores": _stats_per_sample(noise_error),
        "response_scores": _stats_per_sample(epsilon_hat),
    }

    if return_maps:
        output.update(
            {
                "z": z,
                "z_gamma": z_gamma,
                "epsilon": epsilon,
                "epsilon_hat": epsilon_hat,
                "noise_error": noise_error,
            }
        )

    return output
def _stack_score_dict(score_dicts: list[dict]) -> dict:
    """
    Stack score dictionaries along the gamma dimension.

    Each score tensor has shape [B].
    Output tensors have shape [B, K].
    """
    keys = score_dicts[0].keys()

    return {
        key: torch.stack([scores[key] for scores in score_dicts], dim=1)
        for key in keys
    }


@torch.no_grad()
def extract_noise_prediction_trajectory(
    x: torch.Tensor,
    wrapper,
    gammas: list[float] | tuple[float, ...] = (0.95, 0.85, 0.70, 0.55, 0.40, 0.25, 0.10),
    prompt: str = "a photo",
    noise: torch.Tensor | None = None,
    fixed_noise: bool = True,
    return_maps: bool = False,
) -> dict:
    """
    Multi-step / multi-gamma noise prediction trajectory.

    For each gamma:
        image x
        -> latent z
        -> noisy latent z_gamma
        -> true noise epsilon
        -> predicted noise epsilon_hat
        -> noise error |epsilon - epsilon_hat|

    Important:
        For T2I-ImageNet / CAD-I, this is a latent-space noise trajectory.

    Returns:
        {
            "gammas": tensor [K],
            "scores": noise error statistics, shape [B, K],
            "response_scores": predicted noise statistics, shape [B, K]
        }
    """
    device = getattr(wrapper, "device", x.device)
    x = x.to(device)

    z = wrapper.encode_image(x, sample=False)

    if noise is not None:
        base_noise = noise.to(device=z.device, dtype=z.dtype)
    elif fixed_noise:
        base_noise = torch.randn_like(z)
    else:
        base_noise = None

    noise_error_score_list = []
    response_score_list = []

    if return_maps:
        z_gammas = []
        epsilons = []
        epsilon_hats = []
        noise_errors = []

    for gamma in gammas:
        current_noise = base_noise if fixed_noise or noise is not None else None

        z_gamma, epsilon, gamma_tensor = wrapper.add_noise(
            z=z,
            noise=current_noise,
            gamma=gamma,
        )

        epsilon_hat = wrapper.predict_noise(
            z_t=z_gamma,
            gamma=gamma_tensor,
            prompt=prompt,
        )

        noise_error = torch.abs(epsilon - epsilon_hat)

        noise_error_score_list.append(_stats_per_sample(noise_error))
        response_score_list.append(_stats_per_sample(epsilon_hat))

        if return_maps:
            z_gammas.append(z_gamma.detach().cpu())
            epsilons.append(epsilon.detach().cpu())
            epsilon_hats.append(epsilon_hat.detach().cpu())
            noise_errors.append(noise_error.detach().cpu())

    output = {
        "gammas": torch.tensor(gammas, dtype=torch.float32),
        "scores": _stack_score_dict(noise_error_score_list),
        "response_scores": _stack_score_dict(response_score_list),
    }

    if return_maps:
        output.update(
            {
                "z": z.detach().cpu(),
                "z_gammas": torch.stack(z_gammas, dim=1),
                "epsilons": torch.stack(epsilons, dim=1),
                "epsilon_hats": torch.stack(epsilon_hats, dim=1),
                "noise_errors": torch.stack(noise_errors, dim=1),
            }
        )

    return output