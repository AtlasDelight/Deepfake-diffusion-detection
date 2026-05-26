import torch


@torch.no_grad()
def extract_anl_like_predicted_noise(
    x: torch.Tensor,
    wrapper,
    gamma: float = 0.5,
    prompt=None,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
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