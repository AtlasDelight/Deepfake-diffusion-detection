import torch


@torch.no_grad()
def extract_vae_reconstruction_map(x: torch.Tensor, wrapper) -> torch.Tensor:
    """
    Extrait une carte d'erreur de reconstruction image.

    Principe :
        x -> z -> x_hat
        M(x) = |x - x_hat|
    """
    return wrapper.reconstruction_error_map(x)