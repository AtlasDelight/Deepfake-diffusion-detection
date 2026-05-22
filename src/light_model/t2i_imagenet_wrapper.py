import torch
from t2i_imagenet import T2IPipeline


class T2IImageNetWrapper:
    """
    Wrapper pour utiliser T2I-ImageNet comme modèle génératif léger.

    Première version :
    - accès au VAE Stable Diffusion utilisé par le pipeline ;
    - reconstruction image x -> z -> x_hat ;
    - extraction de cartes d'erreur |x - x_hat|.
    """

    def __init__(self, model_name="Lucasdegeorge/CAD-I", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_name = model_name

        self.pipe = T2IPipeline(model_name)
        self.pipe.device = device

        # Composant VAE trouvé dans l'inspection
        self.vae = self.pipe.postprocessing.vae.to(device)
        self.vae.eval()

        # Facteur classique des VAE Stable Diffusion
        self.scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)

    @torch.no_grad()
    def encode_image(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """
        Encode une image dans l'espace latent du VAE.

        Args:
            x: image [B, 3, H, W], normalisée dans [-1, 1]
            sample: si True, échantillonne dans la distribution latente.
                    si False, utilise le mode/moyenne pour une reconstruction stable.
        """
        x = x.to(self.device)
        posterior = self.vae.encode(x).latent_dist

        if sample:
            z = posterior.sample()
        else:
            z = posterior.mode()

        z = z * self.scaling_factor
        return z

    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        """
        Décode un latent VAE vers une image.

        Sortie :
            x_hat : tenseur [B, 3, H, W], normalement proche de [-1, 1]
        """
        z = z.to(self.device)
        x_hat = self.vae.decode(z / self.scaling_factor).sample

        if clamp:
            x_hat = x_hat.clamp(-1, 1)

        return x_hat

    @torch.no_grad()
    def reconstruct_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruction image :
            x -> z -> x_hat
        """
        z = self.encode_image(x)
        x_hat = self.decode_latent(z,clamp=True)
        return x_hat

    @torch.no_grad()
    def reconstruction_error_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Carte d'erreur image :
            M(x) = |x - x_hat|
        """
        x = x.to(self.device)
        x_hat = self.reconstruct_image(x)
        error_map = torch.abs(x - x_hat)
        return error_map
    
    @torch.no_grad()
    def reconstruct_latent_via_vae_cycle(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruction latente par cycle VAE :

            x -> z
            z -> x_hat
            x_hat -> z_hat

        Returns:
            z: latent original
            z_hat: latent reconstruit après décodage/ré-encodage
        """
        x = x.to(self.device)

        z = self.encode_image(x, sample=False)
        x_hat = self.decode_latent(z, clamp=True)
        z_hat = self.encode_image(x_hat, sample=False)

        return z, z_hat


    @torch.no_grad()
    def latent_reconstruction_error_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Carte d'erreur latente :

            M(x) = |z - z_hat|
        """
        z, z_hat = self.reconstruct_latent_via_vae_cycle(x)
        error_map = torch.abs(z - z_hat)
        return error_map