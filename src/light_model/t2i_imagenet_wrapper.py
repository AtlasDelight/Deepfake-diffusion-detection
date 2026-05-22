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
    
    
    @torch.no_grad()
    def add_noise(self,z: torch.Tensor,noise: torch.Tensor,timestep: int | float | None = None,) -> torch.Tensor:
        """
        Add Gaussian noise to a latent representation.
        This is a temporary generic noising function.
        Later, this should be aligned with the scheduler used by T2I-ImageNet.
        """

        if timestep is None:
            alpha_t = 0.5
        else:
            # Simple normalized timestep placeholder.
            # This is not yet the official scheduler behavior.
            t = float(timestep)
            if t > 1:
                t = t / 1000.0
            alpha_t = max(0.0, min(1.0, 1.0 - t))

        alpha_t = torch.tensor(alpha_t, device=z.device, dtype=z.dtype)

        z_t = torch.sqrt(alpha_t) * z + torch.sqrt(1.0 - alpha_t) * noise

        return z_t


    @torch.no_grad()
    def predict_noise(self,z_t: torch.Tensor,timestep: int | float | None = None,) -> torch.Tensor:
        """
        Predict noise from a noisy latent.

        This method is not implemented yet because we still need to inspect
        how T2I-ImageNet expects its input batch for network/preconditioning.

        Required for DNF-like and ANL-like methods.
        """

        raise NotImplementedError(
            "predict_noise is not implemented yet. "
            "We need to inspect T2I-ImageNet's network/preconditioning batch format."
        )
        
    @torch.no_grad()
    def build_conditioned_batch(self, z_t: torch.Tensor, gamma: torch.Tensor, prompt=None):
        """
        Build the batch expected by the T2I-ImageNet network.

        The inspected network expects at least:
            batch["y"]
            batch["gamma"]
            batch["previous_latents"]
            batch["coherence"]

        It may also use text-conditioning fields added by cond_preprocessing.
        """

        z_t = z_t.to(self.device)
        gamma = gamma.to(self.device)

        batch_size = z_t.shape[0]

        if prompt is None:
            prompt = ["a photo"] * batch_size
        elif isinstance(prompt, str):
            prompt = [prompt] * batch_size

        batch = {
            "y": z_t,
            "gamma": gamma,
            "previous_latents": None,

            # Required by T2I-ImageNet cond_mapping(...)
            # 1.0 corresponds to a coherent/standard conditioning signal.
            "coherence": torch.ones(
                batch_size,
                device=self.device,
                dtype=torch.float32,
            ),
        }

        # Add text conditioning using the pipeline preprocessing
        batch[self.pipe.cond_preprocessing.input_key] = prompt
        batch = self.pipe.cond_preprocessing(batch, device=self.device)

        return batch


    @torch.no_grad()
    def add_noise(
        self,
        z: torch.Tensor,
        noise: torch.Tensor | None = None,
        gamma: float = 0.5,
    ):
        """
        Add Gaussian noise to a latent representation.

        z_t = sqrt(gamma) * z + sqrt(1 - gamma) * epsilon

        Here gamma is used as a continuous noise-level parameter in [0, 1].
        """

        z = z.to(self.device)

        if noise is None:
            noise = torch.randn_like(z)
        else:
            noise = noise.to(self.device)

        gamma_tensor = torch.full(
            (z.shape[0],),
            float(gamma),
            device=z.device,
            dtype=z.dtype,
        )

        gamma_map = gamma_tensor.view(-1, 1, 1, 1)

        z_t = torch.sqrt(gamma_map) * z + torch.sqrt(1.0 - gamma_map) * noise

        return z_t, noise, gamma_tensor


    @torch.no_grad()
    def predict_noise(
        self,
        z_t: torch.Tensor,
        gamma: torch.Tensor,
        prompt=None,
    ):
        """
        Predict a noise-like model response from a noisy latent.

        This uses the internal T2I-ImageNet network response.

        The model output has the same spatial structure as the latent input.
        We use it as a DNF-like predicted noise signal.
        """

        batch = self.build_conditioned_batch(
            z_t=z_t,
            gamma=gamma,
            prompt=prompt,
        )

        # The preconditioning module internally calls the network
        model_response, previous_latents = self.pipe.preconditioning(
            self.pipe.network,
            batch,
        )

        return model_response
    
    @torch.no_grad()
    def diffuse_latent_with_gamma(
        self,
        z: torch.Tensor,
        noise: torch.Tensor,
        gamma: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to a latent representation with a given gamma.

        Formula:
            z_gamma = sqrt(gamma) * z + sqrt(1 - gamma) * noise

        Returns:
            z_gamma: noisy latent
            gamma_tensor: tensor version of gamma
        """

        z = z.to(self.device)
        noise = noise.to(self.device)

        gamma_tensor = torch.full(
            (z.shape[0],),
            float(gamma),
            device=z.device,
            dtype=z.dtype,
        )

        gamma_map = gamma_tensor.view(-1, 1, 1, 1)

        z_gamma = torch.sqrt(gamma_map) * z + torch.sqrt(1.0 - gamma_map) * noise

        return z_gamma, gamma_tensor


    @torch.no_grad()
    def estimate_clean_latent_from_noise(
        self,
        z_t: torch.Tensor,
        epsilon_hat: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate the clean latent z0 from a noisy latent z_t and a predicted noise.

        Formula:
            z0_hat = (z_t - sqrt(1 - gamma) * epsilon_hat) / sqrt(gamma)
        """

        z_t = z_t.to(self.device)
        epsilon_hat = epsilon_hat.to(self.device)
        gamma = gamma.to(self.device)

        gamma_map = gamma.view(-1, 1, 1, 1)

        z0_hat = (
            z_t - torch.sqrt(1.0 - gamma_map) * epsilon_hat
        ) / (torch.sqrt(gamma_map) + 1e-8)

        return z0_hat


    @torch.no_grad()
    def sedid_reverse_psi(
        self,
        z0: torch.Tensor,
        noise: torch.Tensor,
        gamma_target: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        SeDID-style reverse / diffusion path.

        Starting from the clean latent z0, move to the target noise level.

        Formula:
            z_target_reverse =
                sqrt(gamma_target) * z0
                + sqrt(1 - gamma_target) * noise
        """

        z_target_reverse, gamma_target_tensor = self.diffuse_latent_with_gamma(
            z=z0,
            noise=noise,
            gamma=gamma_target,
        )

        return z_target_reverse, gamma_target_tensor


    @torch.no_grad()
    def sedid_denoise_phi(
        self,
        z_source: torch.Tensor,
        gamma_source: torch.Tensor,
        gamma_target: float,
        prompt=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        SeDID-style denoising path.

        Starting from a noisier latent z_source, the model predicts a noise-like
        response, estimates a clean latent, then moves it back to gamma_target.
        """

        epsilon_hat = self.predict_noise(
            z_t=z_source,
            gamma=gamma_source,
            prompt=prompt,
        )

        z0_hat = self.estimate_clean_latent_from_noise(
            z_t=z_source,
            epsilon_hat=epsilon_hat,
            gamma=gamma_source,
        )

        z_target_denoise, _ = self.diffuse_latent_with_gamma(
            z=z0_hat,
            noise=epsilon_hat,
            gamma=gamma_target,
        )

        return z_target_denoise, epsilon_hat