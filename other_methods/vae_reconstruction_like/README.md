# VAE Reconstruction-like

Cette méthode utilise le VAE du pipeline T2I-ImageNet pour reconstruire une image puis calculer une carte d'erreur.

Elle est proche des approches fondées sur la reconstruction image, comme AEROBLADE ou certaines variantes inspirées de DIRE, mais elle ne reproduit pas le pipeline officiel de ces méthodes.

## Principe

```text
image x
  -> encodeur VAE
  -> latent z
  -> décodeur VAE
  -> reconstruction x_hat
  -> carte d'erreur |x - x_hat|