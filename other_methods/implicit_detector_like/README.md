# Implicit Detector-like

This folder contains a lightweight adaptation inspired by the paper *Your Diffusion Model is an Implicit Synthetic Image Detector*.

The goal is not to reproduce the official implementation.  
Instead, we keep the central idea: a pretrained generative model can reveal useful forensic information through its internal responses to noisy inputs.

## Principle

Given an input image \(x\), we first encode it into a latent representation:

```math
z = E(x)
```

Then, for several noise levels \(\gamma\), we construct noisy latent states:

```math
z_{\gamma}
=
\sqrt{\gamma}z
+
\sqrt{1-\gamma}\varepsilon
```

The model is queried on each noisy latent:

```math
\hat{\varepsilon}_{\theta}
=
f_{\theta}(z_{\gamma},\gamma)
```

The extracted signal is built from the model responses:

```math
F(x)
=
\operatorname{Concat}
\left(
\varepsilon,
\hat{\varepsilon}_{\theta},
|\varepsilon-\hat{\varepsilon}_{\theta}|
\right)
```

## Pipeline

```text
input image x
    -> VAE encoder
    -> latent z
    -> add noise at several gamma levels
    -> query the pretrained model
    -> collect model responses
    -> build implicit detector-like features
```

## Difference from the official method

The official method samples and collects responses from a pretrained diffusion model, then uses a classifier to predict whether an image is real or synthetic.

This implementation follows the same intuition, but uses the internal response of the T2I-ImageNet model. Therefore, it should be understood as a lightweight implicit-detector-inspired approximation, not an official reproduction.