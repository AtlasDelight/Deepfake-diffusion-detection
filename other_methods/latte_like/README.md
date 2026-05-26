# LATTE-like

This folder contains a lightweight implementation inspired by **LATTE: Latent Trajectory Embedding for Diffusion-Generated Image Detection**.

The goal is not to reproduce the official LATTE method exactly. Instead, this implementation adapts its main idea to our project: using a lightweight pretrained generative model to extract a latent trajectory, then using this trajectory together with visual features for real/fake classification.

## General idea

Most reconstruction-based detectors use a single reconstruction error or a single model response. LATTE follows a richer idea: instead of relying on one step only, it extracts information from several denoising or noise levels.

In our implementation, an input image \(x\) is encoded into a latent representation:

```math
z_0 = E(x)
```

Then, for several noise levels \(\gamma_1, \gamma_2, \dots, \gamma_K\), we build noisy latent states:

```math
z_{\gamma_k}
=
\sqrt{\gamma_k}z_0
+
\sqrt{1-\gamma_k}\varepsilon
```

For each noisy latent state, the pretrained model predicts a noise-like response:

```math
\hat{\varepsilon}_{\theta}^{(\gamma_k)}
=
f_{\theta}(z_{\gamma_k},\gamma_k)
```

From this response, we estimate a single-step reconstructed latent:

```math
\hat{z}_0^{(\gamma_k)}
=
\frac{
z_{\gamma_k}
-
\sqrt{1-\gamma_k}
\hat{\varepsilon}_{\theta}^{(\gamma_k)}
}{
\sqrt{\gamma_k}
}
```

The LATTE-like trajectory is then stored as a sequence:

```math
\mathcal{T}(x)
=
\left[
\hat{z}_0^{(\gamma_1)},
\hat{z}_0^{(\gamma_2)},
\dots,
\hat{z}_0^{(\gamma_K)}
\right]
```

The important point is that we keep the trajectory dimension:

```text
[K, C, H, W]
```

instead of flattening everything into:

```text
[K * C, H, W]
```

This makes the representation closer to the LATTE idea, because the classifier can model the evolution of the latent states across several noise levels.

---



## Extraction pipeline

The extraction script is:

```text
scripts/extract_latte_like_sequences_from_folder.py
```

It takes an image folder as input and saves one `.pt` file per image.

Expected input structure:

```text
src/data/images/train/
├── real/
│   ├── real_001.jpg
│   └── real_002.jpg
└── fake/
    ├── fake_001.jpg
    └── fake_002.jpg
```

The output structure will be:

```text
error_maps/latte_like/train/
├── real/
│   ├── real_001.pt
│   └── real_002.pt
└── fake/
    ├── fake_001.pt
    └── fake_002.pt

```
## Classifier architecture

The LATTE-like classifier follows this structure:

```text
image x
    -> ResNet backbone
    -> visual feature map
    -> visual tokens

LATTE-like sequence [K, C, H, W]
    -> latent trajectory tokenizer
    -> latent tokens

latent tokens + visual tokens
    -> cross-attention fusion
    -> global aggregation
    -> binary classifier
```