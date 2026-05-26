# DRCT-like


## Main idea

For an input image \(x\), we compute a reconstructed image:

\[
\hat{x} = R(x)
\]

where \(R\) is the reconstruction process based on our lightweight pretrained model.




## Loss function

The total loss is:

\[
\mathcal{L}
=
\mathcal{L}_{cls}
+
\lambda \mathcal{L}_{contr}
\]

where:

```text
L_cls    = classification loss
L_contr  = contrastive loss
lambda   = weight of the contrastive loss
```

The classification loss separates real and fake images.

The contrastive loss encourages embeddings from the same class to be close and embeddings from different classes to be far apart.

## Difference from official DRCT

This is a simplified DRCT-like version.

Main differences:

```text
1. We use T2I-ImageNet as a lightweight reconstruction model.
2. We use a simplified reconstruction pipeline.
3. We use a margin-based contrastive loss.
4. The goal is low-cost reverse engineering, not exact reproduction.
```



