## N1. Distribution of the injected noise

Measured on the noise actually added to the 19,964,560 training pixels (float32 difference of the perturbed and the clean tensors). Laplace(0, b) has mean 0, variance 2b^2, mean absolute value b and P(|e| > b) = 1/e = 0.3679. The Kolmogorov-Smirnov test uses 200,000 sampled pixels.

| b | mean | variance / 2b^2 | mean |e| / b | P(|e| > b) | KS distance to Laplace(0, b) | correlation of neighbouring pixels' noise | correlation of noise with the pixel value |
|---|---|---|---|---|---|---|---|
| 0.1 | -0.00003 | 1.0001 | 1.0000 | 0.3679 | 0.0021 (p = 0.34) | -0.0004 | +0.0000 |
| 1.0 | +0.00020 | 1.0002 | 1.0000 | 0.3678 | 0.0015 (p = 0.77) | -0.0006 | +0.0008 |
| 5.0 | +0.00061 | 1.0006 | 1.0004 | 0.3681 | 0.0013 (p = 0.88) | +0.0004 | -0.0001 |

