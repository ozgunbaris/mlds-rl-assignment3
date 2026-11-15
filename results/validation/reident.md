## N4. Re-identification of perturbed images

An observer holds 2,000 candidate clean images and one perturbed image of one of them, and names the candidate with the smallest L1 distance, which is the maximum-likelihood guess under Laplace noise. Chance is 0.0005. 'Label read' is the share of guesses with the correct class. The last column feeds perturbed test images to the model trained on clean data (seed 0). The energy ratio is the mean squared ink of an image divided by the noise energy 784 x 2b^2.

| b | epsilon per image, worst case | ink energy / noise energy | original identified among 2,000 | label read from the nearest candidate | clean-trained model on perturbed test images |
|---|---|---|---|---|---|
| 0.02 | 39200 | 32.499 | 1.000 | 1.000 | 0.8067 |
| 0.05 | 15680 | 5.200 | 1.000 | 1.000 | 0.7932 |
| 0.1 | 7840 | 1.300 | 1.000 | 1.000 | 0.7332 |
| 0.2 | 3920 | 0.325 | 0.980 | 0.992 | 0.5253 |
| 0.3 | 2613 | 0.144 | 0.932 | 0.968 | 0.3568 |
| 0.5 | 1568 | 0.052 | 0.768 | 0.858 | 0.1685 |
| 0.75 | 1045 | 0.023 | 0.480 | 0.600 | 0.0961 |
| 1.0 | 784 | 0.013 | 0.300 | 0.408 | 0.0704 |
| 1.5 | 523 | 0.006 | 0.100 | 0.176 | 0.0403 |
| 2.0 | 392 | 0.003 | 0.058 | 0.112 | 0.0359 |
| 3.0 | 261 | 0.001 | 0.022 | 0.068 | 0.0329 |
| 5.0 | 157 | 0.001 | 0.000 | 0.028 | 0.0304 |
| 10.0 | 78 | 0.000 | 0.002 | 0.040 | 0.0229 |
| 20.0 | 39 | 0.000 | 0.006 | 0.018 | 0.0204 |
| 50.0 | 16 | 0.000 | 0.000 | 0.016 | 0.0249 |

