## N3, N5, N6, N7. Noise sweep checks

**N3.** The run with b = 0 (`results/noise/b0.0_seed0`) and the Part 1 run (`results/main/seed0`) have identical final weights: True.

**N5.** Round-300 accuracies at learning rate 0.5, mean of 3 seeds. 'Training (perturbed)' is measured on the images the clients trained on; 'training (clean)' on the same images before the noise. The input energy is the mean squared norm of one network input, 20.8 + 1568 b^2. A seed counts as unstable when its aggregated training loss exceeded 20 or left the finite range in some round. 'Rounds without update' counts rounds in which every selected client returned non-finite weights, so the server kept the old model.

| b | input energy | training (perturbed) | training (clean) | validation (perturbed) | test (clean) | std of test over seeds | unstable seeds | rounds without update, per seed | distinct classes predicted on the test set, per seed |
|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 21 | 0.8862 | 0.8862 | 0.8046 | 0.8060 | 0.0043 | 0 of 3 | 0/0/0 | 61/62/62 |
| 0.02 | 21 | 0.9013 | 0.8818 | 0.8000 | 0.8032 | 0.0044 | 0 of 3 | 0/0/0 | 62/62/62 |
| 0.05 | 25 | 0.9398 | 0.8598 | 0.7690 | 0.7969 | 0.0018 | 0 of 3 | 0/0/0 | 62/62/62 |
| 0.1 | 36 | 0.9751 | 0.8118 | 0.6913 | 0.7735 | 0.0017 | 0 of 3 | 0/0/0 | 61/62/62 |
| 0.2 | 83 | 0.9932 | 0.7398 | 0.5194 | 0.7227 | 0.0029 | 0 of 3 | 0/0/0 | 58/58/59 |
| 0.3 | 162 | 0.9976 | 0.6818 | 0.3750 | 0.6711 | 0.0072 | 0 of 3 | 0/0/0 | 53/57/58 |
| 0.5 | 413 | 0.9976 | 0.4051 | 0.1757 | 0.3984 | 0.0239 | 0 of 3 | 0/0/0 | 50/52/49 |
| 0.75 | 903 | 0.5332 | 0.0294 | 0.0699 | 0.0318 | 0.0000 | 2 of 3 | 0/0/0 | 4/3/3 |
| 1.0 | 1589 | 0.1875 | 0.0577 | 0.0561 | 0.0610 | 0.0000 | 3 of 3 | 0/0/0 | 1/1/1 |
| 1.5 | 3549 | 0.0268 | 0.0184 | 0.0237 | 0.0190 | 0.0120 | 3 of 3 | 292/292/293 | 13/19/14 |
| 2.0 | 6293 | 0.0323 | 0.0223 | 0.0290 | 0.0207 | 0.0117 | 3 of 3 | 297/297/297 | 8/19/14 |
| 3.0 | 14133 | 0.0214 | 0.0274 | 0.0224 | 0.0264 | 0.0058 | 3 of 3 | 299/299/298 | 15/10/16 |
| 5.0 | 39221 | 0.0269 | 0.0173 | 0.0271 | 0.0205 | 0.0116 | 3 of 3 | 299/299/299 | 7/6/8 |

**N6. Edge-case variants (seed 0).** 'Clipped' clips perturbed pixels to [0, 1] before training. 'Clean validation' perturbs the training split only.

| b | train | validation | test | clipped: train | clipped: validation | clipped: test | clean validation: train | clean validation: validation | clean validation: test |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.9759 | 0.6937 | 0.7735 | 0.9387 | 0.7409 | 0.7307 | 0.9759 | 0.7748 | 0.7735 |
| 0.3 | 0.9980 | 0.3807 | 0.6752 | 0.9214 | 0.4726 | 0.5151 | 0.9980 | 0.6654 | 0.6752 |
| 0.5 | 0.9967 | 0.1774 | 0.3806 | 0.1749 | 0.1502 | 0.2486 | 0.9967 | 0.3671 | 0.3806 |
| 1.0 | 0.1943 | 0.0560 | 0.0610 | 0.0577 | 0.0577 | 0.0610 | 0.1943 | 0.0577 | 0.0610 |
| 2.0 | 0.0419 | 0.0401 | 0.0075 | 0.0577 | 0.0577 | 0.0610 | 0.0419 | 0.0094 | 0.0075 |
| 5.0 | 0.0419 | 0.0407 | 0.0323 | 0.0577 | 0.0577 | 0.0610 | 0.0419 | 0.0285 | 0.0323 |

Final weights of the clean-validation run identical to the standard run (torch.equal): 6 of 6 scales.

**N7. Learning rate re-tuned per noise scale (seed 0, 300 rounds).** 'Chosen' marks the E = 5 run with the highest round-300 validation accuracy among the runs that did not diverge; the test column was not used for the choice. The column 'best round' gives the round with the highest validation accuracy and the clean training accuracy in that round, which is what early stopping on the perturbed validation split would return.

| b | learning rate | E | learning rate x b^2 | training (perturbed) | validation (perturbed) | training (clean) | test (clean) | best round / clean training accuracy there | state | choice |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 0.1 | 1 | 0.0040 | 0.7493 | 0.5346 | 0.6946 | 0.6885 | 280 / 0.6827 |  |  |
| 0.2 | 0.5 | 1 | 0.0200 | 0.9848 | 0.5406 | 0.7557 | 0.7363 | 108 / 0.7155 |  |  |
| 0.2 | 0.001 | 5 | 0.0000 | 0.2923 | 0.2709 | 0.3153 | 0.3278 | 300 / 0.3153 |  |  |
| 0.2 | 0.005 | 5 | 0.0002 | 0.5574 | 0.4965 | 0.5896 | 0.5929 | 282 / 0.5844 |  |  |
| 0.2 | 0.02 | 5 | 0.0008 | 0.7109 | 0.5017 | 0.6697 | 0.6656 | 127 / 0.6212 |  |  |
| 0.2 | 0.1 | 5 | 0.0040 | 0.8598 | 0.4881 | 0.6964 | 0.6816 | 176 / 0.6724 |  |  |
| 0.2 | 0.5 | 5 | 0.0200 | 0.9933 | 0.5248 | 0.7404 | 0.7260 | 55 / 0.6793 |  | chosen |
| 0.5 | 0.1 | 1 | 0.0250 | 0.8066 | 0.2204 | 0.5374 | 0.5311 | 73 / 0.5246 |  |  |
| 0.5 | 0.5 | 1 | 0.1250 | 0.9984 | 0.1846 | 0.4428 | 0.4369 | 23 / 0.5063 |  |  |
| 0.5 | 0.001 | 5 | 0.0003 | 0.2128 | 0.1673 | 0.2996 | 0.3099 | 298 / 0.2995 |  |  |
| 0.5 | 0.005 | 5 | 0.0013 | 0.4586 | 0.2550 | 0.5336 | 0.5319 | 235 / 0.5199 |  | chosen |
| 0.5 | 0.02 | 5 | 0.0050 | 0.6101 | 0.2171 | 0.5359 | 0.5313 | 53 / 0.5090 |  |  |
| 0.5 | 0.1 | 5 | 0.0250 | 0.8851 | 0.1976 | 0.5665 | 0.5653 | 22 / 0.5070 |  |  |
| 0.5 | 0.5 | 5 | 0.1250 | 0.9967 | 0.1774 | 0.3821 | 0.3806 | 26 / 0.4664 |  |  |
| 1.0 | 0.1 | 1 | 0.1000 | 0.8785 | 0.0747 | 0.2482 | 0.2469 | 41 / 0.3615 |  |  |
| 1.0 | 0.5 | 1 | 0.5000 | 0.6880 | 0.0590 | 0.0294 | 0.0318 | 15 / 0.0601 | one class on clean images |  |
| 1.0 | 0.001 | 5 | 0.0010 | 0.1737 | 0.0942 | 0.2429 | 0.2477 | 300 / 0.2429 |  |  |
| 1.0 | 0.005 | 5 | 0.0050 | 0.4995 | 0.0987 | 0.3733 | 0.3725 | 113 / 0.3772 |  | chosen |
| 1.0 | 0.02 | 5 | 0.0200 | 0.6778 | 0.0829 | 0.3689 | 0.3645 | 106 / 0.3873 |  |  |
| 1.0 | 0.1 | 5 | 0.1000 | 0.9054 | 0.0719 | 0.3144 | 0.3046 | 28 / 0.3780 |  |  |
| 1.0 | 0.5 | 5 | 0.5000 | 0.1943 | 0.0560 | 0.0577 | 0.0610 | 81 / 0.0531 | diverged |  |
| 2.0 | 0.001 | 5 | 0.0040 | 0.2388 | 0.0527 | 0.1665 | 0.1610 | 212 / 0.1279 |  | chosen |
| 2.0 | 0.005 | 5 | 0.0200 | 0.5909 | 0.0454 | 0.2478 | 0.2499 | 103 / 0.2340 |  |  |
| 2.0 | 0.02 | 5 | 0.0800 | 0.7583 | 0.0425 | 0.2536 | 0.2538 | 52 / 0.2462 |  |  |
| 2.0 | 0.1 | 5 | 0.4000 | 0.7038 | 0.0351 | 0.0294 | 0.0318 | 22 / 0.1265 | one class on clean images |  |
| 2.0 | 0.5 | 5 | 2.0000 | 0.0419 | 0.0401 | 0.0101 | 0.0075 | 3 / 0.0101 | diverged |  |
| 5.0 | 0.001 | 5 | 0.0250 | 0.4487 | 0.0349 | 0.0948 | 0.0939 | 58 / 0.0560 |  |  |
| 5.0 | 0.005 | 5 | 0.1250 | 0.6734 | 0.0351 | 0.1136 | 0.1135 | 191 / 0.1104 |  | chosen |
| 5.0 | 0.02 | 5 | 0.5000 | 0.7269 | 0.0289 | 0.0565 | 0.0605 | 44 / 0.0658 |  |  |
| 5.0 | 0.1 | 5 | 2.5000 | 0.0407 | 0.0346 | 0.0195 | 0.0215 | 8 / 0.0573 | diverged |  |
| 5.0 | 0.5 | 5 | 12.5000 | 0.0419 | 0.0407 | 0.0291 | 0.0323 | 5 / 0.0291 | diverged |  |

