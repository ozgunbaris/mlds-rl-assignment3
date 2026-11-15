## F3, F4, F7, F8, F9. Runs that change the structure of the problem

**F4. Writers against a random deal of the same images.** The 'IID' federation deals the same 31,825 images to 100 clients at random with the original client sizes. Prediction: with identically distributed clients the local updates point in more similar directions, so the server step is a larger fraction of the mean client step, and the clients' validation accuracies differ less. If 10 equally weighted client steps were orthogonal the ratio would be 1/sqrt(10) = 0.32; if they were identical it would be 1.

| E | partition | validation accuracy, round 300 | test accuracy | server step / mean client step, rounds 1-50 | std of per-client validation accuracy | rounds to 70% validation accuracy (2 seeds) |
|---|---|---|---|---|---|---|
| 1 | real writers | 0.8030 | 0.7987 | 0.412 | 0.101 | 58 and 56 |
| 1 | random (IID) deal | 0.8263 | 0.8159 | 0.446 | 0.049 | 42 and 41 |
| 5 | real writers | 0.8046 | 0.8045 | 0.371 | 0.099 | 24 and 22 |
| 5 | random (IID) deal | 0.8282 | 0.8176 | 0.381 | 0.045 | 13 and 12 |
| 20 | real writers | 0.7994 | 0.7965 | 0.361 | 0.101 | 20 and 18 |
| 20 | random (IID) deal | 0.8226 | 0.8146 | 0.366 | 0.048 | 8 and 8 |

**F3. Shuffled labels.** The training labels are permuted inside each client; validation labels are untouched. No image carries information about its label, so validation accuracy must stay near the majority-class rate (5.8%). Measured: highest validation accuracy over 300 rounds 0.0840, final 0.0429, test 0.0447; training accuracy on the shuffled labels 0.1288; training loss 3.653 (ln 62 = 4.127).

**F7. Input scaling.** 'invert' feeds 1 - x (ink 1, page 0), 'raw' feeds x (page 1), 'center' feeds 1 - 2x (page -1, ink 1). The three inputs are affine maps of each other, so the network can represent the same functions; any difference comes from the optimisation.

| input | learning rate | validation accuracy, round 50 | round 300 | training accuracy | test accuracy |
|---|---|---|---|---|---|
| invert | 0.05 | 0.6299 | 0.7571 | 0.7925 | 0.7622 |
| invert | 0.5 | 0.7373 | 0.8082 | 0.8879 | 0.8078 |
| raw | 0.05 | 0.4571 | 0.6255 | 0.6353 | 0.6374 |
| raw | 0.5 | 0.0516 | 0.0577 | 0.0577 | 0.0610 |
| center | 0.05 | 0.6060 | 0.7392 | 0.7687 | 0.7399 |
| center | 0.5 | 0.0516 | 0.0577 | 0.0577 | 0.0610 |

**F8. Weights n_k / n against 1 / m on the real data.** Client sizes range from 94 to 314 training images, so the two rules differ little here (test F1 shows a case where only n_k / n is correct). Validation accuracy at round 300 over seeds 0-2: n_k / n 0.8046 (0.8082, 0.8011, 0.8044); 1 / m 0.8071 (0.8132, 0.8039, 0.8041).

**F9. Learning-rate decay per round (seed 0).**

| decay factor per round | validation accuracy | validation loss | training accuracy | test accuracy |
|---|---|---|---|---|
| 1 (constant) | 0.8082 | 0.8516 | 0.8879 | 0.8078 |
| 0.995 | 0.8050 | 0.7936 | 0.8800 | 0.8036 |
| 0.99 | 0.8039 | 0.7433 | 0.8729 | 0.8034 |

