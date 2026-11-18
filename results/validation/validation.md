# Validation experiments

Written by `python -m fedavg.validate`.

## F1. FedSGD limit equals centralized gradient descent

Prediction: with all 100 clients, one local epoch and full-batch steps, the sample-weighted average of the client models equals one gradient-descent step on the pooled training loss. 20 rounds, learning rate 0.5.

| precision | max abs weight difference, FedAvg vs centralized | same with 1/m weights | largest weight change from the start | FedAvg training loss | centralized training loss |
|---|---|---|---|---|---|
| float64 | 1.94e-16 | 5.48e-03 | 0.257 | 3.589503 | 3.589503 |
| float32 | 6.46e-06 | 5.48e-03 | 0.257 | 3.589503 | 3.589503 |


## F2. The first 10 rounds step by step

Settings: 10 clients per round, E = 5, B = 20, learning rate 0.5, seed 0. The last column is the largest absolute difference between the server's new weights and an independent numpy recomputation of sum_k (n_k / n) w_k.

| round | selected clients | their samples | local loss, first step to last epoch | global training loss, before to after | training accuracy | validation accuracy | mean client move | server move | ratio | recomputation error |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 12 13 16 19 35 43 47 54 63 96 | 2663 | 4.125 to 1.645 | 4.122 to 2.834 | 0.356 | 0.340 | 8.52 | 5.61 | 0.66 | 4.5e-08 |
| 2 | 23 32 34 35 37 50 53 65 93 95 | 2707 | 2.929 to 1.030 | 2.834 to 2.211 | 0.459 | 0.445 | 8.12 | 4.84 | 0.60 | 4.0e-08 |
| 3 | 3 23 39 40 56 57 74 76 80 91 | 2643 | 2.171 to 0.623 | 2.211 to 1.871 | 0.526 | 0.522 | 6.75 | 3.53 | 0.52 | 6.0e-08 |
| 4 | 8 33 44 55 68 70 80 83 92 94 | 2651 | 1.645 to 0.605 | 1.871 to 1.731 | 0.549 | 0.553 | 6.66 | 3.11 | 0.47 | 5.7e-08 |
| 5 | 14 26 27 30 31 39 67 80 84 85 | 2319 | 1.734 to 0.460 | 1.731 to 1.621 | 0.581 | 0.573 | 6.00 | 2.49 | 0.42 | 6.2e-08 |
| 6 | 1 3 25 28 41 51 53 55 57 92 | 2642 | 1.581 to 0.333 | 1.621 to 1.544 | 0.599 | 0.597 | 5.46 | 2.23 | 0.41 | 8.5e-08 |
| 7 | 11 35 46 50 54 59 71 78 81 97 | 2414 | 1.880 to 0.379 | 1.544 to 1.496 | 0.608 | 0.603 | 5.84 | 2.29 | 0.39 | 6.5e-08 |
| 8 | 2 22 30 42 58 64 66 67 82 98 | 2566 | 1.621 to 0.356 | 1.496 to 1.411 | 0.625 | 0.621 | 5.68 | 2.27 | 0.40 | 8.6e-08 |
| 9 | 2 9 12 24 37 38 59 68 72 88 | 2334 | 1.760 to 0.319 | 1.411 to 1.402 | 0.622 | 0.609 | 6.27 | 2.52 | 0.40 | 6.4e-08 |
| 10 | 14 15 21 35 62 72 81 86 92 94 | 2622 | 1.397 to 0.249 | 1.402 to 1.337 | 0.643 | 0.629 | 5.27 | 2.03 | 0.39 | 6.6e-08 |

Round 1 in detail. Every client starts from the same initial weights, whose predictions are close to uniform, so the first minibatch loss must be close to ln 62 = 4.127.

| client | n_k | weight n_k / n | SGD steps | loss at first step | mean loss in last epoch | accuracy of its own model on its data | accuracy of the new global model on its data |
|---|---|---|---|---|---|---|---|
| 12 | 205 | 0.0770 | 55 | 4.117 | 2.216 | 0.454 | 0.468 |
| 13 | 293 | 0.1100 | 75 | 4.096 | 1.299 | 0.727 | 0.468 |
| 16 | 300 | 0.1127 | 75 | 4.142 | 1.383 | 0.760 | 0.377 |
| 19 | 250 | 0.0939 | 65 | 4.123 | 2.230 | 0.596 | 0.392 |
| 35 | 308 | 0.1157 | 80 | 4.153 | 1.471 | 0.714 | 0.198 |
| 43 | 245 | 0.0920 | 65 | 4.109 | 1.504 | 0.612 | 0.498 |
| 47 | 266 | 0.0999 | 70 | 4.137 | 1.817 | 0.605 | 0.353 |
| 54 | 287 | 0.1078 | 75 | 4.120 | 1.329 | 0.474 | 0.544 |
| 63 | 286 | 0.1074 | 75 | 4.118 | 1.679 | 0.458 | 0.346 |
| 96 | 223 | 0.0837 | 60 | 4.130 | 1.827 | 0.143 | 0.507 |


## F5. Reference models that bracket FedAvg

Prediction: a correct FedAvg model must score above both majority-class predictors and above clients trained alone, and at most slightly above centralized training of the same network on the pooled data.

| reference | pooled validation accuracy |
|---|---|
| most frequent class of the pooled training data | 0.0577 |
| each client's own most frequent class | 0.0662 |
| each client trained alone (best of 2 learning rates and 8 epoch counts: lr 0.5, 40 epochs) | 0.6682 |
| centralized SGD on pooled data (best of 4 learning rates: 0.1, best epoch 40 of 60) | 0.8241 |
| centralized SGD, same run after 60 epochs | 0.8143 |

FedAvg (seed 0, round 300) reaches 0.8082. Its global model beats the client's own model on 90 of 100 clients (median gain 0.147, smallest gain -0.208).

433 of the 6360 validation images belong to a class of which their client has no training image. A client trained alone classifies 0.000 of them correctly; the FedAvg model classifies 0.478 (0.832 on the other validation images).


## F6. Client sampling statistics

Prediction: every round has exactly 10 distinct clients; a client's number of selections in 300 rounds is Binomial(300, 0.1) with mean 30 and standard deviation 5.2; the expected number of clients not yet selected after 10 rounds is 100 x 0.9^10 = 34.9. The chi-square statistic has 99 degrees of freedom.

| run | 10 distinct clients in every round | expected selections | fewest | most | chi-square | p-value | clients unseen after 10 rounds | round by which all were selected |
|---|---|---|---|---|---|---|---|---|
| seed0 | True | 30 | 19 | 46 | 94.7 | 0.60 | 26 | 85 |
| seed1 | True | 30 | 17 | 44 | 108.0 | 0.25 | 31 | 48 |
| seed2 | True | 30 | 18 | 51 | 113.0 | 0.16 | 30 | 71 |
| seed3 | True | 30 | 17 | 51 | 111.9 | 0.18 | 33 | 38 |
| seed4 | True | 30 | 16 | 47 | 126.1 | 0.03 | 36 | 46 |


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


## F10. Structure of the remaining test errors and duplicated images

Prediction: handwritten letters such as c/C, o/O, s/S differ between cases mainly in size, which a 28 x 28 crop removes, and 0/O/o and 1/I/l share one shape. If the model is otherwise correct, a large share of its test errors must be such pairs, and accuracy with upper and lower case merged must be much higher.

| model | test accuracy | errors of 3,621 | same letter, other case | within {0, O, o} or {1, I, l} | accuracy with cases merged | accuracy without the 18 duplicated images | accuracy on the 18 |
|---|---|---|---|---|---|---|---|
| seed0 | 0.8078 | 696 | 86 (12%) | 98 (14%) | 0.8315 | 0.8077 | 0.833 |
| seed1 | 0.8012 | 720 | 83 (12%) | 112 (16%) | 0.8241 | 0.8013 | 0.778 |
| seed2 | 0.8092 | 691 | 90 (13%) | 93 (13%) | 0.8340 | 0.8090 | 0.833 |
| seed3 | 0.8028 | 714 | 96 (13%) | 93 (13%) | 0.8293 | 0.8024 | 0.889 |
| seed4 | 0.8053 | 705 | 96 (14%) | 101 (14%) | 0.8318 | 0.8049 | 0.889 |

The random 80/20 split does not remove duplicates. 22 of the 6360 validation images have a pixel-identical copy with the same label in the training split of the same client, and 1 more have one at another client only. A model that had memorised its training images would classify these images almost perfectly. Removing them can change validation accuracy by at most 0.0036.

| model | validation accuracy | without the duplicated images | on the duplicated images |
|---|---|---|---|
| seed0 | 0.8082 | 0.8083 | 0.783 |
| seed1 | 0.8011 | 0.8007 | 0.913 |
| seed2 | 0.8044 | 0.8046 | 0.739 |
| seed3 | 0.8036 | 0.8037 | 0.783 |
| seed4 | 0.8052 | 0.8051 | 0.826 |

Most confused pairs for seed 0 (both directions added): I/l 46, 1/l 17, 0/O 15, C/e 14, M/m 12, F/f 11, 9/q 11, 5/S 11, 1/I 10, 0/o 10, D/O 9, P/p 8. Spearman correlation between a class's number of training images and its test accuracy: 0.79 (p = 2.1e-14).


## F11. Client drift in weight space

Non-standard plot `drift_pca.png`. Each panel projects the 108,478-dimensional weights of the first 8 rounds onto their two largest principal components. Prediction: more local epochs move each client further toward its own optimum, so the client end points spread more and the server step becomes a smaller fraction of the mean client step.

| E | server step / mean client step, rounds 1-8 | variance in the plotted plane |
|---|---|---|
| 1 | 0.584 | 66% |
| 5 | 0.485 | 46% |
| 20 | 0.447 | 36% |


## R1, R2. Ray actors reproduce the sequential run and run concurrently

Run `results/ray/seed0`: 4 clients per round, 300 rounds, E = 5, B = 20, learning rate 0.5. Prediction: a client update depends only on the global weights, the client's data and a seed derived from (run seed, round, client), so the actors must return exactly what the sequential loop returns.

| quantity | value |
|---|---|
| final weights identical to the sequential run (torch.equal) | True |
| largest absolute weight difference | 0.0e+00 |
| per-round validation loss and training accuracy identical | True |
| client updates / distinct worker process ids | 1200 / 1200 |
| rounds in which all 4 actors were alive at the same time | 48 of 300 |
| rounds in which all 4 training calls overlapped in time | 43 of 300 |
| mean seconds from actor construction to its result | 0.05 |
| largest number of actors alive at one time | 4 |
| mean seconds per round with Ray | 1.56 |
| mean seconds of one client's training inside its actor | 0.041 |
| seconds per round, sequential loop | 0.083 |


## R3. Client failures during a round

Prediction: the server averages the clients that returned finite weights, with weights renormalised over them, and keeps the old model when nobody returned. The reference is a sequential run that omits the same (round, client) pairs.

| round | injected fault | updates used | seconds | validation accuracy | server step |
|---|---|---|---|---|---|
| 1 | none | 4 of 4 | 1.0 | 0.3522 | 6.60 |
| 2 | one actor process exits during training | 3 of 4 | 1.6 | 0.3695 | 5.76 |
| 3 | one actor sleeps 60 s; the round waits 15 s | 3 of 4 | 15.1 | 0.4692 | 4.74 |
| 4 | one actor returns non-finite weights | 3 of 4 | 1.2 | 0.5178 | 4.33 |
| 5 | all four actor processes exit | 0 of 4 | 1.5 | 0.5178 | unchanged |
| 6 | none | 4 of 4 | 1.5 | 0.5505 | 3.83 |

Final weights identical to the reference run: True. All weights finite: True.


## R4. CPU allocation and wall-clock time

Each actor reserves 1 CPU, so Ray may run at most as many actors at once as it has CPUs. Prediction: the largest number of simultaneously training actors equals min(4, CPUs), the weights are identical in every setting, and parallel execution pays off only when one client's training time exceeds the cost of creating its actor.

| E | CPUs given to Ray | sequential s/round | Ray s/round | training s per client | most actors training at once | most actors alive at once | weights equal to sequential |
|---|---|---|---|---|---|---|---|
| 5 | 1 | 0.10 | 3.33 | 0.02 | 1 | 1 | True |
| 5 | 2 | 0.10 | 2.05 | 0.03 | 2 | 2 | True |
| 5 | 4 | 0.10 | 1.50 | 0.04 | 4 | 4 | True |
| 100 | 1 | 1.63 | 4.90 | 0.30 | 1 | 1 | True |
| 100 | 2 | 1.63 | 3.31 | 0.42 | 2 | 2 | True |
| 100 | 4 | 1.63 | 2.34 | 0.62 | 4 | 4 | True |


## R11. Memory of actor processes and their lifetime

The handout states that an actor has a large memory overhead and asks for actors of the selected clients only. Part A runs 20 rounds with 4 actors per round and checks, with `os.kill(pid, 0)` on every actor's process id, which actor processes still exist. Prediction: no process of an earlier round exists when a round ends.

| quantity | value |
|---|---|
| client updates / distinct actor processes | 80 / 80 |
| peak resident memory of one actor process, MiB (mean, min to max) | 273 (264 to 277) |
| actor processes of earlier rounds that still exist at the end of a round (max over rounds) | 0 |
| actor processes of the same round that still exist right after ray.kill (min to max) | 1 to 3 |
| seconds until all processes of a round are gone after ray.kill (mean, max) | 0.012, 0.018 |
| actor processes that exist 2 s after the last round | 0 |

Part B keeps k actors alive at once, each after one training call, and reads their resident memory with `ps`. Prediction: the total grows in proportion to k, so 100 persistent actors would need 25 times the memory of 4.

| actors alive at once | processes measured | total resident memory (MiB) | mean per actor (MiB) | seconds to create and train once |
|---|---|---|---|---|
| 4 | 4 | 1065 | 266 | 1.5 |
| 8 | 8 | 2138 | 267 | 2.2 |
| 16 | 16 | 4072 | 255 | 4.6 |
| 32 | 32 | 7264 | 227 | 10.4 |

Estimate for 100 actors alive at once, from the smallest and the largest mean per actor: 22.2 to 26.1 GiB, against 1.04 GiB measured for 4; the machine has 32 GiB of memory.
Resident memory counts pages of shared libraries in every process, so the sum overstates the unique memory.


## N1. Distribution of the injected noise

Measured on the noise actually added to the 19,964,560 training pixels (float32 difference of the perturbed and the clean tensors). Laplace(0, b) has mean 0, variance 2b^2, mean absolute value b and P(|e| > b) = 1/e = 0.3679. The Kolmogorov-Smirnov test uses 200,000 sampled pixels.

| b | mean | variance / 2b^2 | mean |e| / b | P(|e| > b) | KS distance to Laplace(0, b) | correlation of neighbouring pixels' noise | correlation of noise with the pixel value |
|---|---|---|---|---|---|---|---|
| 0.1 | -0.00003 | 1.0001 | 1.0000 | 0.3679 | 0.0021 (p = 0.34) | -0.0004 | +0.0000 |
| 1.0 | +0.00020 | 1.0002 | 1.0000 | 0.3678 | 0.0015 (p = 0.77) | -0.0006 | +0.0008 |
| 5.0 | +0.00061 | 1.0006 | 1.0004 | 0.3681 | 0.0013 (p = 0.88) | +0.0004 | -0.0001 |


## N2. The differential-privacy inequality of the Laplace mechanism

For a released image o = x + e the log-likelihood ratio between two candidate originals x and x' is (||o - x'||_1 - ||o - x||_1) / b. The triangle inequality bounds it by ||x - x'||_1 / b, which is the epsilon of the Laplace mechanism for that pair; the worst case over all images in [0, 1]^784 is 784 / b. Checked on 20,000 random pairs of training images: their L1 distance is 37.6 on average and at most 117.1.

| b | epsilon per pixel (1/b) | epsilon per image, worst case (784/b) | epsilon for an average pair | mean measured privacy loss | largest loss / bound | pairs above the bound | share of pairs where the released image favours the wrong original |
|---|---|---|---|---|---|---|---|
| 0.1 | 10.00 | 7840 | 375.8 | 304.82 | 0.936 | 0 | 0.0000 |
| 0.5 | 2.00 | 1568 | 75.2 | 32.90 | 0.825 | 0 | 0.0011 |
| 1.0 | 1.00 | 784 | 37.6 | 10.15 | 0.764 | 0 | 0.0225 |
| 2.0 | 0.50 | 392 | 18.8 | 2.82 | 0.721 | 0 | 0.1331 |
| 5.0 | 0.20 | 157 | 7.5 | 0.49 | 0.715 | 0 | 0.3136 |
| 20.0 | 0.05 | 39 | 1.9 | 0.03 | 0.671 | 0 | 0.4532 |
| 100.0 | 0.01 | 8 | 0.4 | 0.00 | 0.721 | 0 | 0.4840 |

The last column is the error rate of the best possible guess between two equally likely candidate images.


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


## GPU checks of the Ray actors (R5 to R10)

Machine: one Nomad allocation on mlds-deepdish5 with 6 CPU cores and 2 GPUs (NVIDIA GeForce RTX 5060 Ti, 15.6 GiB each). Software: Python 3.12.16, PyTorch 2.14.1+cu130, Ray 2.60.0. Ray counted 2 GPUs without being told.

### R5. GPU share per actor

Ray places a fractional GPU request on one device. Prediction: a share of G/m lets all m actors run at once only when m/G is an integer, whereas the share 1/ceil(m/G) always does. Probe actors that sleep 1.5 s were run with logical GPUs on the laptop (no CUDA call is made).

| actors m | GPUs G | share G/m | actors alive at once | seconds | share 1/ceil(m/G) | actors alive at once | seconds |
|---|---|---|---|---|---|---|---|
| 3 | 2 | 0.6666 | 2 | 3.8 | 0.5000 | 3 | 1.9 |
| 4 | 1 | 0.2500 | 4 | 2.0 | 0.2500 | 4 | 2.1 |
| 4 | 2 | 0.5000 | 4 | 2.0 | 0.5000 | 4 | 2.0 |
| 4 | 3 | 0.7500 | 3 | 3.8 | 0.5000 | 4 | 2.0 |
| 4 | 4 | 1.0000 | 4 | 2.1 | 1.0000 | 4 | 2.1 |
| 4 | 5 | 1.0000 | 4 | 2.0 | 1.0000 | 4 | 2.1 |
| 10 | 4 | 0.4000 | 8 | 4.1 | 0.3333 | 10 | 2.4 |

On the 2 real GPUs with 3 clients per round (30 rounds of training): share 0.5 gives 3 actors alive at once in 30 of 30 rounds and 1.57 s per round; share 0.6666 gives at most 2 actors alive at once and 2.75 s per round. Both runs end with the same validation accuracy (0.6893 and 0.6893).

### R6. The actors train on the GPUs that Ray assigned

Prediction: every update runs on a CUDA device, the updates are spread evenly over the devices, all actors of a round are alive at once, and every update has its own process.

| setting | GPU share per actor | updates on CUDA | updates per device | devices used per round (min) | rounds with all actors alive at once | distinct processes | largest memory allocated by PyTorch in one actor (MiB) | largest device memory in use (MiB) |
|---|---|---|---|---|---|---|---|---|
| 4 clients, 2 GPUs | 0.5 | 120 of 120 | GPU 0: 60, GPU 1: 60 | 2 | 30 of 30 | 120 | 67 | 635 |
| 4 clients, 1 GPU | 0.25 | 120 of 120 | GPU 0: 120 | 1 | 30 of 30 | 120 | 67 | 1134 |
| 2 clients, 2 GPUs | 1.0 | 60 of 60 | GPU 0: 30, GPU 1: 30 | 2 | 30 of 30 | 60 | 67 | 386 |
| 3 clients, 2 GPUs | 0.5 | 90 of 90 | GPU 0: 60, GPU 1: 30 | 2 | 30 of 30 | 90 | 67 | 635 |
| 4 clients, GPUs not requested | 0.0 | 0 of 120 | none | 0 | 30 of 30 | 120 | 0 | 0 |

The device memory in use before the job was 137 MiB per GPU. The device figure includes the CUDA context of every process on that GPU.

### R7. One client update on the CPU and on the GPUs

Prediction: the batch order is drawn on the CPU from the update seed, so a GPU update differs from the CPU update only by float32 rounding; repeating it on the same device model gives identical bits.

| client | training images | update | SGD steps | largest weight difference GPU vs CPU | difference relative to the distance moved | repeat on the same GPU identical | GPU 0 and GPU 1 identical |
|---|---|---|---|---|---|---|---|
| 0 | 275 | 1 full-batch step | 1 | 7.5e-09 | 4.9e-07 | True | True |
| 0 | 275 | E = 5, B = 20 | 70 | 1.8e-07 | 2.5e-07 | True | True |
| 67 | 314 | 1 full-batch step | 1 | 7.5e-09 | 4.5e-07 | True | True |
| 67 | 314 | E = 5, B = 20 | 80 | 3.0e-07 | 2.8e-07 | True | True |

Largest difference of the logits of 1,000 test images between CPU and GPU: 6.0e-08.

### R8. A full run with GPU actors against the CPU loop

Setting: 4 clients per round, 300 rounds, E = 5, B = 20, learning rate 0.5, seed 0, both on the cluster machine. Prediction: the two runs start identical, differ by rounding after round 1, and the difference then grows because every later update starts from slightly different weights. The control repeats the CPU loop from initial weights that differ by one float32 rounding step (run on the same cluster machine).

| largest absolute weight difference after round | 1 | 2 | 5 | 10 | 30 | 100 | 200 | 300 |
|---|---|---|---|---|---|---|---|---|
| GPU actors vs CPU loop, learning rate 0.5 | 1.2e-07 | 1.8e-07 | 3.3e-03 | 5.3e-02 | 1.3e-01 | 2.2e-01 | 3.5e-01 | 4.6e-01 |
| CPU loop twice, all initial weights moved one rounding step, learning rate 0.5 | 8.9e-08 | 2.1e-07 | 7.2e-04 | 5.1e-02 | 8.0e-02 | 1.8e-01 | 3.6e-01 | 5.1e-01 |
| CPU loop twice, one initial bias moved one rounding step, learning rate 0.5 | 1.2e-07 | 1.8e-07 | 7.2e-04 | 5.1e-02 | 2.1e-01 | 2.0e-01 | 3.7e-01 | 4.5e-01 |
| CPU loop twice, all initial weights moved one rounding step, learning rate 0.05 | 6.0e-08 | 6.0e-08 | 6.7e-08 | 1.2e-07 | 2.6e-04 | 3.3e-03 | 7.1e-03 | 1.4e-02 |


| run | final training accuracy | final validation accuracy | test accuracy | test predictions that differ from the CPU loop | validation accuracy gap over the rounds (largest / mean) |
|---|---|---|---|---|---|
| CPU loop | 0.8697 | 0.7972 | 0.7926 |  |  |
| GPU actors | 0.8702 | 0.7975 | 0.7945 | 172 of 3621 | 0.0160 / 0.0019 |
| CPU loop, initial weights one rounding step away |  | 0.7942 | 0.7940 | 192 of 3621 | 0.0175 / 0.0019 |


CPU actors on the cluster machine are identical to the CPU loop after 30 rounds: True (largest difference 0). Two GPU runs with the same seed are identical: True (largest difference 0); 100% of their updates ran on the same device index in both runs. The full GPU run used 1200 updates in 1200 processes, all on CUDA: True.


The same run on two CPUs (laptop, Apple M5, against the cluster machine, x86-64): largest weight difference after 300 rounds 3.4e-01, test accuracy 0.7940 against 0.7926, per-round validation accuracy identical in the first 16 rounds.

### R9. Wall-clock time with CPU actors and GPU actors

Prediction: this network has 108,478 weights and a batch has 20 images, so a GPU step is dominated by the fixed cost of launching its operations and a GPU actor is not faster than a CPU actor; each new actor process must also initialise CUDA.

| E | runner | seconds per round | seconds in the actor constructor | seconds of training per client |
|---|---|---|---|---|
| 5 | sequential CPU loop | 0.08 |  | 0.020 |
| 5 | Ray, CPU actors | 1.28 | 0.001 | 0.192 |
| 5 | Ray, GPU actors on 2 GPUs | 1.71 | 0.333 | 0.335 |
| 5 | Ray, GPU actors on 1 GPU | 1.76 | 0.348 | 0.341 |
| 100 | sequential CPU loop | 1.58 |  | 0.397 |
| 100 | Ray, CPU actors | 1.73 | 0.001 | 0.611 |
| 100 | Ray, GPU actors on 2 GPUs | 2.31 | 0.333 | 0.917 |
| 100 | Ray, GPU actors on 1 GPU | 2.55 | 0.345 | 1.126 |


The same update (client 67, 314 images, E = 5) in one process after a warm-up call:

| batch size | SGD steps | CPU seconds (1 thread) | GPU seconds | CPU time / GPU time |
|---|---|---|---|---|
| B = 20 | 80 | 0.023 | 0.024 | 0.97 |
| B = 200 | 10 | 0.008 | 0.003 | 2.40 |
| full batch | 5 | 0.007 | 0.002 | 4.24 |

### R10. Failing GPU actors

Round 2: one actor process exits during training. Round 4: one actor returns non-finite weights. Round 5: all four exit. Updates used in rounds 1 to 6: [4, 3, 4, 3, 0, 4]. Devices of the round-6 updates: {'cuda': 4}. Largest difference to a CPU reference run that omits the same clients: 1.1e-02. All weights finite: True. Largest device memory in use per round (MiB): 1: 771, 2: 751, 3: 771, 4: 771, 5: no update, 6: 771.

