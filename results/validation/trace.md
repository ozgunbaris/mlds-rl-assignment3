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

