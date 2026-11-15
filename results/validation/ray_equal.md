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

