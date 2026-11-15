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

