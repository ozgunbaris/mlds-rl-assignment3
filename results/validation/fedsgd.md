## F1. FedSGD limit equals centralized gradient descent

Prediction: with all 100 clients, one local epoch and full-batch steps, the sample-weighted average of the client models equals one gradient-descent step on the pooled training loss. 20 rounds, learning rate 0.5.

| precision | max abs weight difference, FedAvg vs centralized | same with 1/m weights | largest weight change from the start | FedAvg training loss | centralized training loss |
|---|---|---|---|---|---|
| float64 | 1.94e-16 | 5.48e-03 | 0.257 | 3.589503 | 3.589503 |
| float32 | 6.46e-06 | 5.48e-03 | 0.257 | 3.589503 | 3.589503 |

