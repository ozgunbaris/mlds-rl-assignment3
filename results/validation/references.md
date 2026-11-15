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

