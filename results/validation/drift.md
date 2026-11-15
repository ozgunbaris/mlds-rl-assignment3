## F11. Client drift in weight space

Non-standard plot `drift_pca.png`. Each panel projects the 108,478-dimensional weights of the first 8 rounds onto their two largest principal components. Prediction: more local epochs move each client further toward its own optimum, so the client end points spread more and the server step becomes a smaller fraction of the mean client step.

| E | server step / mean client step, rounds 1-8 | variance in the plotted plane |
|---|---|---|
| 1 | 0.584 | 66% |
| 5 | 0.485 | 46% |
| 20 | 0.447 | 36% |

