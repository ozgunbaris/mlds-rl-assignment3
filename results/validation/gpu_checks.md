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
