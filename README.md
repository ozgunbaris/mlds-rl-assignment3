# IEMS 469 Reinforcement Learning, Assignment 3: Federated Learning and Differential Privacy

FedAvg on the Federated EMNIST subset of the course (100 writers, 62 classes), written with PyTorch tensors and Ray Core only.

| Part | Task | Code | Results |
|---|---|---|---|
| 1.1 | FedAvg, client fraction C and local epochs E | `fedavg/core.py`, `fedavg/sweep.py` | `results/tune`, `results/ce`, `results/main` |
| 1.2 | Client updates as Ray actors, 4 clients per round | `fedavg/ray_clients.py` | `results/ray` |
| 1.2 | The same actors on 2 GPUs of the lab cluster | `fedavg/gpu_checks.py`, `scripts/nomad` | `results/gpu`, `results/gpu_share` |
| 2 | Laplace noise of scale b on every pixel | `fedavg/data.py` | `results/noise`, `results/noise_lr`, `results/noise_variants` |

The written report is `report/report.pdf` (source `report/report.md`; rebuild with `report/build.sh`, which needs pandoc and xelatex).

## Setup

Python 3.12 was used. All dependencies are in `requirements.txt`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The data is distributed on Canvas (Files) and is not part of this repository. Put `train_data.npy` and `test_data.npy` in a folder `data/` at the repository root, or set `FEMNIST_DIR` to their folder. Every command below is run from the repository root.

```bash
python -m pytest tests -q        # 19 mechanical checks, about 15 s (the checks on real data need the files;
                                 # one check needs a CUDA device and is skipped without it)
```

## Results

| Part | Setting | Result |
|---|---|---|
| 1.1 | C = 10%, E = 5, B = 20, learning rate 0.5, 300 rounds, 5 seeds | validation accuracy 0.804, test accuracy 0.805 (0.801 to 0.809) |
| 1.1 | client fraction C = 1%, 2%, 5%, 10% at E = 5 | validation accuracy 0.699, 0.774, 0.800, 0.804 |
| 1.1 | local epochs E = 1, 2, 5, 10, 20 at C = 10% | validation accuracy 0.800, 0.805, 0.804, 0.800, 0.798; rounds to 75%: 100, 73, 54, 49, 50 |
| 1.2 | Ray actors, 4 clients per round, 3 seeds | validation accuracy 0.794, test accuracy 0.796; weights identical to the sequential loop |
| 1.2 | GPU actors on 2 RTX 5060 Ti, seed 0 | test accuracy 0.795 against 0.793 for the CPU loop on the same machine; 1.71 s per round against 1.28 s with CPU actors |
| 2 | Laplace scale b = 0.1, 0.2, 0.3, 0.5 at learning rate 0.5, 3 seeds | test accuracy 0.774, 0.723, 0.671, 0.398 |
| 2 | b = 0.5, 1, 2, 5 with the learning rate chosen per b, seed 0 | test accuracy 0.532, 0.373, 0.161, 0.114 |

Learning rate 0.5 is unstable for b of 0.75 and above, because the squared input norm grows as 1568 b^2. The report suggests b = 1 with learning rate 0.005 and states the conditions of that choice.

## Running the experiments

```bash
scripts/run_all.sh               # every experiment, validation test and figure that needs no GPU
```

The script runs the steps below. A run whose `summary.json` exists is skipped.

The trained weights (`final_weights.pt`, 81 MB in 194 runs) are not in the repository. The option `--restore-weights` of `fedavg.sweep` retrains every run whose summary exists but whose weight file is missing, and writes only the weight file. `scripts/run_all.sh` uses it for the runs whose weights the validation and plotting scripts read (`main`, `noise`, `noise_variants`, `noise_lr`), and it repeats the Ray runs when their weight files are missing. On the machine of the report the restored weights are identical to the original ones; another machine can differ by float32 rounding (test R8).

```bash
python -m fedavg.sweep --name tune            # learning rate x batch size, 24 runs
python -m fedavg.sweep --name ce              # C in {1, 2, 5, 10}% x E in {1, 2, 5, 10, 20}, 3 seeds
python -m fedavg.sweep --name main            # chosen hyperparameters, 5 seeds
python -m fedavg.sweep --name checks          # validation runs that change the problem
python -m fedavg.sweep --name noise           # Part 2: 13 noise scales, 3 seeds
python -m fedavg.sweep --name noise_variants  # Part 2: clipped pixels, clean validation split
python -m fedavg.sweep --name noise_lr        # Part 2: learning rate re-tuned per noise scale, 26 runs
python -m fedavg.run --out results/ray/seed0 --clients-per-round 4 --E 5 --B 20 --lr 0.5 \
    --rounds 300 --seed 0 --ray               # Part 1.2, one seed
python -m fedavg.validate                     # validation experiments (list them with --list)
python -m fedavg.make_plots                   # figures in results/figures
```

GPU tests R5 to R10 (`fedavg/gpu_checks.py`):

```bash
python -m fedavg.gpu_checks --out results/gpu_share --only share        # R5, logical GPUs, any machine
python -m fedavg.gpu_checks --out results/gpu_share --only sensitivity  # R8 control, CPU only
python -m fedavg.gpu_checks --out results/gpu                           # R6 to R10, needs CUDA devices
python -m fedavg.gpu_checks --report                                    # tables and figure from the stored numbers
```

The stored GPU results come from one job on the lab's Nomad cluster (node mlds-deepdish5, 2 RTX 5060 Ti, 6 CPU cores, 28 minutes). The scripts in `scripts/nomad` reproduce it and need an account on that cluster. The upload and fetch scripts read the login node from the environment variable `NOMAD_SSH` (user@host).

```bash
scripts/nomad/make_job.sh                    # writes scripts/nomad/fedavg-gpu-checks.nomad with the code embedded
scripts/nomad/upload_data.sh <alloc-id>      # streams the course data into the running allocation
scripts/nomad/fetch_results.sh <alloc-id>    # copies the results to results/gpu
```

`results/gpu/gpu_checks.json` holds every number of the GPU tests, and `results/gpu/cluster_logs` holds the job logs.

One model with any setting. The defaults of `fedavg.run` are the chosen hyperparameters (E = 5, B = 20, learning rate 0.5, 300 rounds, seed 0), and the noise seed equals `--seed` as in the sweeps:

```bash
python -m fedavg.run --out results/try --C 0.1                             # Part 1.1, chosen hyperparameters
python -m fedavg.run --out results/try_ray --clients-per-round 4 --ray     # Part 1.2
python -m fedavg.run --out results/try_noise --C 0.1 --noise-b 0.5        # Part 2, b = 0.5
python -m fedavg.run --help
```

Each run folder holds four files.

- `history.csv` has one row per round: aggregated training and validation loss and accuracy over all 100 clients, the same for the selected clients, local losses and step sizes.
- `arrays.npz` holds the per-client accuracy per round, the selection mask and the actor timings.
- `final_weights.pt` holds the final global weights (not in the repository, see above).
- `summary.json` holds the configuration, the final metrics and the test accuracy.

## Design decisions

- **Model.** 784 inputs, 128 hidden units with ReLU, 62 outputs; 108,478 parameters kept as one flat vector.
- **Input.** The files store pixels in [0, 1] with a white page (1) and dark ink. The network receives 1 - x.
- **Split.** Each client's images are split at random into 80% training and 20% validation (`data.split_indices`).
- **Client update.** E epochs of minibatch SGD with batch size B, started from the global weights, with no state kept between rounds.
- **Server step.** The new global weights are the average of the returned client weights with weights n_k / n.
- **Reported curves.** After every round the new global model is evaluated on the training and validation splits of all 100 clients.
- **Test data.** The test images are loaded with the training files and evaluated once, after the last round of a run. No training or selection step reads them. Hyperparameters were chosen on validation accuracy.
- **Noise.** Laplace noise of scale b is added once to every pixel of every local image, on the [0, 1] scale, before training. The test images stay clean. The training and validation splits are both perturbed.
- **Ray.** One actor per selected client per round, 1 CPU each, killed after its update is collected. With G CUDA devices and m clients per round an actor also reserves 1 GPU when G >= m and 1/ceil(m/G) of one GPU otherwise.

## Tests

`tests/test_fedavg.py` holds the mechanical checks. `fedavg/validate.py` holds the experiments that test the behaviour on the real data; each writes `results/validation/<id>.md` with its numbers, and `results/validation/validation.md` joins them.

| id | `--only` name | what it does | how it validates |
|---|---|---|---|
| F1 | `fedsgd` | all 100 clients, E = 1, full batch, 20 rounds | the weights must equal 20 gradient-descent steps on the pooled data (to 1e-15 in float64); 1/m weights must not |
| F2 | `trace` | first 10 rounds, one table row each | the server step is recomputed in numpy; the first loss must be ln 62 |
| F3 | `structure` | training labels permuted inside each client | validation accuracy must stay at the majority-class rate |
| F4 | `structure` | the same images dealt to the clients at random | client updates must agree more and per-client accuracy must spread less |
| F5 | `references` | majority class, clients trained alone, centralized SGD | FedAvg must lie between the lower and the upper reference |
| F6 | `sampling` | selection counts of the 5 main runs | chi-square test against uniform sampling without replacement |
| F7-F9 | `structure` | input scaling, 1/m weights, learning-rate decay | sensitivity of the result to choices the handout leaves open |
| F10 | `errors` | confusion structure of the test errors; images duplicated between test, validation and training data | errors must concentrate on glyphs that share a shape; duplicates must not change accuracy |
| F11 | `drift` | weights of the first 8 rounds in a principal-component plane | non-standard plot; client spread must grow with E |
| R1, R2 | `ray_equal` | Ray run against the sequential loop | identical weights; 4 overlapping updates in different processes |
| R3 | `ray_faults` | crashed, slow and diverged actors, and a round where all fail | the result must equal a run that omits those clients |
| R4 | `ray_scaling` | 1, 2 and 4 CPUs, E = 5 and E = 100 | concurrency must equal the CPU count; weights never change |
| R11 | `ray_memory` | resident memory of 4 to 32 actor processes; process ids probed after every round | no process of an earlier round may exist; memory must grow with the number of actors |
| R5 | `gpu_checks`: `share` | GPU share per actor for 7 pairs of clients and GPUs | all actors of a round must be alive at once with the share 1/ceil(m/G) |
| R6 | `gpu_checks`: `device_path` | shares of 0.5, 0.25 and 1 GPU on 2 real GPUs | every update must run on CUDA, spread evenly over the devices |
| R7 | `gpu_checks`: `arithmetic` | one client update on the CPU and on each GPU | the difference must have the size of float32 rounding |
| R8 | `gpu_checks`: `full_run`, `sensitivity` | 300 rounds with GPU actors against the CPU loop; CPU loop repeated from weights one rounding step apart | the GPU run must separate from the CPU run as two CPU runs separate |
| R9 | `gpu_checks`: `timing` | seconds per round for the CPU loop, CPU actors and GPU actors | measures the cost of CUDA initialisation per actor |
| R10 | `gpu_checks`: `faults` | exiting and diverging GPU actors | the round continues and later rounds use the GPUs again |
| N1 | `noise_stats` | moments and KS test of the added noise | Laplace(0, b), independent of pixel and neighbour |
| N2 | `dp_bound` | privacy loss of 20,000 image pairs | never above \|\|x - x'\|\|_1 / b |
| N3, N5, N6 | `noise_checks` | b = 0 run, sweep table with stability counts, clipped and clean-validation variants | b = 0 must reproduce Part 1 bit for bit; a clean validation split must leave the weights identical (6 of 6 scales) |
| N7 | `noise_checks` | learning rate re-tuned for b = 0.2 to 5 | the stable learning rate must fall as 1 / b^2 |
| N4 | `reident` | nearest-neighbour matching of perturbed images | measures what each b hides from an observer |
