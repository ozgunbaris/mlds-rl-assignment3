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

