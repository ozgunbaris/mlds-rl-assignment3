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

