## F10. Structure of the remaining test errors and duplicated images

Prediction: handwritten letters such as c/C, o/O, s/S differ between cases mainly in size, which a 28 x 28 crop removes, and 0/O/o and 1/I/l share one shape. If the model is otherwise correct, a large share of its test errors must be such pairs, and accuracy with upper and lower case merged must be much higher.

| model | test accuracy | errors of 3,621 | same letter, other case | within {0, O, o} or {1, I, l} | accuracy with cases merged | accuracy without the 18 duplicated images | accuracy on the 18 |
|---|---|---|---|---|---|---|---|
| seed0 | 0.8078 | 696 | 86 (12%) | 98 (14%) | 0.8315 | 0.8077 | 0.833 |
| seed1 | 0.8012 | 720 | 83 (12%) | 112 (16%) | 0.8241 | 0.8013 | 0.778 |
| seed2 | 0.8092 | 691 | 90 (13%) | 93 (13%) | 0.8340 | 0.8090 | 0.833 |
| seed3 | 0.8028 | 714 | 96 (13%) | 93 (13%) | 0.8293 | 0.8024 | 0.889 |
| seed4 | 0.8053 | 705 | 96 (14%) | 101 (14%) | 0.8318 | 0.8049 | 0.889 |

The random 80/20 split does not remove duplicates. 22 of the 6360 validation images have a pixel-identical copy with the same label in the training split of the same client, and 1 more have one at another client only. A model that had memorised its training images would classify these images almost perfectly. Removing them can change validation accuracy by at most 0.0036.

| model | validation accuracy | without the duplicated images | on the duplicated images |
|---|---|---|---|
| seed0 | 0.8082 | 0.8083 | 0.783 |
| seed1 | 0.8011 | 0.8007 | 0.913 |
| seed2 | 0.8044 | 0.8046 | 0.739 |
| seed3 | 0.8036 | 0.8037 | 0.783 |
| seed4 | 0.8052 | 0.8051 | 0.826 |

Most confused pairs for seed 0 (both directions added): I/l 46, 1/l 17, 0/O 15, C/e 14, M/m 12, F/f 11, 9/q 11, 5/S 11, 1/I 10, 0/o 10, D/O 9, P/p 8. Spearman correlation between a class's number of training images and its test accuracy: 0.79 (p = 2.1e-14).

