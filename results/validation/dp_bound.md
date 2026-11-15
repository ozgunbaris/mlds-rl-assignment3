## N2. The differential-privacy inequality of the Laplace mechanism

For a released image o = x + e the log-likelihood ratio between two candidate originals x and x' is (||o - x'||_1 - ||o - x||_1) / b. The triangle inequality bounds it by ||x - x'||_1 / b, which is the epsilon of the Laplace mechanism for that pair; the worst case over all images in [0, 1]^784 is 784 / b. Checked on 20,000 random pairs of training images: their L1 distance is 37.6 on average and at most 117.1.

| b | epsilon per pixel (1/b) | epsilon per image, worst case (784/b) | epsilon for an average pair | mean measured privacy loss | largest loss / bound | pairs above the bound | share of pairs where the released image favours the wrong original |
|---|---|---|---|---|---|---|---|
| 0.1 | 10.00 | 7840 | 375.8 | 304.82 | 0.936 | 0 | 0.0000 |
| 0.5 | 2.00 | 1568 | 75.2 | 32.90 | 0.825 | 0 | 0.0011 |
| 1.0 | 1.00 | 784 | 37.6 | 10.15 | 0.764 | 0 | 0.0225 |
| 2.0 | 0.50 | 392 | 18.8 | 2.82 | 0.721 | 0 | 0.1331 |
| 5.0 | 0.20 | 157 | 7.5 | 0.49 | 0.715 | 0 | 0.3136 |
| 20.0 | 0.05 | 39 | 1.9 | 0.03 | 0.671 | 0 | 0.4532 |
| 100.0 | 0.01 | 8 | 0.4 | 0.00 | 0.721 | 0 | 0.4840 |

The last column is the error rate of the best possible guess between two equally likely candidate images.

