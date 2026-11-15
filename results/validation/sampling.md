## F6. Client sampling statistics

Prediction: every round has exactly 10 distinct clients; a client's number of selections in 300 rounds is Binomial(300, 0.1) with mean 30 and standard deviation 5.2; the expected number of clients not yet selected after 10 rounds is 100 x 0.9^10 = 34.9. The chi-square statistic has 99 degrees of freedom.

| run | 10 distinct clients in every round | expected selections | fewest | most | chi-square | p-value | clients unseen after 10 rounds | round by which all were selected |
|---|---|---|---|---|---|---|---|---|
| seed0 | True | 30 | 19 | 46 | 94.7 | 0.60 | 26 | 85 |
| seed1 | True | 30 | 17 | 44 | 108.0 | 0.25 | 31 | 48 |
| seed2 | True | 30 | 18 | 51 | 113.0 | 0.16 | 30 | 71 |
| seed3 | True | 30 | 17 | 51 | 111.9 | 0.18 | 33 | 38 |
| seed4 | True | 30 | 16 | 47 | 126.1 | 0.03 | 36 | 46 |

