# Аналіз ефективності HPO з урахуванням RCU (Total Deployment Cost)

## Концепція

Складніший метод HPO може бути **вигіднішим** якщо він:
1. Знаходить кращі гіперпараметри (нижчий loss)
2. Робить це за менше evaluations (швидше конвергує)
3. Тому **сумарна вартість** (HPO + фінальне навчання) — нижча

**Total Deployment Cost** = RCU\_HPO + RCU\_final\_training

## Time-to-5%-Threshold

Поріг: loss ≤ 1.05 × best\_known\_loss (в межах 5% від найкращого)

| Method | Type | Steps to 5% | Reach Rate | RCU total | RCU to 5% |
|--------|------|-------------|------------|-----------|-----------|
| Random | Base | 39.5 | 8.8% | 688 | 715 |
| L-SHADE | Base | 38.4 | 14.9% | 710 | 735 |
| CMA-ES | Base | 39.2 | 12.1% | 752 | 780 |
| TPE | Base | 37.9 | 17.7% | 794 | 822 |
| SMAC | Base | 38.5 | 17.0% | 809 | 833 |
| Sigma-CMA | **Author** | 38.6 | 17.2% | 832 | 861 |
| DEHB | Base | 38.5 | 10.0% | 857 | 891 |
| OrdInv-CMA | **Author** | 38.7 | 13.0% | 874 | 906 |
| SACMA-MAB | **Author** | 38.3 | 19.1% | 895 | 912 |
| SHADE | Base | 38.5 | 12.8% | 896 | 931 |
| IW-MOEA | **Author** | 38.7 | 13.3% | 1075 | 1104 |
| SACMA-base | **Author** | 37.7 | 20.0% | 1297 | 1272 |
| SACMA-DAC | **Author** | 38.2 | 19.8% | 1275 | 1283 |
| GP-BO | Base | 37.5 | 22.1% | 1389 | 1398 |
| WL-CMA | **Author** | 37.8 | 14.7% | 1582 | 1570 |
| SACMA-lazy | **Author** | 38.1 | 15.6% | 1962 | 1858 |

## Quality-per-Deployment-Cost Ranking

Формула: Quality/Cost = (1 / (1 + loss)) / TotalDeployCost × 10000

| # | Method | Type | Loss | RCU HPO | Deploy Cost | Quality/Cost |
|---|--------|------|------|---------|-------------|-------------|
|  | DEHB | Base | 0.2481 | 58 | 60 | 134.640 |
|  | Random | Base | 0.2527 | 60 | 61 | 131.207 |
|  | L-SHADE | Base | 0.2337 | 62 | 63 | 128.735 |
| 4 | CMA-ES | Base | 0.2460 | 62 | 63 | 127.426 |
| 5 | SHADE | Base | 0.2411 | 65 | 66 | 122.186 |
| 6 | OrdInv-CMA | **Author** | 0.2425 | 72 | 74 | 109.398 |
| 7 | TPE | Base | 0.2306 | 95 | 96 | 84.240 |
| 8 | Sigma-CMA | **Author** | 0.2315 | 97 | 99 | 82.073 |
| 9 | SMAC | Base | 0.2323 | 111 | 113 | 71.627 |
| 10 | IW-MOEA | **Author** | 0.2387 | 237 | 242 | 33.382 |
| 11 | SACMA-MAB | **Author** | 0.2253 | 325 | 331 | 24.637 |
| 12 | GP-BO | Base | 0.2217 | 687 | 701 | 11.675 |
| 13 | SACMA-DAC | **Author** | 0.2206 | 694 | 708 | 11.569 |
| 14 | SACMA-base | **Author** | 0.2240 | 752 | 767 | 10.657 |
| 15 | WL-CMA | **Author** | 0.2317 | 935 | 954 | 8.511 |
| 16 | SACMA-lazy | **Author** | 0.2357 | 1298 | 1324 | 6.111 |