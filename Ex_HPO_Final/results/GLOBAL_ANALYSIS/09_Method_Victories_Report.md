# Детальний звіт: перемоги та конкуренти кожного методу

16 методів × 43 задачі × 10 seeds = 6880 записів

### ✦ SACMA-DAC — Avg Rank: 4.83 | Wins: 6/43 | Top-3: 20/43 | Avg RCU: 1275

| Task | Loss | RCU | Runner-up | Gap |
|------|------|-----|-----------|-----|
| yahpo__lcbench__3945/yahpo | 0.1141 | 853 | SACMA-base (0.1619) | +41.8% |
| synth_classification/gb | 0.1167 | 1153 | GP-BO (0.1200) | +2.9% |
| residual__fashion_mini/l4 | 0.2130 | 21557 | SACMA-base (0.2200) | +3.3% |
| synth_friedman/hgb | 1.5980 | 502 | SACMA-MAB (1.6022) | +0.3% |
| synth_friedman/svm | 1.8094 | 465 | Sigma-CMA (1.8201) | +0.6% |
| uniref50_transformer/pd1 | 2.6382 | 408 | WL-CMA (2.6391) | +0.0% |

**Також у Top-3 (14 задач):**
  synth_classification/rf (rank 2), yahpo__lcbench__126025/yahpo (rank 2), yahpo__lcbench__126026/yahpo (rank 2)
  cifar100_wresnet/pd1 (rank 2), imagenet_resnet/pd1 (rank 2), yahpo__iaml_super__40981/yahpo (rank 2)
  synth_classification/hgb (rank 2), sequential__digits/l4 (rank 2), synth_friedman/mlp (rank 3)
  synth_regression/gb (rank 3), yahpo__lcbench__167104/yahpo (rank 3), yahpo__lcbench__167149/yahpo (rank 3)
  yahpo__lcbench__168868/yahpo (rank 3), yahpo__lcbench__7593/yahpo (rank 3)

### ✦ SACMA-base — Avg Rank: 5.09 | Wins: 4/43 | Top-3: 16/43 | Avg RCU: 1297

| Task | Loss | RCU | Runner-up | Gap |
|------|------|-----|-----------|-----|
| synth_classification/svm | 0.1033 | 216 | SACMA-lazy (0.1067) | +3.2% |
| yahpo__iaml_super__40981/yahpo | 0.1229 | 730 | SACMA-DAC (0.1234) | +0.4% |
| cifar100_wresnet/pd1 | 0.9290 | 757 | SACMA-DAC (0.9313) | +0.3% |
| synth_regression/svm | 10.1848 | 322 | SACMA-MAB (10.2575) | +0.7% |

**Також у Top-3 (12 задач):**
  synth_classification/mlp (rank 2), yahpo__lcbench__3945/yahpo (rank 2), dense__digits/l4 (rank 2)
  residual__fashion_mini/l4 (rank 2), synth_classification/hgb (rank 2), sequential__digits/l4 (rank 2)
  synth_friedman/rf (rank 3), yahpo__lcbench__126025/yahpo (rank 3), yahpo__lcbench__167168/yahpo (rank 3)
  yahpo__lcbench__34539/yahpo (rank 3), uniref50_transformer/pd1 (rank 3), fcnet_parkinsons/fcnet (rank 3)

### ✦ SACMA-MAB — Avg Rank: 5.23 | Wins: 6/43 | Top-3: 19/43 | Avg RCU: 895

| Task | Loss | RCU | Runner-up | Gap |
|------|------|-----|-----------|-----|
| fcnet_naval/fcnet | 0.0000 | 211 | Sigma-CMA (0.0000) | +13.1% |
| mnist_cnn/pd1 | 0.0304 | 304 | TPE (0.0316) | +3.8% |
| dense__digits/l4 | 0.0600 | 4070 | SACMA-base (0.0675) | +12.5% |
| yahpo__lcbench__34539/yahpo | 0.3733 | 196 | Sigma-CMA (0.4021) | +7.7% |
| imagenet_resnet/pd1 | 0.9966 | 310 | SACMA-DAC (1.0000) | +0.3% |
| synth_regression/gb | 35.2457 | 502 | Sigma-CMA (35.4600) | +0.6% |

**Також у Top-3 (13 задач):**
  synth_friedman/gb (rank 2), synth_friedman/hgb (rank 2), synth_regression/hgb (rank 2)
  synth_regression/mlp (rank 2), synth_regression/svm (rank 2), yahpo__lcbench__167104/yahpo (rank 2)
  yahpo__lcbench__167149/yahpo (rank 2), yahpo__lcbench__167168/yahpo (rank 2), yahpo__lcbench__168868/yahpo (rank 2)
  yahpo__lcbench__7593/yahpo (rank 2), fcnet_parkinsons/fcnet (rank 2), yahpo__lcbench__126026/yahpo (rank 3)
  yahpo__lcbench__3945/yahpo (rank 3)

### ✦ Sigma-CMA — Avg Rank: 6.21 | Wins: 1/43 | Top-3: 14/43 | Avg RCU: 832

| Task | Loss | RCU | Runner-up | Gap |
|------|------|-----|-----------|-----|
| synth_friedman/mlp | 1.5818 | 666 | TPE (1.6063) | +1.5% |

**Також у Top-3 (13 задач):**
  synth_friedman/svm (rank 2), synth_regression/gb (rank 2), yahpo__lcbench__34539/yahpo (rank 2)
  cifar10_wresnet/pd1 (rank 2), translate_transformer/pd1 (rank 2), fcnet_naval/fcnet (rank 2)
  fcnet_protein/fcnet (rank 2), synth_regression/hgb (rank 3), synth_regression/rf (rank 3)
  synth_regression/svm (rank 3), imagenet_resnet/pd1 (rank 3), mnist_cnn/pd1 (rank 3)
  fcnet_slice/fcnet (rank 3)

### ✦ WL-CMA — Avg Rank: 6.23 | Wins: 3/43 | Top-3: 10/43 | Avg RCU: 1582

| Task | Loss | RCU | Runner-up | Gap |
|------|------|-----|-----------|-----|
| fcnet_parkinsons/fcnet | 0.0099 | 1122 | SACMA-MAB (0.0100) | +0.4% |
| synth_friedman/gb | 1.5577 | 877 | SACMA-MAB (1.5644) | +0.4% |
| translate_transformer/pd1 | 1.7194 | 688 | Sigma-CMA (1.7195) | +0.0% |

**Також у Top-3 (7 задач):**
  synth_friedman/rf (rank 2), synth_regression/rf (rank 2), fashion_cnn/pd1 (rank 2)
  uniref50_transformer/pd1 (rank 2), lm1b_transformer/pd1 (rank 3), svhn_wresnet/pd1 (rank 3)
  yahpo__iaml_super__40981/yahpo (rank 3)

###   TPE — Avg Rank: 6.73 | Wins: 3/43 | Top-3: 9/43 | Avg RCU: 794

| Task | Loss | RCU | Runner-up | Gap |
|------|------|-----|-----------|-----|
| fcnet_slice/fcnet | 0.0003 | 20 | IW-MOEA (0.0003) | +2.8% |
| fcnet_protein/fcnet | 0.2238 | 20 | Sigma-CMA (0.2306) | +3.0% |
| synth_regression/rf | 65.3883 | 702 | WL-CMA (65.6012) | +0.3% |

**Також у Top-3 (6 задач):**
  synth_friedman/mlp (rank 2), mnist_cnn/pd1 (rank 2), synth_classification/hgb (rank 2)
  sequential__digits/l4 (rank 2), synth_friedman/svm (rank 3), yahpo__lcbench__167152/yahpo (rank 3)

###   GP-BO — Avg Rank: 6.93 | Wins: 11/43 | Top-3: 14/43 | Avg RCU: 1389

| Task | Loss | RCU | Runner-up | Gap |
|------|------|-----|-----------|-----|
| yahpo__lcbench__126026/yahpo | 0.0284 | 1539 | SACMA-DAC (0.0297) | +4.3% |
| yahpo__lcbench__168868/yahpo | 0.0307 | 1324 | SACMA-MAB (0.0343) | +11.9% |
| yahpo__lcbench__167149/yahpo | 0.0419 | 1692 | SACMA-MAB (0.0446) | +6.3% |
| yahpo__lcbench__126025/yahpo | 0.0828 | 1809 | SACMA-DAC (0.0985) | +19.1% |
| yahpo__lcbench__167104/yahpo | 0.1154 | 1268 | SACMA-MAB (0.1240) | +7.4% |
| yahpo__lcbench__167152/yahpo | 0.1161 | 1517 | SMAC (0.1204) | +3.7% |
| yahpo__lcbench__7593/yahpo | 0.1496 | 2393 | SACMA-MAB (0.1623) | +8.4% |
| yahpo__lcbench__167168/yahpo | 0.2232 | 2229 | SACMA-MAB (0.2535) | +13.5% |
| synth_friedman/rf | 2.0990 | 859 | WL-CMA (2.0996) | +0.0% |
| lm1b_transformer/pd1 | 3.4803 | 490 | SHADE (3.4884) | +0.2% |
| synth_regression/hgb | 41.4581 | 682 | SACMA-MAB (42.3064) | +2.0% |

**Також у Top-3 (3 задач):**
  synth_friedman/hgb (rank 3), fashion_cnn/pd1 (rank 3), dense__digits/l4 (rank 3)

###   SMAC — Avg Rank: 7.55 | Wins: 1/43 | Top-3: 5/43 | Avg RCU: 809

| Task | Loss | RCU | Runner-up | Gap |
|------|------|-----|-----------|-----|
| synth_classification/rf | 0.1333 | 834 | SACMA-DAC (0.1367) | +2.5% |

**Також у Top-3 (4 задач):**
  yahpo__lcbench__167152/yahpo (rank 2), yahpo__nb301__None/yahpo (rank 3), residual__fashion_mini/l4 (rank 3)
  fcnet_protein/fcnet (rank 3)

### ✦ SACMA-lazy — Avg Rank: 8.03 | Wins: 1/43 | Top-3: 7/43 | Avg RCU: 1962

| Task | Loss | RCU | Runner-up | Gap |
|------|------|-----|-----------|-----|
| yahpo__nb301__None/yahpo | 0.0564 | 562 | CMA-ES (0.0581) | +3.0% |

**Також у Top-3 (6 задач):**
  synth_classification/mlp (rank 2), synth_classification/svm (rank 2), synth_classification/hgb (rank 2)
  sequential__digits/l4 (rank 2), synth_regression/mlp (rank 3), cifar100_wresnet/pd1 (rank 3)

###   L-SHADE — Avg Rank: 8.83 | Wins: 3/43 | Top-3: 4/43 | Avg RCU: 710

| Task | Loss | RCU | Runner-up | Gap |
|------|------|-----|-----------|-----|
| svhn_wresnet/pd1 | 0.2078 | 67 | DEHB (0.2104) | +1.2% |
| fashion_cnn/pd1 | 0.2335 | 91 | WL-CMA (0.2347) | +0.5% |
| synth_regression/mlp | 10.7639 | 1179 | SACMA-MAB (10.7862) | +0.2% |

**Також у Top-3 (1 задач):**
  cifar10_wresnet/pd1 (rank 3)

### ✦ IW-MOEA — Avg Rank: 9.83 | Wins: 1/43 | Top-3: 3/43 | Avg RCU: 1075

| Task | Loss | RCU | Runner-up | Gap |
|------|------|-----|-----------|-----|
| cifar10_wresnet/pd1 | 0.2599 | 301 | Sigma-CMA (0.2616) | +0.7% |

**Також у Top-3 (2 задач):**
  fcnet_slice/fcnet (rank 2), fcnet_naval/fcnet (rank 3)

### ✦ OrdInv-CMA — Avg Rank: 10.28 | Wins: 0/43 | Top-3: 0/43 | Avg RCU: 874
  *Жодної перемоги (rank=1) на жодній задачі.*

###   CMA-ES — Avg Rank: 11.36 | Wins: 0/43 | Top-3: 2/43 | Avg RCU: 752
  *Жодної перемоги (rank=1) на жодній задачі.*

**Також у Top-3 (2 задач):**
  yahpo__nb301__None/yahpo (rank 2), synth_friedman/gb (rank 3)

###   SHADE — Avg Rank: 11.49 | Wins: 0/43 | Top-3: 1/43 | Avg RCU: 896
  *Жодної перемоги (rank=1) на жодній задачі.*

**Також у Top-3 (1 задач):**
  lm1b_transformer/pd1 (rank 2)

###   DEHB — Avg Rank: 13.10 | Wins: 0/43 | Top-3: 1/43 | Avg RCU: 857
  *Жодної перемоги (rank=1) на жодній задачі.*

**Також у Top-3 (1 задач):**
  svhn_wresnet/pd1 (rank 2)

###   Random — Avg Rank: 14.28 | Wins: 0/43 | Top-3: 0/43 | Avg RCU: 688
  *Жодної перемоги (rank=1) на жодній задачі.*