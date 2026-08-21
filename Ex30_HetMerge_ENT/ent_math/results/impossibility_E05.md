# Impossibility Analysis: Data-Free Proxies for CNN Merge (E04-E05)

## Finding

Data-free zero-cost proxies (DAVE and output balance) on random noise
do NOT provide useful signal for guiding CNN merge optimization.

## Evidence

| Proxy | Correlation with acc | Correlation with min_class | p-value |
|-------|:-------------------:|:--------------------------:|:-------:|
| DAVE (var+entropy) | r=+0.13 | r=+0.07 | p=0.57 |
| Balance (KL+entropy) | r=−0.38 | r=−0.44 | p=0.02 |

DAVE: essentially random (no signal).
Balance: ANTI-correlates — maximizing balance makes accuracy WORSE.

## Root Cause

Random noise N(0,1) does not exercise learned features.
- Conv features are tuned to natural image statistics (edges, textures, etc.)
- Random noise produces near-random activations regardless of weight quality
- BN layers (with noise-based stats) further flatten the signal

The fundamental issue: model quality is defined by behavior on the DATA DISTRIBUTION,
but random noise lies far outside that distribution.

## What Works Instead

E01-E02 (permutation alignment + real data BN + EA) achieved 4/4 on toy.
The key was using REAL calibration data for both BN reset and EA fitness.

## Decision

Accept impossibility of data-free proxy approach for this model scale.
Proceed with the proven data-aware pipeline (alignment + real BN + EA).
