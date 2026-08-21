#!/bin/bash
# Run OrdInv-CMA across all tiers with 10 seeds
# Does NOT touch any existing results — only creates new ordinv_cma__*.json files

set -e

METHOD="ordinv_cma"
SEEDS=10

echo "========================================"
echo "  OrdInv-CMA Full Benchmark Run"
echo "  Method: $METHOD | Seeds: $SEEDS"
echo "========================================"

cd "$(dirname "$0")"

# L0: synth tasks, sklearn models (fast)
echo ""
echo ">>> Tier L0 (synth × sklearn) ..."
python3 run_method.py $METHOD L0 $SEEDS

# L2: LCBench YAHPO surrogates (fast)
echo ""
echo ">>> Tier L2 (LCBench YAHPO) ..."
python3 run_method.py $METHOD L2 $SEEDS

# L2_MLP_PD1: PD1 surrogates — WideResNet, Transformer, ResNet (fast)
echo ""
echo ">>> Tier L2_MLP_PD1 (PD1 surrogates) ..."
python3 run_method.py $METHOD L2_MLP_PD1 $SEEDS

# L3_NAS_SUPER: NAS-Bench-301 + IAML (fast)
echo ""
echo ">>> Tier L3_NAS_SUPER ..."
python3 run_method.py $METHOD L3_NAS_SUPER $SEEDS

# L4: Real PyTorch training (slower, ~2-5 min per seed)
echo ""
echo ">>> Tier L4 (real PyTorch) ..."
python3 run_method.py $METHOD L4 $SEEDS

# L5_FCNET: FCNet tabular benchmark (fast)
echo ""
echo ">>> Tier L5_FCNET ..."
python3 run_method.py $METHOD L5_FCNET $SEEDS

# L_ABLATION is skipped — it's for SACMA variants only

echo ""
echo "========================================"
echo "  OrdInv-CMA Benchmark COMPLETE"
echo "========================================"
