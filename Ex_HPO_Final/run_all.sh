#!/bin/bash
echo "Starting L2 benchmark..."
python3 run_benchmark.py L2 --force
echo "Starting L4 benchmark..."
python3 run_benchmark.py L4 --force
echo "Starting L_ABLATION benchmark..."
python3 run_benchmark.py L_ABLATION --force
echo "All benchmarks completed!"
