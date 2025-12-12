#!/bin/bash
# Systematic experiments for Wavelet-FreqPure ablation study
# This script runs multiple configurations and logs results

# Create results directory
RESULTS_DIR="./ablation_results"
mkdir -p $RESULTS_DIR

# Experiment configurations to test
# Format: "wavelet_levels,delta,description"
CONFIGS=(
    "1,0.3,level1_delta03"
    "2,0.1,level2_delta01"
    "2,0.2,level2_delta02"
    "2,0.5,level2_delta05"
    "3,0.3,level3_delta03"
)

# Number of samples (use 50 for quick ablation, 512 for full)
NUM_SAMPLES=50
BATCH_SIZE=25

echo "=========================================="
echo "Wavelet-FreqPure Ablation Study"
echo "=========================================="
echo "Samples: $NUM_SAMPLES"
echo "Configurations: ${#CONFIGS[@]}"
echo ""

# Run each configuration
for config in "${CONFIGS[@]}"; do
    IFS=',' read -r levels delta desc <<< "$config"
    
    echo "=========================================="
    echo "Running: wavelet_levels=$levels, delta=$delta"
    echo "=========================================="
    
    OUTPUT_FILE="$RESULTS_DIR/results_${desc}.txt"
    
    PATH=/usr/local/cuda-12.6/bin:$PATH \
    CUDA_HOME=/usr/local/cuda-12.6 \
    CUDA_VISIBLE_DEVICES=0,1 \
    torchrun --nproc_per_node=2 ddp_test.py \
        --transform_type wavelet \
        --wavelet_levels $levels \
        --delta $delta \
        --num_samples $NUM_SAMPLES \
        --batch_size $BATCH_SIZE \
        --num_ensemble_runs 10 \
        2>&1 | tee "$OUTPUT_FILE"
    
    echo ""
    echo "Results saved to: $OUTPUT_FILE"
    echo ""
done

echo "=========================================="
echo "All experiments completed!"
echo "Results saved in: $RESULTS_DIR/"
echo "=========================================="
