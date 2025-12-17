# Wavelet-FreqPure Ablation Study Results

## Summary Table

All experiments run on 50 CIFAR-10 samples with:
- PGD-200 attack with EOT-20
- ε = 8/255

| Configuration | Levels | Delta | Natural Acc | Adversarial Acc | Notes |
|--------------|--------|-------|-------------|-----------------|-------|
| **DFT Baseline** | - | 0.3 | 94.34% | 69.73% | Original method (512 samples) |
| Wavelet | 1 | 0.3 | **98%** | 82% | Best natural accuracy |
| Wavelet | 2 | 0.1 | 88% | **90%** | **Best adversarial accuracy** |
| Wavelet | 2 | 0.2 | 84% | 72% | Similar to baseline |
| Wavelet | 2 | 0.3 | 80% | 72% | Quick test baseline |
| Wavelet | 2 | 0.5 | 84% | 72% | Higher delta, similar results |
| Wavelet | 3 | 0.3 | 60% | 52% | Too aggressive, LL too small |

## Key Findings

### 1. Delta Effect (at Level 2)
- **δ = 0.1**: Best adversarial accuracy (90%) - tight constraint preserves adversarial info
- **δ = 0.2-0.5**: Similar performance (~72%) - trade-off region
- Smaller delta = stricter projection = more adversarial-like details preserved

### 2. Level Effect (at δ = 0.3)
- **Level 1**: Best natural (98%), good adversarial (82%) - preserves most detail
- **Level 2**: Balanced (80% nat, 72% adv) - default choice
- **Level 3**: Too aggressive (60% nat, 52% adv) - LL too small (4×4), loses information

### 3. Best Configuration
For adversarial robustness: **Level 2, Delta 0.1** → 88% natural, **90% adversarial**

## Interpretation

| Parameter | Effect of Increasing |
|-----------|---------------------|
| `wavelet_levels` | More aggressive purification, smaller LL band, risk of losing structure |
| `delta` | Looser constraint on details, more diffusion freedom, less adversarial structure |

## Recommended Settings

| Use Case | Levels | Delta | Expected Performance |
|----------|--------|-------|----------------------|
| **Max Robustness** | 2 | 0.1 | ~90% adv acc, ~88% nat acc |
| **Balanced** | 2 | 0.3 | ~72% adv acc, ~80% nat acc |
| **Max Natural Acc** | 1 | 0.3 | ~82% adv acc, ~98% nat acc |

## Experiment Log Files
- `results_level1_delta03.txt` - Level 1, δ=0.3
- `results_level2_delta01.txt` - Level 2, δ=0.1
- `results_level2_delta02.txt` - Level 2, δ=0.2
- `results_level2_delta05.txt` - Level 2, δ=0.5
- `results_level3_delta03.txt` - Level 3, δ=0.3

## Full Dataset Results (512 samples)
- `../results_dft.txt` - DFT baseline (94.34% nat, 69.73% adv)
- `../results_wavelet.txt` - Level 2, δ=0.3 (83.01% nat, 78.13% adv)
