# Wavelet-FreqPure: Technical Summary Report

## Overview

This report summarizes the implementation of **Wavelet-based Frequency Purification** as an alternative to the original DFT-based approach in the FreqPure defense system.

---

## 1. What is FreqPure?

FreqPure is a **defense against adversarial attacks** on image classifiers. It works by:
1. Taking an adversarial image (image with small, invisible perturbations that fool the classifier)
2. Running it through a **diffusion model** (denoising process)
3. **Exchanging frequency components** between the original and denoised image
4. The result is a "purified" image that the classifier can correctly identify

### The Original DFT Approach
The original paper uses **DFT (Discrete DFT Transform)** to:
- Separate images into frequency components
- Replace low-frequency amplitude (structure) from the adversarial image
- Constrain low-frequency phase to stay close to the adversarial image
- Let the diffusion model regenerate high frequencies

---

## 2. Our Modification: Wavelet Transform

### Why Wavelets?

| Feature | DFT | Wavelet (DWT) |
|---------|-----|---------------|
| **Spatial Info** | Lost (global transform) | Preserved (localized) |
| **Multi-scale** | Single scale | Natural multi-scale |
| **Edge Handling** | Ringing artifacts | Sharp edges preserved |

### What We Changed

We replaced the DFT frequency exchange with a **Haar Wavelet Transform (DWT)** that provides:
- **Multi-scale decomposition**: Break image into multiple resolution levels
- **Spatially localized**: Know WHERE frequencies occur, not just WHAT frequencies

---

## 3. Files Modified

### New File: `wavelet_utils.py`
```
Purpose: Implement differentiable Haar Wavelet Transform in PyTorch

Functions:
├── dwt_2d()          # Single-level 2D wavelet decomposition
├── idwt_2d()         # Single-level inverse transform
├── pad_if_needed()   # Handle odd-sized inputs
├── multi_level_dwt() # Multi-level decomposition (default: 2 levels)
└── multi_level_idwt() # Multi-level reconstruction
```

**How DWT works:**
```
Original Image (32x32)
        │
        ▼
    ┌───────────────────────────────────┐
    │         2D Haar DWT               │
    └───────────────────────────────────┘
        │
        ▼
    Level 1 Output:
    ┌─────────┬─────────┐
    │   LL    │   HL    │  LL = Low-Low (structure/average)
    │  (16x16)│  (16x16)│  HL = High-Low (vertical edges)
    ├─────────┼─────────┤  LH = Low-High (horizontal edges)
    │   LH    │   HH    │  HH = High-High (diagonal details)
    │  (16x16)│  (16x16)│
    └─────────┴─────────┘
        │
        ▼ (Apply DWT again to LL)
    Level 2 Output:
    ┌────┬────┬─────────┐
    │LL2 │HL2 │         │
    ├────┼────┤   HL1   │
    │LH2 │HH2 │         │
    ├────┴────┼─────────┤
    │   LH1   │   HH1   │
    └─────────┴─────────┘
```

### Modified File: `purification.py`

**Changes to `PurificationForward` class:**

```python
# NEW Parameters added to __init__:
transform_type='dft'    # 'dft' (original) or 'wavelet' (new)
wavelet_levels=2        # Number of decomposition levels

# NEW Method added:
def wavelet_exchange(self, x_ref, x_est):
    """
    x_ref: Reference image (adversarial input)
    x_est: Estimated image (from diffusion model)
    """
```

**The Wavelet Exchange Logic:**

```
┌─────────────────────────────────────────────────────────────┐
│                    WAVELET EXCHANGE                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1: Decompose both images with DWT                     │
│                                                             │
│    Adversarial ──► DWT ──► LL_adv  + Details_adv           │
│    Diffusion   ──► DWT ──► LL_diff + Details_diff          │
│                                                             │
│  Step 2: Exchange LL (Low Frequency = Structure)            │
│                                                             │
│    LL_new = LL_adv   ◄── Keep adversarial's structure       │
│                          (less perturbed, more reliable)    │
│                                                             │
│  Step 3: Project Details (High Frequency)                   │
│                                                             │
│    For each detail band (LH, HL, HH):                       │
│      Detail_new = clamp(Detail_diff,                        │
│                         Detail_adv - δ,                     │
│                         Detail_adv + δ)                     │
│                                                             │
│      ◄── Allow diffusion to modify details                  │
│          but keep them close to original (within δ=0.3)     │
│                                                             │
│  Step 4: Reconstruct with Inverse DWT                       │
│                                                             │
│    Purified = IDWT(LL_new, Details_new)                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Modified File: `ddp_test.py`

**New command-line arguments:**
```bash
--transform_type dft|wavelet   # Choose which method to use
--wavelet_levels 2             # Number of decomposition levels
```

---

## 4. Key Design Decisions

### Why replace LL completely?
- LL contains **low-frequency structure** (shapes, colors, major features)
- Adversarial perturbations are typically **high-frequency** (noise-like)
- LL is therefore **less corrupted** → safe to use directly

### Why project (clamp) details instead of replacing?
- Details contain **edges and textures** 
- Some adversarial noise IS in details, but so is important texture info
- Projection allows diffusion to **clean the noise** while staying **close to original**
- δ=0.3 means each coefficient can change by ±0.3 from original

### Why 2 levels?
- For 32×32 CIFAR-10 images:
  - Level 1: LL is 16×16 (keeps moderate detail)
  - Level 2: LL is 8×8 (coarser structure)
- 2 levels balances **structure preservation** vs **purification strength**

---

## 5. Results Summary

### Full CIFAR-10 Evaluation (512 samples)

| Metric | DFT-FreqPure | Wavelet-FreqPure | Change |
|--------|--------------|------------------|--------|
| **Natural Accuracy** | 94.34% | 83.01% | -11.33% |
| **Adversarial Accuracy** | 69.73% | **78.13%** | **+8.40%** |

### Interpretation

1. **Better Robustness**: +8.4% improvement against PGD-200 attack with EOT-20
   - This is a strong adaptive attack
   - Improvement suggests wavelets better preserve adversarial-resistant structure

2. **Lower Natural Accuracy**: -11.3% on clean images
   - Trade-off: stronger purification → more distortion
   - LL replacement is "aggressive" - might lose some texture

3. **Why This Happens**:
   - DFT: Global frequency decomposition, circular masks
   - DWT: Localized decomposition, multi-scale processing
   - Adversarial noise is better separated in wavelet domain

---

## 6. How to Run

### Quick Test (50 samples)
```bash
# Wavelet-FreqPure
torchrun --nproc_per_node=2 ddp_test.py \
    --transform_type wavelet \
    --wavelet_levels 2 \
    --num_samples 50

# DFT-FreqPure (baseline)
torchrun --nproc_per_node=2 ddp_test.py \
    --transform_type dft \
    --num_samples 50
```

### Full Evaluation (512 samples)
```bash
torchrun --nproc_per_node=2 ddp_test.py \
    --transform_type wavelet \
    --num_samples 512
```

---

## 7. Files Overview

```
FreqPure-main/
├── wavelet_utils.py        # NEW: Haar DWT/IDWT implementation
├── test_wavelet.py         # NEW: Unit tests for wavelet transform
├── purification.py         # MODIFIED: Added wavelet_exchange method
├── ddp_test.py            # MODIFIED: Added CLI arguments
├── visualize_comparison.py # NEW: Visualization script
├── results_dft.txt        # DFT evaluation results
├── results_wavelet.txt    # Wavelet evaluation results
├── comparison_results/    # Visualization outputs
│   ├── summary_grid.png
│   ├── wavelet_decomposition.png
│   └── comparison_sample_*.png
└── SETUP.md               # Updated setup instructions
```

---

## 8. Potential Improvements

1. **Tune δ (delta)**: Try δ=0.1, 0.2, 0.5 to find optimal trade-off
2. **Try different wavelet levels**: Level 1 (less aggressive) or Level 3 (more aggressive)
3. **Different wavelets**: Daubechies (db2, db4) instead of Haar
4. **Hybrid approach**: Blend DFT and DWT results

---

## 9. Conclusion

The Wavelet-based FreqPure modification successfully improves adversarial robustness by +8.4% at the cost of -11.3% natural accuracy. The multi-scale, spatially-localized nature of wavelet decomposition provides better separation of adversarial perturbations from image structure, leading to more effective purification against strong adaptive attacks.
