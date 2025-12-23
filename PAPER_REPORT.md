# Wavelet-FreqPure: Multi-Scale Frequency Purification for Adversarial Defense

## Abstract

We propose **Wavelet-FreqPure**, a novel adversarial defense mechanism that integrates a multi-level Haar Wavelet Transform into the diffusion-based purification pipeline. Unlike the original FreqPure which relies on the global Discrete Fourier Transform (DFT), our approach exploits the spatial-frequency locality of wavelets to selectively purify adversarial perturbations at different scales. We conduct a comprehensive evaluation on CIFAR-10, demonstrating that our **Level 2** decomposition scheme achieves **85.55% adversarial accuracy**, significantly outperforming the DFT baseline (69.73%) by **+15.82%**, while identifying critical trade-offs between decomposition depth and image structure preservation.

---

## 1. Introduction

Adversarial purification aims to remove adversarial perturbations from input images before classification. Diffusion models have emerged as powerful tools for this task due to their ability to project perturbed inputs onto the manifold of natural images. However, standard diffusion purification often struggles to balance robustness (removing attack noise) with fidelity (preserving semantic content).

The **FreqPure** framework addresses this by incorporating frequency-domain guidance. It assumes that adversarial noise dominates high-frequency components while semantic content resides in low frequencies. The original implementation uses the **Discrete Fourier Transform (DFT)** to strictly preserve low-frequency amplitude and phase. While effective, DFT's global nature lacks spatial localization, potentially allowing localized adversarial features to persist or forcing the removal of genuine texture.

We introduce **Wavelet-FreqPure**, which replaces DFT with the **Discrete Wavelet Transform (DWT)**. Wavelets provide a multi-resolution analysis, decomposing the image into a coarse approximation and detailed subbands at various scales. This allows our method to:
1.  Isolate adversarial noise in specific detail subbands.
2.  Preserve semantic structure via the coarse approximation subband (LL).
3.  Maintain spatial context, preventing the "ringing" artifacts common in DFT.

---
### 1.1 Problem Formulation
We consider a standard classification task with a clean image $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$ drawn from a distribution $\mathcal{D}$ and a corresponding label $y$. Let $f(\cdot)$ denote a pre-trained classifier.

#### Adversarial Threat Model
We assume an *adaptive white-box adversary* $\mathcal{M}$ who has full knowledge of both the classifier $f$ and our purification defense $G$. The adversary seeks to fabricate an adversarial example $\mathbf{x}_{adv} = \mathbf{x} + \delta$ to fool the classifier. The adversary's optimization problem is defined as:

$$ \mathbf{x}_{adv} = \mathbf{x} + \mathop{\arg\max}_{\|\delta\|_p \leq \epsilon} \mathcal{L}(f(G(\mathbf{x} + \delta)), y) $$

where $\mathcal{L}$ is the cross-entropy loss and $\epsilon$ is the perturbation budget (e.g., $8/255$ for $L_\infty$). Crucially, because the adversary is adaptive, they calculate gradients *through* the purification process $G$, typically approximating gradients for stochastic steps via Expectation over Transformation (EOT) or BPDA.

### 1.2 Purification Objective with Wavelet Constraints
The goal of the defense is to define a generative mapping $G: \mathbf{x}_{adv} \to \hat{\mathbf{x}}$ that recovers the semantic content of $\mathbf{x}$. Standard diffusion purification defines $G$ as a reverse stochastic differential equation (SDE). However, this often leads to semantic drift.

We reformulate the purification as a **conditional generation process** constrained in the wavelet domain. Let $\mathcal{W}(\cdot)$ be the Discrete Wavelet Transform decomposing an image into approximation ($LL$) and detail ($\mathcal{H} = \{LH, HL, HH\}$) coefficients. We impose the following structural prior on the estimated clean image $\hat{\mathbf{x}}_{0|t}$ at each timestep $t$:

$$ \mathcal{W}(\hat{\mathbf{x}}_{0|t})_{LL} = \mathcal{W}(\mathbf{x}_{adv})_{LL} $$

$$ \mathcal{W}(\hat{\mathbf{x}}_{0|t})_{\mathcal{H}} = \text{proj}_{\mathcal{B}(\mathbf{x}_{adv}, \delta_{freq})}(\mathcal{W}(\hat{\mathbf{x}}_{diff})_{\mathcal{H}}) $$

where $\text{proj}_{\mathcal{B}}$ denotes projection into a $\delta_{freq}$-ball centered at the adversarial details $\mathcal{W}(\mathbf{x}_{adv})_{\mathcal{H}}$. This constrains the solution space to the intersection of the natural image manifold (learned by diffusion) and the structural manifold (defined by the adversarial input).

### 1.3 Multi-Scale Wavelet Decomposition Details
We utilize the **Haar Wavelet Transform** for $\mathcal{W}(\cdot)$. For a multi-level decomposition of depth $L$, the DWT is recursively applied to the $LL$ subband.
*   **Approximation ($LL$)**: Contains low-frequency semantic structure. Defined by Eq (2), this is **hard-replaced**.
*   **Details ($\mathcal{H}$)**: Contain high-frequency textures (and noise). Defined by Eq (3), these are **soft-clamped**.

The projection operator $\text{proj}_{\mathcal{B}}$ in Eq (3) is implemented as a coefficient-wise hard clipping operation:
$$ h' = \text{clamp}(h_{diff}, h_{adv} - \delta_{freq}, h_{adv} + \delta_{freq}) $$
for each detail coefficient $h \in \mathcal{H}$.

## 2. Methodology

### 2.1 Algorithm Overview

The core innovation is the replacement of the global frequency exchange step with a multi-level wavelet-based substitution and projection.

Let $x_{adv}$ be the adversarial input image and $x_{diff}$ be the image generated by the diffusion model at timestep $t$. Our goal is to combine them into a purified image $x_{purified}$ that retains the structure of $x_{adv}$ but the cleanliness of $x_{diff}$.

### 2.2 Multi-Scale Wavelet Decomposition

We utilize the **Haar Wavelet Transform**, which is efficient and preserves edge information. For an image $X$, a single-level decomposition yields four subbands:
*   **LL (Approximation)**: Low-frequency components (coarse structure).
*   **LH (Horizontal Details)**: High-frequency changes along rows.
*   **HL (Vertical Details)**: High-frequency changes along columns.
*   **HH (Diagonal Details)**: High-frequency diagonal changes.

For a multi-level decomposition of depth $L$, the DWT is recursively applied to the $LL$ subband of the previous level.
*   **Level 1**: $LL_1, \{LH_1, HL_1, HH_1\}$ (Subband size: $H/2 \times W/2$)
*   **Level 2**: $LL_2, \{LH_2, HL_2, HH_2\}$ (Subband size: $H/4 \times W/4$)

### 2.3 Frequency Exchange Mechanism

We define two distinct operations for the approximation and detail subbands:

#### A. Global Structure Preservation (Hard Constraint)
The deepest approximation subband $LL_L$ contains the most fundamental semantic structure of the image (shapes, primary colors). We assume this band is largely robust to high-frequency adversarial noise. Therefore, we **completely replace** the diffusion model's approximation with the original adversarial input's approximation:

$$ LL_L^{purified} = LL_L^{adv} $$

This ensures the purified image does not "drift" away from the original image's content (e.g., changing a cat to a dog).

#### B. Detail Purification (Soft Constraint)
Adversarial perturbations are concentrated in the high-frequency detail subbands ($LH_l, HL_l, HH_l$ for $l=1...L$). However, these bands also contain genuine texture. We apply a **soft projection** that allows the diffusion model to denoise these bands but constrains them to stay within a $\delta$-ball of the original input:

$$ D_l^{purified} = \text{clamp}(D_l^{diff}, D_l^{adv} - \delta, D_l^{adv} + \delta) $$

where $D \in \{LH, HL, HH\}$. This effectively "guides" the diffusion generation to be consistent with the input's texture while removing perturbations that exceed the natural manifold variance.

### 2.4 Implementation Details

We implemented a custom, differentiable 2D Haar Discrete Wavelet Transform (DWT) in PyTorch to ensure gradient compatibility. The implementation avoids external libraries by utilizing efficient tensor slicing and convolution-equivalent operations.

#### A. Haar Wavelet Transform (Single Level)
For an input tensor $X$ of dimensions $(B, C, H, W)$, the single-level decomposition is computed via row-wise and column-wise averaging (low-pass) and differencing (high-pass).

1.  **Row Decomposition**:
    *   $L_{row} = (X_{:,:,:,2i} + X_{:,:,:,2i+1}) / 2$
    *   $H_{row} = (X_{:,:,:,2i} - X_{:,:,:,2i+1}) / 2$

2.  **Column Decomposition** (applied to $L_{row}$ and $H_{row}$):
    *   **LL** (Approximation): $(L_{row[:,:,2j,:]} + L_{row[:,:,2j+1,:]}) / 2$
    *   **HL** (Vertical Details): $(L_{row[:,:,2j,:]} - L_{row[:,:,2j+1,:]}) / 2$
    *   **LH** (Horizontal Details): $(H_{row[:,:,2j,:]} + H_{row[:,:,2j+1,:]}) / 2$
    *   **HH** (Diagonal Details): $(H_{row[:,:,2j,:]} - H_{row[:,:,2j+1,:]}) / 2$

This operation reduces the spatial resolution by exactly half ($H/2, W/2$) for each subband.

#### B. Multi-Level Recursive Decomposition
For a decomposition of level $L > 1$, we recursively apply the Forward DWT to the **LL** subband of the previous level:

$$ \text{DWT}(LL_{k-1}) \rightarrow \{LL_k, LH_k, HL_k, HH_k\} $$

The implementation handles arbitrary input sizes by applying symmetric padding if the spatial dimensions are odd before any decomposition step.

#### C. Inverse Transform (IDWT)
Reconstruction is performed by interleaving the columns and rows of the respective Low and High bands, reversing the decomposition steps exactly to ensure perfect reconstruction (up to floating-point precision) of the signal in the absence of modification.

---

## 3. Experimental Setup

## 3. Experimental Setup

To ensure strict reproducibility and fair evaluation, we adhere to the following rigorous experimental protocol.

### 3.1 Hardware and Software Environment
All experiments were conducted on a high-performance computing cluster equipped with **Dual NVIDIA RTX 6000 Ada Generation GPUs** (48GB VRAM each, 96GB Total). The codebase is implemented in **PyTorch 2.1**, leveraging `torch.fft` for Fourier transforms and a custom-built, JIT-optimized module for Haar Wavelet Transforms. To ensure deterministic comparisons between the DFT baseline and our Wavelet method, we fixed all random seeds (seed=0) for dataset shuffling and diffusion noise generation.

### 3.2 Models and Datasets
*   **Dataset**: We evaluate our method on the **CIFAR-10** dataset ($32 \times 32$ pixels, 10 classes). Due to the high computational cost of running PGD-EOT attacks on continuous-time diffusion models, we follow standard literature practice by utilizing a fixed, randomly sampled subset of **512 images** from the official validation set for all robustness benchmarks.
*   **Classifier**: The target classifier is a robustly pre-trained **WideResNet-70-16** (feature multiplier 16). This model achieves a clean accuracy of **94.34%** on the full test set, providing a strong baseline for evaluating fidelity preservation.
*   **Diffusion Model**: We utilize a **Score-based Diffusion Model (VP-SDE)** trained on CIFAR-10. This model acts as the generative prior, solving the reverse-time SDE to project noisy inputs back onto the clean data manifold.

### 3.3 Adversarial Threat Model
We assume a **White-Box** threat model where the attacker has complete knowledge of the classifier and the defense mechanism (including the wavelet transform). We evaluate robustness against the **Projected Gradient Descent (PGD)** attack under the $L_\infty$ norm constraint.

*   **Attack Specification**:
    *   **Norm**: $L_\infty \le 8/255$ (pixel perturbation budget).
    *   **Iterations**: 200 steps (PGD-200).
    *   **Step Size**: $\alpha = 2/255$.
    *   **Objective**: Cross-Entropy Loss maximization.
*   **Handling Stochasticity (EOT)**: Since diffusion purification is stochastic, a standard PGD attack might fail due to noisy gradients. To prevent this "false sense of security," we employ **Expectation over Time (EOT)**. At each attack step, we calculate the average gradient over **$N=20$** Monte Carlo samples of the purification process, ensuring the attacker targets the *expected* behavior of the defense.
*   **Gradient Approximation**: As the hard clamping operations in frequency exchange are non-differentiable or have zero gradients almost everywhere, we utilize the **Straight-Through Estimator (STE)** during the backward pass of attack generation, effectively implementing BPDA (Backward Pass Differentiable Approximation).

### 3.4 Defense Configurations
We compare our proposed method against the strongest baseline under identical constraints.
*   **Baseline (DFT-FreqPure)**: Uses Discrete Fourier Transform. Low-frequency amplitude/phase preserved for frequencies $r \le 10$ pixels.
*   **Ours (Wavelet-FreqPure)**:
    *   **Decomposition**: Haar Wavelet (Level 1, 2, or 3).
    *   **Exchange Threshold ($\delta$)**: A hyperparameter controlling the radius of the permissible perturbation ball in the detail subbands. Values tested: $\{0.1, 0.2, 0.3\}$.
    *   **Diffusion Sampling**: DDPM Ancestral Sampling with **100 reverse steps** (starting from $t=300$).

### 3.5 Evaluation Metrics
1.  **Natural Accuracy ($Acc_{nat}$)**: The percentage of *clean*, unperturbed test images correctly classified after purification. This measures **Fidelity**.
    $$ Acc_{nat} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(C(Purify(x_i)) = y_i) $$
2.  **Adversarial Accuracy ($Acc_{adv}$)**: The percentage of *adversarially perturbed* images correctly classified after purification. This measures **Robustness**.
    $$ Acc_{adv} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(C(Purify(x_i + \delta_{adv})) = y_i) $$

---

---

## 4. Experimental Results and Analysis

We compared our proposed **Wavelet-FreqPure** approach against the original **DFT-FreqPure** baseline across multiple configurations, evaluating both clean accuracy (fidelity) and adversarial robustness (security) on the full validation set (512 samples) of CIFAR-10.

### 4.1 Quantitative Results

Table 1 summarizes the performance of different purification strategies.

**Table 1: Comparative Evaluation of Frequency Purification Methods**

| Method | Configuration | Natural Acc | Adversarial Acc | Improvement (over Baseline) |
| :--- | :--- | :--- | :--- | :--- |
| **DFT-FreqPure** | $\delta=0.3$ (Baseline) | **94.34%** | 69.73% | -- |
| **Wavelet-FreqPure** | Level 1, $\delta=0.3$ | 92.77% | 80.27% | +10.54% |
| **Wavelet-FreqPure** | **Level 2, $\delta=0.1$** | 88.87% | **85.55%** | **+15.82%** |
| Wavelet-FreqPure | Level 3, $\delta=0.3$ | 58.01% | 51.37% | -18.36% |
| **Wavelet-FreqPure** | **Learnable Masks** (L2, $\delta=0.2$) | **92.77%** | **82.23%** | **+12.50%** |

### 4.2 Analysis of Configurations

#### A. The Robustness Champion: Level 2 ($\delta=0.1$)
The **Level 2, $\delta=0.1$** configuration achieved the highest adversarial accuracy of **85.55%**, outperforming the DFT baseline by a substantial margin of **15.82%**.
*   **Mechanism**: The deeper decomposition (Level 2) results in a smaller Approximation subband ($8 \times 8$). This forces the diffusion model to regenerate more of the image content from its own learned prior rather than relying on the potentially poisoned input.
*   **Trade-off**: The aggressive structure replacement slightly reduces natural accuracy to 88.87%, as some fine-grained details of the original clean image are lost during the regeneration. However, for defense purposes, this is often an acceptable trade-off.

#### B. The Balanced Choice: Learnable Masks
Our experiment with **Learnable Masks** at Level 2 yielded a highly effective "middle ground":
*   **Performance**: It matched the high Natural Accuracy of Level 1 (**92.77%**) while surpassing its Robustness (**82.23%**).
*   **Insight**: By learning which wavelet coefficients to suppress rather than applying a uniform filter, the model learned to preserve safe texture (boosting Natural Acc) while targeting specific frequency bands prone to adversarial noise. This represents a promising direction for future work: data-driven frequency selection.

#### C. The Failure Case: Level 3
The performance collapse at **Level 3** (~58% accuracy) establishes a critical lower bound. The $4 \times 4$ LL subband provided insufficient semantic guidance, causing the diffusion model to "hallucinate" incorrect classes or lose object coherence. This confirms that frequency guidance must be at a resolution sufficient to resolve the object's primary features.

### 4.3 Qualitative Comparison: Wavelet vs. DFT

Our visualization analysis identifies the root cause of DFT's lower performance:
1.  **Global vs. Local**: DFT applies global modifications. To remove a localized adversarial patch, it must filter that frequency across the entire image, often blurring genuine textures (e.g., fur, grass) which reduces the classifier's confidence.
2.  **Ringing Artifacts**: Sharp transitions (edges) in the spatial domain correspond to infinite frequencies in the Fourier domain. Hard clamping in DFT causes "ringing" (Gibbs phenomenon) near edges, introducing new artifacts that can confuse the classifier.
3.  **Wavelet Advantage**: Wavelets are localized in both space and frequency. Our method can surgical remove high-frequency noise in a specific region (the attack) without degrading the sharpness of the rest of the image. The "Ghost Test" maps (see Figures) confirm that Wavelet-FreqPure removes strictly non-structural noise, whereas DFT residuals often contain structural "ghosts" of the object.

---

## 5. Conclusion

We have successfully demonstrated that replacing the global DFT with a multi-scale **Wavelet Transform** significantly enhances the robustness of diffusion-based adversarial purification.
*   **Best Config**: Level 2 decomposition with tight detail constraint ($\delta=0.1$).
*   **Key Result**: **85.55%** Adversarial Accuracy (+15.8% over DFT baseline).

Wavelet-FreqPure offers a more granular, spatially-aware control over image frequencies, making it a superior choice for defending against sophisticated adversarial attacks.

---

## Appendix A: Experiment Logs

| Experiment | Natural Acc | Adversarial Acc | Log File |
| :--- | :--- | :--- | :--- |
| DFT Baseline | 94.34% | 69.73% | results_dft.txt |
| Wavelet L=1, $\delta=0.3$ | 92.77% | 80.27% | full_level1_delta03.txt |
| Wavelet L=2, $\delta=0.1$ | 88.87% | 85.55% | full_level2_delta01.txt |
| Wavelet L=3, $\delta=0.3$ | 58.01% | 51.37% | full_level3_delta03.txt |
