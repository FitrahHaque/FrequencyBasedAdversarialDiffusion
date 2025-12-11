# FreqPure Setup & Inference Guide

This guide explains how to set up the environment and run the image purification inference on CIFAR-10.

## Prerequisites

- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 6000 Ada Generation)
- **CUDA**: CUDA 12.x toolkit installed (verify with `nvcc --version`)
- **Python**: Python 3.10+
- **OS**: Linux (tested on Ubuntu)

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: You may need to upgrade PyTorch for CUDA 12 compatibility:

```bash
pip install torch torchvision xformers --upgrade
```

## Step 2: Download Pretrained Models

### Diffusion Model (CIFAR-10)

Download the Score SDE checkpoint from [DiffPure](https://github.com/NVlabs/DiffPure):

```bash
# Install gdown if not available
pip install gdown

# Create directory and download
mkdir -p ./pretrained/guided_diffusion
gdown 16_-Ahc6ImZV5ClUc0vM5Iivf8OJ1VSif -O ./pretrained/guided_diffusion/checkpoint_8.pth
```

### Classifier (WideResNet-70-16)

Download from DiffPure's pretrained classifiers:

```bash
# Create directory and download
mkdir -p ./models/cifar10/Linf
gdown --folder https://drive.google.com/drive/folders/1OeuFx2r26xeHncs8bGuqgY6ns_N77Avi -O ./models/cifar10/Linf/

# Move file to expected location
mv ./models/cifar10/Linf/wresnet-76-10/weights-best.pt ./models/cifar10/Linf/
rm -rf ./models/cifar10/Linf/wresnet-76-10
```

## Step 3: Download Dataset

CIFAR-10 will be automatically downloaded on first run.

Alternatively, create the dataset directory:

```bash
mkdir -p ./dataset
```

## Step 4: Run Inference

### Basic Command

For a quick test with 50 samples on 2 GPUs:

```bash
PATH=/usr/local/cuda-12.6/bin:$PATH \
CUDA_HOME=/usr/local/cuda-12.6 \
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 ddp_test.py \
    --amplitude_cut_range 10 \
    --phase_cut_range 10 \
    --delta 0.3 \
    --def_max_timesteps 1000 \
    --def_num_denoising_steps 100 \
    --att_max_timesteps 1000 \
    --att_num_denoising_steps 1 \
    --num_ensemble_runs 10 \
    --num_samples 50 \
    --batch_size 25
```

### Full Evaluation (512 samples)

For the full evaluation:

```bash
PATH=/usr/local/cuda-12.6/bin:$PATH \
CUDA_HOME=/usr/local/cuda-12.6 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 ddp_test.py \
    --amplitude_cut_range 10 \
    --phase_cut_range 10 \
    --delta 0.3 \
    --def_max_timesteps 1000 \
    --def_num_denoising_steps 100 \
    --att_max_timesteps 1000 \
    --att_num_denoising_steps 1 \
    --num_ensemble_runs 10 \
    --num_samples 512 \
    --batch_size 64
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num_samples` | Number of test samples to use | 512 |
| `--batch_size` | Batch size per GPU | 64 |
| `--amplitude_cut_range` | Frequency amplitude cut range | 10 |
| `--phase_cut_range` | Frequency phase cut range | 10 |
| `--delta` | Purification delta parameter | 0.3 |
| `--num_ensemble_runs` | Number of ensemble runs for prediction | 10 |
| `--n_iter` | Number of PGD attack iterations | 200 |
| `--eot` | EOT samples for attack | 20 |
| `--attack_method` | Attack type: pgd, pgd_l2, bpda, aa, aa_l2 | pgd |

## Output

After running, results are saved to:

| Directory | Content |
|-----------|---------|
| `./original/` | Original clean images |
| `./adv/` | Adversarial images after PGD attack |
| `./pure_images/` | Purified images (with correct/false subdirectories) |
| `./output_logs.txt` | Detailed logs |

## Expected Results

On CIFAR-10 with default parameters:

- **Natural Accuracy (acc_nat)**: ~92%
- **Adversarial Accuracy (acc_adv)**: ~66%

## Troubleshooting

### CUDA Architecture Error

If you see `nvcc fatal: Unsupported gpu architecture 'compute_89'`, set the correct CUDA path:

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.6
```

### Torch Extensions Cache

If CUDA extensions fail to compile, clear the cache:

```bash
rm -rf ~/.cache/torch_extensions/
```

### Dataset Download Issue

If dataset download fails with DDP, download manually first:

```bash
python -c "import torchvision; torchvision.datasets.CIFAR10('./dataset/cifar10', download=True, train=False)"
```

## Reference

Original paper and code: [DiffPure](https://github.com/NVlabs/DiffPure)
