# 3D Conditional GAN for Meniscus Tissue Generation

This repository contains the official implementation of the 3D Conditional Generative Adversarial Network (cGAN) for synthesizing realistic meniscus tissue volumes, as presented in the paper: **"Generative Modeling of 3D Meniscus Tissue Microstructure using Conditional GANs"** (Journal: Biomechanics and Modeling in Mechanobiology - BMMB).

## Overview

The model is designed to generate 3D volumetric patches ($512 \times 512 \times 32$ pixels) of meniscus tissue, controllable via biological and structural parameters. While the code supports four conditioning factors, the journal publication focuses on:
1. **Fiber Volume Fraction (FVF)**
2. **Fiber Alignment**

The architecture leverages 2D-based convolutions that process the depth dimension as channels, ensuring inter-slice continuity and realistic volumetric features.

## Project Structure

```text
.
├── base_code/           # Core GAN modules
│   ├── conditions.py    # Condition manager and extraction functions
│   ├── data_augmentation.py
│   ├── data_processing.py # 3D volume loading and patch extraction
│   ├── models.py        # Generator and Discriminator architectures
│   ├── training.py      # Training loop and loss functions
│   └── utils.py         # Visualization and evaluation tools
├── data/                # Place your training data here
├── results/             # Output directory for models and plots
├── main.py              # Main entry point for training
├── submit_aire.sh       # SLURM submission script for HPC (Aire)
├── requirements.txt     # Python dependencies
└── README.md
```

## Requirements

The code is compatible with both local workstations (Windows/Linux) and HPC systems (SLURM).

### Environment Setup
1. Create a conda environment:
   ```bash
   conda create -n meniscus_cgan python=3.9
   conda activate meniscus_cgan
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Preparation
Organize your meniscus z-stack volumes in the `data/meniscus_data/` directory. Each sample should be a subdirectory containing sequential `.tif` or `.png` slices, or all slices can be in a single directory if using the flat structure.

### 2. Local Training
Run the training script with default parameters (aligned with the paper):
```bash
python main.py --data_dir ./data/meniscus_data --results_base_dir ./results
```

### 3. HPC Training (Aire)
To run on the University of Leeds Aire HPC system:
```bash
sbatch submit_aire.sh
```

## Key Configuration (Paper Parameters)
- **Patch Size**: $512 \times 512$
- **Depth Slices**: 32
- **Latent Dimension**: 100
- **Epochs**: 100
- **Batch Size**: 16
- **Active Conditions**: FVF and Fiber Alignment

## Citation
If you use this code in your research, please cite our paper:
*(Citation details will be updated upon publication)*

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
