#!/bin/bash
#SBATCH --job-name=meniscus-cgan-3d
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=cgan_3d_%j.out
#SBATCH --error=cgan_3d_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@leeds.ac.uk

echo "========================================="
echo "Meniscus 3D cGAN Training"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "========================================="

# Show GPU info
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# Load environment
module load miniforge
conda activate cgan_gpu

# Verify environment
echo "Python: $(which python)"
echo "TensorFlow version check:"
python -c "import tensorflow as tf; print(f'TF: {tf.__version__}, GPUs: {len(tf.config.list_physical_devices(\"GPU\"))}')"
echo ""

# Navigate to code directory
cd ${SLURM_SUBMIT_DIR:-.}

echo "Starting training with paper parameters..."

# Set data and results directories (use scratch for performance on HPC)
DATA_DIR=${1:-"./data/meniscus_data"}
RESULTS_DIR=${2:-"./results"}

python main.py \
    --data_dir "$DATA_DIR" \
    --results_base_dir "$RESULTS_DIR" \
    --patch_size 512 \
    --num_patches_total 15000 \
    --epochs 100 \
    --batch_size 16 \
    --n_channels 32 \
    --latent_dim 100 \
    --learning_rate 0.0002 \
    --enable_fvf \
    --enable_alignment \
    --desired_patches_per_class 160 \
    --min_patches_per_class 15 \
    --min_tissue_ratio 0.95 \



echo ""
echo "========================================="
echo "Training completed: $(date)"
echo "========================================="

# Check if output files were created
echo "Checking outputs in $RESULTS_DIR..."
find "$RESULTS_DIR" -name "*.h5" | head -5

echo "Job finished!"