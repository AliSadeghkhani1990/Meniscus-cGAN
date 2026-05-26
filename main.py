import os
import sys
import argparse
from pathlib import Path
import datetime
# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import numpy as np
# Import functions from your custom modules
from base_code.data_processing import (
    load_images_meniscus,  # Meniscus-specific loader
    extract_and_save_patch_info_meniscus_3D,  # Meniscus extraction
    categorize_patches_meniscus,  # Multi-condition categorization
    ImageType, read_patch, load_image_stacks_meniscus
)
from base_code.data_augmentation import balance_and_analyze_dataset
from base_code.models import generator_model, discriminator_model
from base_code.training import train
from base_code.conditions import create_default_condition_manager  # Meniscus condition manager
from base_code.utils import (
    analyze_patch_info, display_random_patches_with_properties,
    draw_heatmaps, plot_generated_images_inference_enhanced,
    evaluate_generator_accuracy, print_data_ranges,
    save_inference_metadata, create_results_directory,
    save_training_images_by_class, create_models_subdirectory,
    create_plot_directories, plot_condition_distributions
)
from tensorflow.keras.models import load_model


# ============================================================================
# GPU CONFIGURATION FOR HPC COMPATIBILITY
# ============================================================================

def setup_gpu():
    """Configure GPU settings - works on both PC and AIRE HPC"""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        print(f"\n✓ Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"    GPU {i}: {gpu}")
            try:
                # Enable memory growth to prevent OOM errors
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"    Warning: {e}")
        return len(gpus)
    else:
        print("\n⚠ No GPUs found, using CPU (training will be slow)")
        return 0


# ============ MENISCUS CONFIGURATION ============
# Code version
code_version = "meniscus_v1"

# Image configuration (meniscus is grayscale only)
image_type = ImageType.GRAYSCALE
n_channels = 32

plot_dirs = None

# Patch extraction parameters
patch_size = 512  # Standard size for paper
min_tissue_ratio = 0.95  # Require 95% tissue content
num_patches_total = 15000  # Total patches to extract (matching HPC script)
overlap = 0.50  # Overlap for grid extraction

# Conditioning parameters
n_classes_per_condition = {
    'fvf': 10,
    'alignment': 10,
    'thickness': 4,
    'complexity': 6
} # Classes per condition for categorization
desired_patches_per_class = 160  # Target for balancing
min_patches_per_class = 15  # Minimum to avoid empty classes

# Training parameters
n_batch = 16
epochs = 100
latent_dim = 100
d_slices = 32
learning_rate = 0.0002
beta_1 = 0.5
saving_step = 5
save_interval = True
last_epochs_to_save = 20
bat_per_epo = 300

# Directory configuration
directory_path = 'data/meniscus_data'  # Generic path for GitHub
reload = False
model_names = ['G.h5', 'D.h5']
local_model_path_loading = None  # Set this if you want to load existing models

def load_or_create_models(n_channels, patch_size, condition_manager, reload_flag, model_path):
    """Load existing models or create new ones"""
    if reload_flag and model_path:
        print(f"\n🔍 Attempting to load models from: {model_path}")
        import glob, re

        g_files = glob.glob(os.path.join(model_path, 'G*.h5'))
        d_files = glob.glob(os.path.join(model_path, 'D*.h5'))

        if not g_files or not d_files:
            print(f"⚠️ No model files found in {model_path}")
            return create_new_models(n_channels, patch_size, condition_manager), 0

        g_files.sort()
        d_files.sort()
        latest_g, latest_d = g_files[-1], d_files[-1]

        print(f"✅ Found: {os.path.basename(latest_g)}, {os.path.basename(latest_d)}")

        epoch_match = re.search(r'epoch_(\d+)', os.path.basename(latest_g))
        start_epoch = int(epoch_match.group(1)) if epoch_match else 0

        try:
            g_model = load_model(latest_g, compile=False)
            d_model = load_model(latest_d, compile=False)
            print(f"✅ Loaded models from epoch {start_epoch}, continuing from {start_epoch + 1}")
            return (g_model, d_model), start_epoch
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return create_new_models(n_channels, patch_size, condition_manager), 0
    else:
        print("\n🚀 Creating new models")
        return create_new_models(n_channels, patch_size, condition_manager), 0


def create_new_models(n_channels, patch_size, condition_manager):
    """Create new models for training"""
    print("Creating new models...")
    g_model = generator_model(latent_dim, n_channels, patch_size, condition_manager)
    d_model = discriminator_model((patch_size, patch_size, d_slices), condition_manager)
    return g_model, d_model


# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================

def parse_args():
    """Parse command line arguments for flexible configuration"""
    parser = argparse.ArgumentParser(
        description='Meniscus Tissue cGAN Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ========== PATH ARGUMENTS ==========
    parser.add_argument('--data_dir', type=str,
                        default='data/meniscus_data',
                        help='Path to meniscus data directory')
    parser.add_argument('--results_base_dir', type=str,
                        default='results',
                        help='Base directory for results')

    # ========== TRAINING PARAMETERS ==========
    parser.add_argument('--patch_size', type=int, default=512,
                        help='Size of image patches')
    parser.add_argument('--num_patches_total', type=int, default=15000,
                        help='Total number of patches to extract')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--n_channels', type=int, default=32,
                        help='Number of output channels (depth slices)')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='Dimension of latent noise vector')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                        help='Learning rate for optimizers')

    # ========== CONDITION PARAMETERS ==========
    parser.add_argument('--enable_fvf', action='store_true', default=True,
                        help='Enable Fiber Volume Fraction condition')
    parser.add_argument('--enable_alignment', action='store_true', default=True,
                        help='Enable Fiber Alignment condition')
    parser.add_argument('--enable_thickness', action='store_true', default=False,
                        help='Enable Fiber Thickness condition')
    parser.add_argument('--enable_complexity', action='store_true', default=False,
                        help='Enable Texture Complexity condition')

    # ========== OTHER PARAMETERS ==========
    parser.add_argument('--min_tissue_ratio', type=float, default=0.95,
                        help='Minimum tissue content in patches')
    parser.add_argument('--desired_patches_per_class', type=int, default=160,
                        help='Target patches per class for balancing')
    parser.add_argument('--min_patches_per_class', type=int, default=20,
                        help='Minimum patches required per class')
    parser.add_argument('--reload', action='store_true',
                        help='Load existing models')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to existing models to load')
    parser.add_argument('--test_run', action='store_true',
                        help='Quick test run with reduced parameters')

    return parser.parse_args()


def main():
    # ========== PARSE ARGUMENTS ==========
    args = parse_args()

    # Apply test_run overrides if specified
    if args.test_run:
        print("\n⚠️ TEST RUN MODE - Using reduced parameters")
        args.epochs = 2
        args.num_patches_total = 100
        args.batch_size = 4
        args.min_patches_per_class = 5
        args.desired_patches_per_class = 10

        # ========== SETUP GPU ==========
    print("=" * 60)
    print("Starting Meniscus Tissue cGAN Training Process")
    print("=" * 60)

    num_gpus = setup_gpu()

    # ========== GLOBAL VARIABLES (from args) ==========
    global plot_dirs, code_version, image_type, n_channels
    global patch_size, min_tissue_ratio, num_patches_total, overlap
    global n_classes_per_condition, desired_patches_per_class, min_patches_per_class
    global n_batch, epochs, latent_dim, learning_rate, beta_1
    global saving_step, save_interval, last_epochs_to_save, bat_per_epo
    global directory_path, reload, model_names, local_model_path_loading

    # Assign from args
    code_version = "meniscus_v1"
    image_type = ImageType.GRAYSCALE
    n_channels = args.n_channels
    patch_size = args.patch_size
    min_tissue_ratio = args.min_tissue_ratio
    num_patches_total = args.num_patches_total
    overlap = 0.50
    n_classes_per_condition = {'fvf': 10, 'alignment': 10, 'thickness': 4, 'complexity': 6}
    desired_patches_per_class = args.desired_patches_per_class
    min_patches_per_class = args.min_patches_per_class
    n_batch = args.batch_size
    epochs = args.epochs
    latent_dim = args.latent_dim
    learning_rate = args.learning_rate
    beta_1 = 0.5
    saving_step = 5
    save_interval = True
    last_epochs_to_save = 20
    bat_per_epo = 300
    directory_path = args.data_dir
    reload = args.reload
    model_names = ['G.h5', 'D.h5']
    local_model_path_loading = args.model_path

    # ========== PRINT CONFIGURATION ==========
    print("\n📋 Configuration:")
    print("-" * 60)
    print(f"  Data directory: {directory_path}")
    print(f"  Results base: {args.results_base_dir}")
    print(f"  Patch size: {patch_size}")
    print(f"  Total patches: {num_patches_total}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {n_batch}")
    print(f"  Channels: {n_channels}")
    print(f"  GPUs available: {num_gpus}")
    print("-" * 60)

    # ========== SETUP CONDITION MANAGER ==========
    print("\nSetting up condition manager with meniscus-specific conditions...")
    condition_manager = create_default_condition_manager()

    # Apply condition flags from arguments
    if not args.enable_fvf:
        condition_manager.disable_condition('fvf')
    if not args.enable_alignment:
        condition_manager.disable_condition('alignment')
    if not args.enable_thickness:
        condition_manager.disable_condition('thickness')
    if not args.enable_complexity:
        condition_manager.disable_condition('complexity')

    # List active conditions
    active_conditions = [cond.name for cond in condition_manager.active_conditions]
    if not active_conditions:
        print("\n⚠️ UNCONDITIONAL MODE: No conditions enabled")
        print("   - Training as standard GAN (no controllability)")
        print("   - All patches will be used without class balancing\n")
    else:
        print("\n✓ Active Conditions:")
        for cond in active_conditions:
            print(f"  - {cond}")

    # ========== CREATE RESULTS DIRECTORY ==========
    results_directory = create_results_directory(
        patch_size, image_type, condition_manager, code_version,
        base_dir=args.results_base_dir
    )
    print(f"\n📁 Results will be saved to: {results_directory}")

    # Create plot directories
    plot_dirs = create_plot_directories(results_directory)

    # Create models subdirectory
    model_path_saving = create_models_subdirectory(results_directory)
    print(f"💾 Models will be saved to: {model_path_saving}")

    # ========== LOAD DATA ==========
    print("\n📂 Loading meniscus z-stack volumes...")
    original_volumes = load_image_stacks_meniscus(directory_path, n_slices=None)
    print(f"✓ Loaded {len(original_volumes)} 3D volumes")
    print(f"  Volume shape: {original_volumes[0].shape}")

    original_images = original_volumes

    # ========== EXTRACT PATCHES ==========
    print("\n🔬 Extracting and analyzing meniscus patches...")
    patch_info = extract_and_save_patch_info_meniscus_3D(
        original_images,
        patch_size,
        num_patches_total,
        condition_manager,
        n_slices=args.n_channels,  # This will be 16
        min_tissue_ratio=min_tissue_ratio,
        overlap=overlap
    )

    # ========== ANALYZE PATCH DISTRIBUTION ==========
    print("\n" + "=" * 60)
    print("PATCH SOURCE DISTRIBUTION ANALYSIS")
    print("=" * 60)

    image_distribution = {}
    for patch in patch_info:
        img_id = patch['p']
        image_distribution[img_id] = image_distribution.get(img_id, 0) + 1

    print(f"\nTotal images available: {len(original_images)}")
    print(f"Images actually used: {len(image_distribution)}")
    print(f"Total patches extracted: {len(patch_info)}\n")

    print("Patches per image:")
    for img_id in sorted(image_distribution.keys()):
        count = image_distribution[img_id]
        percentage = (count / len(patch_info)) * 100
        bar = "█" * int(percentage / 2)
        print(f"  Image {img_id:2d}: {count:4d} patches ({percentage:5.1f}%) {bar}")

    counts = list(image_distribution.values())
    print(f"\nStatistics:")
    print(f"  Mean patches per used image: {np.mean(counts):.1f}")
    print(f"  Std deviation: {np.std(counts):.1f}")
    print(f"  Min: {min(counts)}, Max: {max(counts)}")
    print("=" * 60 + "\n")

    # ========== CATEGORIZE PATCHES ==========
    print("\n📊 Categorizing patches based on conditions...")
    patch_info = categorize_patches_meniscus(
        patch_info,
        n_classes_per_condition,
        condition_manager
    )

    # ========== BALANCE DATASET ==========
    print("\n⚖️ Balancing dataset...")
    balanced_patch_info, repetition_percentages, empty_classes, class_counts, non_empty_classes = \
        balance_and_analyze_dataset(
            patch_info,
            desired_patches_per_class,
            min_patches_per_class,
            condition_manager
        )

    # ========== SAVE TRAINING IMAGES ==========
    print("\n💾 Saving training images by class...")
    save_training_images_by_class(
        original_images,
        balanced_patch_info,
        patch_size,
        results_directory,
        condition_manager,
        n_channels
    )

    # ========== DATASET SUMMARY ==========
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total images loaded: {len(original_images)}")
    print(f"Total patches extracted: {len(patch_info)}")
    print(f"Patches after balancing: {len(balanced_patch_info)}")
    print(f"Class distribution: {class_counts}")
    print("=" * 60 + "\n")

    # ========== ANALYZE DATA ==========
    print("📈 Analyzing data ranges for generation...")
    print_data_ranges(balanced_patch_info, condition_manager)

    print("\n📊 Analyzing patch statistics...")
    analyze_patch_info(patch_info, condition_manager)

    print("\n🖼️ Visualizing random patches...")
    display_random_patches_with_properties(
        original_images,
        patch_info,
        patch_size,
        None,
        n_channels,
        condition_manager,
        n_slices=n_channels
    )

    print("\n📊 Generating distribution visualizations...")
    draw_heatmaps(patch_info, balanced_patch_info, condition_manager)

    # ========== LOAD OR CREATE MODELS ==========
    print("\n🤖 Initializing models...")
    (g_model, d_model), start_epoch = load_or_create_models(
    n_channels, patch_size, condition_manager, args.reload, args.model_path)

    print("\n📝 Generator Summary:")
    g_model.summary()
    print("\n📝 Discriminator Summary:")
    d_model.summary()

    # ========== PREPARE DATASET ==========
    dataset = [original_images]

    # ========== TRAIN ==========
    print("\n" + "=" * 60)
    print("🚀 Starting model training...")
    print("=" * 60)
    train(
        g_model, d_model, dataset, balanced_patch_info, latent_dim, epochs, n_batch,
        saving_step, non_empty_classes, patch_size, model_path_saving,
        n_classes_per_condition, n_classes_per_condition,
        save_interval, last_epochs_to_save, bat_per_epo, n_channels, condition_manager,
        n_channels, d_slices=d_slices, start_epoch=start_epoch
    )

    # ========== EVALUATE ==========
    print("\n📊 Evaluating generator performance...")
    evaluation_results = evaluate_generator_accuracy(
        g_model, latent_dim, balanced_patch_info,
        n_classes_per_condition, None, image_type,
        None, condition_manager
    )

    if evaluation_results:
        print(f"\n📈 Evaluation Results:")
        for key, value in evaluation_results.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")

    # ========== GENERATE INFERENCE IMAGES ==========
    print("\n🎨 Generating inference visualizations...")
    plot_generated_images_inference_enhanced(
        g_model, latent_dim, n_channels, condition_manager,
        model_path_saving, patch_size, balanced_patch_info,
        non_empty_classes
    )

    # ========== SAVE METADATA ==========
    print("\n💾 Saving metadata for future inference...")
    save_inference_metadata(
        model_path=model_path_saving,
        n_channels=n_channels,
        patch_size=patch_size,
        latent_dim=latent_dim,
        balanced_patch_info=balanced_patch_info,
        condition_manager=condition_manager,
        image_type=image_type,
        porosity_method=None,
        porosity_threshold_value=None
    )

    print("\n" + "=" * 60)
    print("✅ Meniscus cGAN Training Process Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()