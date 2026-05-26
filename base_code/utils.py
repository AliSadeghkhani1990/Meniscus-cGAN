import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import datetime
import random
import json
import pickle
from pathlib import Path
import datetime
from base_code.data_processing import read_patch


def create_plot_directories(results_directory):
    """Create directories for saving plots."""
    plot_dirs = {
        'analysis_plots': os.path.join(results_directory, 'analysis_plots'),
        'training_plots': os.path.join(results_directory, 'training_plots'),
        'evaluation_plots': os.path.join(results_directory, 'evaluation_plots')
    }
    for dir_path in plot_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    return plot_dirs


def plot_condition_distributions(patch_info, condition_manager, save_path):
    """Plot distribution histograms for all active conditions."""
    active_conditions = [cond.name for cond in condition_manager.active_conditions]

    if not active_conditions:
        return

    fig, axes = plt.subplots(1, len(active_conditions), figsize=(5 * len(active_conditions), 4))
    if len(active_conditions) == 1:
        axes = [axes]

    for idx, condition_name in enumerate(active_conditions):
        values = [p[f'{condition_name}_value'] for p in patch_info]

        axes[idx].hist(values, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[idx].set_title(f'{condition_name.upper()} Distribution')
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Count')
        axes[idx].axvline(np.mean(values), color='red', linestyle='--',
                          label=f'Mean: {np.mean(values):.3f}')
        axes[idx].axvline(np.median(values), color='green', linestyle='--',
                          label=f'Median: {np.median(values):.3f}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle('Meniscus Condition Distributions', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_generated_images_enhanced(generator, epoch, latent_dim, n_channels, condition_manager,
                                   balanced_patch_info, fixed_noise=None, fixed_conditions=None):
    """
    Generate and visualize images/volumes with different condition values.

    For 3D mode (n_channels=10):
    - Generates volumes and shows 5 depth slices per sample
    - Displays slices: 0, 2, 5, 7, 9 (evenly spaced through depth)
    - Grid: 5 samples × 5 slices = 25 subplots

    For 2D mode (n_channels=1):
    - Shows 10 samples in 2×5 grid
    """
    print(f"\nGenerating {'3D volumes' if n_channels > 1 else '2D images'} for epoch {epoch}")

    # Get active conditions
    active_conditions = [cond.name for cond in condition_manager.active_conditions]

    # Set number of images to generate
    n_images = 10 if n_channels == 1 else 5  # 10 for 2D, 5 for 3D

    # Create fixed noise if not provided
    if fixed_noise is None:
        fixed_noise = np.random.randn(n_images, latent_dim)

    # ========== HANDLE UNCONDITIONAL CASE ==========
    if not active_conditions:
        all_images = []
        for i in range(n_images):
            noise = fixed_noise[i].reshape(1, -1)
            img = generator.predict([noise], verbose=0)
            all_images.append((img[0] + 1) / 2)

        # Display based on mode
        if n_channels == 1:
            # 2D unconditional
            fig, axes = plt.subplots(2, 5, figsize=(15, 7))
            axes = axes.flatten()
            for i, img in enumerate(all_images):
                axes[i].imshow(img[:, :, 0], cmap='gray')
                axes[i].axis('off')
                axes[i].set_title(f'Sample {i + 1}', fontsize=8)
        else:
            # 3D unconditional - show slices
            slice_indices = [0, 2, 5, 7, 9]
            fig, axes = plt.subplots(n_images, len(slice_indices),
                                     figsize=(len(slice_indices) * 2.5, n_images * 2.5))
            if n_images == 1:
                axes = axes.reshape(1, -1)

            for sample_idx, volume in enumerate(all_images):
                for slice_idx, depth_idx in enumerate(slice_indices):
                    ax = axes[sample_idx, slice_idx]
                    ax.imshow(volume[:, :, depth_idx], cmap='gray', vmin=0, vmax=1)
                    ax.axis('off')
                    if sample_idx == 0:
                        ax.set_title(f'Slice {depth_idx}', fontsize=10)

        plt.suptitle(f'Unconditional Samples at Epoch {epoch}', fontsize=14)
        plt.tight_layout()

        try:
            from __main__ import plot_dirs
            plt.savefig(os.path.join(plot_dirs['training_plots'], f'epoch_{epoch:03d}.png'),
                        dpi=150, bbox_inches='tight')
        except:
            plt.savefig(f'generated_epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')

        plt.close()
        return fixed_noise, None

    # ========== CONDITIONAL CASE ==========
    # Create condition values that vary across the range
    all_inputs = []
    all_labels = []

    for i in range(n_images):
        noise = fixed_noise[i].reshape(1, -1)
        generator_inputs = [noise]
        input_labels = {}

        # For each active condition, sample from the dataset distribution
        for condition in condition_manager.active_conditions:
            # Get all values for this condition
            values = [p[f'{condition.name}_value'] for p in balanced_patch_info]

            # For visualization, use evenly spaced values
            min_val, max_val = min(values), max(values)
            value = min_val + (max_val - min_val) * i / (n_images - 1)

            generator_inputs.append(np.array([[value]]))
            input_labels[condition.name] = value

        all_inputs.append(generator_inputs)
        all_labels.append(input_labels)

    # Generate images/volumes
    all_outputs = []
    for inputs in all_inputs:
        output = generator.predict(inputs, verbose=0)
        all_outputs.append((output[0] + 1) / 2)  # Convert from [-1,1] to [0,1]

    # ========== DISPLAY BASED ON MODE ==========
    if n_channels == 1:
        # 2D MODE
        fig, axes = plt.subplots(2, 5, figsize=(15, 7))
        axes = axes.flatten()

        for i, (img, labels) in enumerate(zip(all_outputs, all_labels)):
            axes[i].imshow(img[:, :, 0], cmap='gray')
            axes[i].axis('off')

            # Create title with condition values
            title_parts = []
            for cond_name, value in labels.items():
                title_parts.append(f"{cond_name[:3]}: {value:.2f}")
            title = '\n'.join(title_parts[:2])
            axes[i].set_title(title, fontsize=8)

        plt.suptitle(f'Generated Images at Epoch {epoch}', fontsize=14)

    else:
        # 3D MODE - Show multiple slices per sample
        slice_indices = [0, 2, 5, 7, 9]  # Show 5 out of 10 slices
        n_slices_to_show = len(slice_indices)

        fig, axes = plt.subplots(n_images, n_slices_to_show,
                                 figsize=(n_slices_to_show * 2.5, n_images * 2.5))

        if n_images == 1:
            axes = axes.reshape(1, -1)

        for sample_idx, (volume, labels) in enumerate(zip(all_outputs, all_labels)):
            for slice_idx, depth_idx in enumerate(slice_indices):
                ax = axes[sample_idx, slice_idx]

                # Display slice
                ax.imshow(volume[:, :, depth_idx], cmap='gray', vmin=0, vmax=1)
                ax.axis('off')

                # Add titles
                if sample_idx == 0:
                    # Top row: show slice number
                    ax.set_title(f'Slice {depth_idx}', fontsize=10)

                if slice_idx == 0:
                    # Left column: show condition values
                    title_parts = []
                    for cond_name, value in labels.items():
                        title_parts.append(f"{cond_name[:3]}: {value:.2f}")
                    title = '\n'.join(title_parts[:2])
                    ax.set_ylabel(title, fontsize=9, rotation=0, ha='right', va='center')

        plt.suptitle(f'Generated 3D Volumes at Epoch {epoch} (5 slices per sample)', fontsize=14)

    plt.tight_layout()

    # Save plot
    try:
        from __main__ import plot_dirs
        save_path = os.path.join(plot_dirs['training_plots'], f'epoch_{epoch:03d}.png')
    except:
        save_path = f'generated_epoch_{epoch:03d}.png'

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return fixed_noise, fixed_conditions


def analyze_patch_info(patch_info, condition_manager):
    """Analyze patch information based on active conditions."""
    analysis_results = {}
    active_conditions = [cond.name for cond in condition_manager.active_conditions]

    print("\nCondition Statistics:")
    for condition in condition_manager.active_conditions:
        if f'{condition.name}_value' in patch_info[0]:
            values = [patch[f'{condition.name}_value'] for patch in patch_info]
            analysis_results[condition.name] = {
                'values': values,
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }

            print(f"\n{condition.name.upper()}:")
            print(f"  Mean: {analysis_results[condition.name]['mean']:.3f}")
            print(f"  Std:  {analysis_results[condition.name]['std']:.3f}")
            print(f"  Range: [{analysis_results[condition.name]['min']:.3f}, "
                  f"{analysis_results[condition.name]['max']:.3f}]")

            # Create histogram
            plt.figure(figsize=(8, 5))
            plt.hist(values, bins=30, edgecolor='black', alpha=0.7)
            plt.title(f'{condition.name.upper()} Distribution')
            plt.xlabel(f'{condition.name} Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)

            try:
                from __main__ import plot_dirs
                plt.savefig(os.path.join(plot_dirs['analysis_plots'], f'{condition.name}_distribution.png'),
                            dpi=300, bbox_inches='tight')
            except:
                plt.savefig(f'{condition.name}_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()

    return analysis_results


def display_random_patches_with_properties(original_images, patch_info, patch_size, unet_model,
                                           n_channels, condition_manager, num_patches=10, n_slices=16):
    """Display random patches with their condition values (supports 2D and 3D)."""
    random_indices = random.sample(range(len(patch_info)), min(num_patches, len(patch_info)))

    # Detect if we're working with 3D volumes
    is_3d = len(original_images[0].shape) == 3 and original_images[0].shape[2] > 1

    if is_3d:
        from base_code.data_processing import read_patch_3D
        middle_slice_idx = n_slices // 2
    else:
        from base_code.data_processing import read_patch

    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()

    for i, idx in enumerate(random_indices):
        patch_data = patch_info[idx]

        if is_3d:
            # Read 3D patch and extract middle slice
            patch_3d = read_patch_3D(original_images, patch_data, patch_size, n_slices)
            patch = patch_3d[:, :, middle_slice_idx]  # Extract middle slice for display
        else:
            # Read 2D patch
            patch = read_patch(original_images, patch_data, patch_size)

        # Display patch
        if is_3d or n_channels == 1:
            # Display as grayscale
            if len(patch.shape) == 3:
                axes[i].imshow(patch[:, :, 0] if patch.shape[2] == 1 else patch, cmap='gray')
            else:
                axes[i].imshow(patch, cmap='gray')
        else:
            # Display as color (n_channels == 3)
            axes[i].imshow(patch)

        axes[i].axis('off')

        # Build title with condition values
        title_parts = []
        for condition in condition_manager.active_conditions:
            if f'{condition.name}_value' in patch_data:
                value = patch_data[f'{condition.name}_value']
                title_parts.append(f"{condition.name}: {value:.3f}")

        title = '\n'.join(title_parts)
        axes[i].set_title(title, fontsize=8)

    title_suffix = " (showing middle slice)" if is_3d else ""
    plt.suptitle(f'Random Patches with Condition Values{title_suffix}', fontsize=14)
    plt.tight_layout()

    try:
        from __main__ import plot_dirs
        plt.savefig(os.path.join(plot_dirs['analysis_plots'], 'random_patches_with_conditions.png'),
                    dpi=300, bbox_inches='tight')
    except:
        plt.savefig('random_patches_with_conditions.png', dpi=300, bbox_inches='tight')

    plt.close()


def draw_heatmaps(original_patch_info, balanced_patch_info, condition_manager):
    """Draw distribution visualizations for conditions."""
    active_conditions = [cond.name for cond in condition_manager.active_conditions]

    if not active_conditions:
        print("No conditions active - skipping heatmap visualization")
        return

    # For each condition, show distribution before and after balancing
    n_conditions = len(active_conditions)
    fig, axes = plt.subplots(n_conditions, 2, figsize=(12, 4 * n_conditions))

    if n_conditions == 1:
        axes = axes.reshape(1, -1)

    for idx, condition in enumerate(condition_manager.active_conditions):
        if f'{condition.name}_value' in original_patch_info[0]:
            # Original distribution
            orig_values = [p[f'{condition.name}_value'] for p in original_patch_info]
            axes[idx, 0].hist(orig_values, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            axes[idx, 0].set_title(f'{condition.name.upper()} - Original Distribution')
            axes[idx, 0].set_xlabel('Value')
            axes[idx, 0].set_ylabel('Count')

            # Balanced distribution
            balanced_values = [p[f'{condition.name}_value'] for p in balanced_patch_info]
            axes[idx, 1].hist(balanced_values, bins=30, edgecolor='black', alpha=0.7, color='darkgreen')
            axes[idx, 1].set_title(f'{condition.name.upper()} - Balanced Distribution')
            axes[idx, 1].set_xlabel('Value')
            axes[idx, 1].set_ylabel('Count')

    plt.tight_layout()

    try:
        from __main__ import plot_dirs
        plt.savefig(os.path.join(plot_dirs['analysis_plots'], 'distribution_comparison.png'),
                    dpi=300, bbox_inches='tight')
    except:
        plt.savefig('distribution_comparison.png', dpi=300, bbox_inches='tight')

    plt.close()


def plot_generated_images_inference_enhanced(generator, latent_dim, n_channels, condition_manager,
                                             model_path_saving, patch_size, balanced_patch_info,
                                             non_empty_classes):
    """
    Generate inference images with varying condition values for all 4 meniscus conditions.
    Creates a grid where each row varies a different condition.
    """
    print("\nGenerating inference visualization grid for all conditions...")

    # Get active conditions (max 4 for visualization)
    active_conditions = [cond.name for cond in condition_manager.active_conditions][:4]
    n_conditions = len(active_conditions)

    if n_conditions == 0:
        print("No active conditions, generating unconditional samples...")

        # Generate unconditional samples
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        axes = axes.flatten()

        for i in range(10):
            noise = np.random.randn(1, latent_dim)
            img = generator.predict([noise], verbose=0)
            img = (img[0] + 1) / 2

            if n_channels == 1:
                axes[i].imshow(img[:, :, 0], cmap='gray')
            else:
                axes[i].imshow(img)

            axes[i].axis('off')
            axes[i].set_title(f'Sample {i + 1}', fontsize=9)

        plt.suptitle('Unconditional Generated Samples', fontsize=12)
        plt.tight_layout()

        save_path = os.path.join(model_path_saving, 'inference_unconditional_samples.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Unconditional samples saved to: {save_path}")
        plt.close()

        return  # Exit after generating unconditional samples

    # Use a single noise vector
    noise_vector = np.random.randn(1, latent_dim)

    # Number of variations per condition
    n_variations = 10

    # Create figure with 4 rows (one per condition) x 10 columns (variations)
    fig, axes = plt.subplots(n_conditions, n_variations, figsize=(n_variations * 2.2, n_conditions * 2.5))

    # Handle single condition case
    if n_conditions == 1:
        axes = axes.reshape(1, -1)

    # For each condition, vary it while keeping others fixed
    for cond_idx, condition_name in enumerate(active_conditions):
        # Get value range for this condition
        values = [p[f'{condition_name}_value'] for p in balanced_patch_info]
        min_val, max_val = min(values), max(values)

        # Calculate mean values for other conditions (to keep fixed)
        fixed_values = {}
        for other_cond in active_conditions:
            if other_cond != condition_name:
                other_values = [p[f'{other_cond}_value'] for p in balanced_patch_info]
                fixed_values[other_cond] = np.mean(other_values)

        for var_idx in range(n_variations):
            # Create generator inputs
            generator_inputs = [noise_vector.copy()]

            # Vary the current condition
            varied_value = min_val + (max_val - min_val) * var_idx / (n_variations - 1)

            # Set all condition values
            for cond in active_conditions:
                if cond == condition_name:
                    # Use the varied value
                    generator_inputs.append(np.array([[varied_value]]))
                else:
                    # Use fixed mean value
                    generator_inputs.append(np.array([[fixed_values[cond]]]))

            # Generate image
            img = generator.predict(generator_inputs, verbose=0)
            img = (img[0] + 1) / 2  # Convert from [-1,1] to [0,1]

            # Display (handle any n_channels)
            if n_channels == 1:
                axes[cond_idx, var_idx].imshow(img[:, :, 0], cmap='gray')
            elif n_channels == 3:
                axes[cond_idx, var_idx].imshow(img)
            else:
                # For multi-channel 3D (n_channels > 3), show middle slice
                middle_slice = n_channels // 2
                axes[cond_idx, var_idx].imshow(img[:, :, middle_slice], cmap='gray')

            # Add title for first row
            if cond_idx == 0:
                axes[cond_idx, var_idx].set_title(f'{varied_value:.2f}', fontsize=9)

            # Add ylabel for first column
            if var_idx == 0:
                condition_label = {
                    'fvf': 'FVF',
                    'alignment': 'Alignment',
                    'thickness': 'Thickness',
                    'complexity': 'Complexity'
                }.get(condition_name, condition_name.upper())
                axes[cond_idx, var_idx].set_ylabel(condition_label, fontsize=11, rotation=90)

    plt.suptitle(
        'Meniscus Generation: Condition Variation Analysis\n(Single noise vector, varying each condition independently)',
        fontsize=13, y=1.02)
    plt.tight_layout()

    # Save figure
    save_path = os.path.join(model_path_saving, 'inference_condition_variations.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Inference visualization saved to: {save_path}")
    plt.close()

    # Also create a combined variation plot
    print("\nGenerating combined condition variation plot...")

    # Create a smaller grid showing diagonal variation (all conditions change together)
    fig2, axes2 = plt.subplots(1, n_variations, figsize=(n_variations * 2.2, 2.5))

    for var_idx in range(n_variations):
        # Create generator inputs
        generator_inputs = [noise_vector.copy()]

        # Vary all conditions together (from low to high)
        progress = var_idx / (n_variations - 1)

        for condition_name in active_conditions:
            values = [p[f'{condition_name}_value'] for p in balanced_patch_info]
            min_val, max_val = min(values), max(values)
            varied_value = min_val + (max_val - min_val) * progress
            generator_inputs.append(np.array([[varied_value]]))

        # Generate image
        img = generator.predict(generator_inputs, verbose=0)
        img = (img[0] + 1) / 2

        # Display (handle any n_channels)
        if n_channels == 1:
            axes2[var_idx].imshow(img[:, :, 0], cmap='gray')
        elif n_channels == 3:
            axes2[var_idx].imshow(img)
        else:
            # For multi-channel 3D (n_channels > 3), show middle slice
            middle_slice = n_channels // 2
            axes2[var_idx].imshow(img[:, :, middle_slice], cmap='gray')

        axes2[var_idx].axis('off')
        axes2[var_idx].set_title(f'{progress:.1%}', fontsize=9)

    plt.suptitle('All Conditions Varying Together (0% = all minimum, 100% = all maximum)', fontsize=12)
    plt.tight_layout()

    save_path2 = os.path.join(model_path_saving, 'inference_combined_variation.png')
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"Combined variation plot saved to: {save_path2}")
    plt.close()


def evaluate_generator_accuracy(generator, latent_dim, balanced_patch_info, n_classes,
                                unet_model, image_type, threshold_value, condition_manager, n_channels=10):
    """
    Evaluate generator's ability to generate images with target conditions.
    """
    print("\nEvaluating generator condition accuracy...")

    # For meniscus, we evaluate how well the generator reproduces the condition values
    num_samples = min(100, len(balanced_patch_info))

    results = {}

    for condition in condition_manager.active_conditions:
        target_values = []
        generated_values = []

        # Sample patches
        selected_samples = random.sample(balanced_patch_info, num_samples)

        for sample in selected_samples:
            # Create generator inputs with target conditions
            generator_inputs = [np.random.randn(1, latent_dim)]

            # Add all condition values
            for cond in condition_manager.active_conditions:
                value = sample[f'{cond.name}_value']
                generator_inputs.append(np.array([[value]]))

                if cond.name == condition.name:
                    target_values.append(value)

            # Generate image
            generated_image = generator.predict(generator_inputs)
            generated_image = ((generated_image[0] + 1) / 2.0 * 255).astype(np.uint8)

            # Calculate the condition value from generated image
            # Import the calculation functions
            from base_code.data_processing import (
                calculate_fiber_volume_fraction,
                calculate_fiber_alignment_rose,
                calculate_mean_fiber_thickness,
                calculate_texture_complexity
            )

            if condition.name == 'fvf':
                calc_value = calculate_fiber_volume_fraction(generated_image)
            elif condition.name == 'alignment':
                calc_value = calculate_fiber_alignment_rose(generated_image)
            elif condition.name == 'thickness':
                calc_value = calculate_mean_fiber_thickness(generated_image)
            elif condition.name == 'complexity':
                calc_value = calculate_texture_complexity(generated_image)
            else:
                calc_value = 0.5  # Default

            generated_values.append(calc_value)

        # Calculate correlation
        target_values = np.array(target_values)
        generated_values = np.array(generated_values)

        if len(target_values) > 0:
            correlation = np.corrcoef(target_values, generated_values)[0, 1]
            mae = np.mean(np.abs(target_values - generated_values))

            results[f'{condition.name}_correlation'] = correlation
            results[f'{condition.name}_mae'] = mae

            print(f"\n{condition.name.upper()}:")
            print(f"  Correlation: {correlation:.3f}")
            print(f"  MAE: {mae:.3f}")

    return results


def print_data_ranges(balanced_patch_info, condition_manager):
    """Print data ranges for active conditions."""
    print("\nData Ranges for Active Conditions:")

    for condition in condition_manager.active_conditions:
        if f'{condition.name}_value' in balanced_patch_info[0]:
            values = [patch[f'{condition.name}_value'] for patch in balanced_patch_info]
            print(f"\n{condition.name.upper()}:")
            print(f"  Range: [{min(values):.4f}, {max(values):.4f}]")
            print(f"  Mean: {np.mean(values):.4f}")
            print(f"  Std: {np.std(values):.4f}")


def save_inference_metadata(model_path, n_channels, patch_size, latent_dim,
                            balanced_patch_info, condition_manager, image_type,
                            porosity_method=None, porosity_threshold_value=None):
    """Save metadata for future inference."""
    # Create metadata directory
    metadata_dir = os.path.join(model_path, 'metadata')
    os.makedirs(metadata_dir, exist_ok=True)

    # Collect metadata
    metadata = {
        'n_channels': n_channels,
        'patch_size': patch_size,
        'latent_dim': latent_dim,
        'image_type': image_type.value if hasattr(image_type, 'value') else str(image_type),
        'meniscus_conditions': True  # Flag for meniscus-specific processing
    }

    # Save as JSON
    with open(os.path.join(metadata_dir, 'config.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save condition information
    condition_data = {
        'active_conditions': [cond.name for cond in condition_manager.active_conditions],
        'condition_stats': {},
        'class_ranges': {}  # NEW: Add class-specific ranges
    }

    # Save statistics for each condition
    for condition in condition_manager.active_conditions:
        if f'{condition.name}_value' in balanced_patch_info[0]:
            values = [patch[f'{condition.name}_value'] for patch in balanced_patch_info]
            condition_data['condition_stats'][condition.name] = {
                'min': float(min(values)),
                'max': float(max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }

    # NEW: Calculate and save class-specific ranges for enabled conditions
    active_conditions = [cond.name for cond in condition_manager.active_conditions]
    
    if len(active_conditions) == 1:
        # Single condition mode
        condition_name = active_conditions[0]
        class_key = f'{condition_name}_class'
        condition_data['class_ranges'][condition_name] = {}
        
        # Group patches by class
        class_groups = {}
        for patch in balanced_patch_info:
            if class_key in patch:
                class_idx = patch[class_key]
                if class_idx not in class_groups:
                    class_groups[class_idx] = []
                class_groups[class_idx].append(patch)
        
        # Calculate ranges for each class
        for class_idx, patches in sorted(class_groups.items()):
            values = [p[f'{condition_name}_value'] for p in patches]
            condition_data['class_ranges'][condition_name][str(class_idx)] = {
                'min': float(min(values)),
                'max': float(max(values)),
                'mean': float(np.mean(values)),
                'count': len(patches)
            }
    
    elif len(active_conditions) > 1:
        # Multi-condition mode
        # First, collect ranges for each condition separately
        for condition_name in active_conditions:
            condition_data['class_ranges'][condition_name] = {}
            class_key = f'{condition_name}_class'
            
            # Group patches by class for this condition
            class_groups = {}
            for patch in balanced_patch_info:
                if class_key in patch:
                    class_idx = patch[class_key]
                    if class_idx not in class_groups:
                        class_groups[class_idx] = []
                    class_groups[class_idx].append(patch)
            
            # Calculate ranges for each class
            for class_idx, patches in sorted(class_groups.items()):
                values = [p[f'{condition_name}_value'] for p in patches]
                condition_data['class_ranges'][condition_name][str(class_idx)] = {
                    'min': float(min(values)),
                    'max': float(max(values)),
                    'mean': float(np.mean(values)),
                    'count': len(patches)
                }
        
        # Also save composite class information
        condition_data['composite_class_ranges'] = {}
        
        # Group patches by composite class keys
        composite_groups = {}
        for patch in balanced_patch_info:
            # Build composite key from all active condition classes
            composite_key = tuple(patch.get(f'{cond}_class', -1) for cond in active_conditions)
            
            # Skip if any condition class is missing
            if -1 in composite_key:
                continue
            
            if composite_key not in composite_groups:
                composite_groups[composite_key] = []
            composite_groups[composite_key].append(patch)
        
        # Calculate ranges for each composite class
        for composite_key, patches in sorted(composite_groups.items()):
            key_str = str(composite_key)
            condition_data['composite_class_ranges'][key_str] = {
                'class_indices': {active_conditions[i]: composite_key[i] for i in range(len(active_conditions))},
                'count': len(patches)
            }
            
            # Add ranges for each condition in this composite class
            for condition_name in active_conditions:
                values = [p[f'{condition_name}_value'] for p in patches]
                condition_data['composite_class_ranges'][key_str][condition_name] = {
                    'min': float(min(values)),
                    'max': float(max(values)),
                    'mean': float(np.mean(values))
                }

    # Save condition data
    with open(os.path.join(metadata_dir, 'conditions.json'), 'w') as f:
        json.dump(condition_data, f, indent=2)

    # Save sample patches for debugging
    sample_patches = balanced_patch_info[:10] if balanced_patch_info else []
    with open(os.path.join(metadata_dir, 'sample_patches.pkl'), 'wb') as f:
        pickle.dump(sample_patches, f)

    print(f"Metadata saved to {metadata_dir}")
    print(f"  - Saved ranges for {len(active_conditions)} active condition(s)")
    if 'class_ranges' in condition_data:
        for cond in active_conditions:
            if cond in condition_data['class_ranges']:
                n_classes = len(condition_data['class_ranges'][cond])
                print(f"  - {cond}: {n_classes} non-empty classes")
    if 'composite_class_ranges' in condition_data:
        print(f"  - Composite classes: {len(condition_data['composite_class_ranges'])} unique combinations")


def create_results_directory(patch_size, image_type, condition_manager, code_version, base_dir=None, base_run_number=1):
    """
    Create results directory with automatic run numbering.

    Args:
        base_dir: Base directory for results. If None, uses default Windows path
    """
    # Use provided base_dir or default
    if base_dir is None:
        base_dir = "results"

    base_dir = Path(base_dir)

    # Get current date
    current_date = datetime.datetime.now().strftime("%y%m%d")

    # Image type
    img_type_str = image_type.value.lower()

    # Condition string based on active conditions
    active_conditions = [cond.name for cond in condition_manager.active_conditions]
    if active_conditions:
        condition_str = "conditional_" + "_".join(active_conditions)
    else:
        condition_str = "unconditional"

    # Find available run number
    run_number = base_run_number
    while True:
        run_size_str = f"R{run_number}_{patch_size}"
        full_path = base_dir / current_date / code_version / img_type_str / condition_str / run_size_str

        if not full_path.exists():
            break

        run_number += 1

    # Create directory
    full_path.mkdir(parents=True, exist_ok=True)
    print(f"Created results directory: {full_path}")

    return str(full_path)


def save_training_images_by_class(original_images, balanced_patch_info, patch_size, save_dir, condition_manager, n_slices=16):
    """Save training images organized by condition classes (supports multi-condition)."""
    import cv2

    # Create training directory
    training_dir = os.path.join(save_dir, "Training_Patches")
    os.makedirs(training_dir, exist_ok=True)

    print(f"\nSaving training patches to: {training_dir}")

    # Detect if we're working with 3D volumes
    is_3d = len(original_images[0].shape) == 3 and original_images[0].shape[2] > 1

    if is_3d:
        print(f"  Detected 3D mode: Saving middle slice (slice {n_slices // 2}) from each volume")
        from base_code.data_processing import read_patch_3D
        # n_slices is now passed as parameter, no need to calculate
        middle_slice_idx = n_slices // 2  # Will be 8 if n_slices=16
    else:
        from base_code.data_processing import read_patch

    # Get active conditions
    active_conditions = [cond.name for cond in condition_manager.active_conditions]

    if not active_conditions:
        # ========== UNCONDITIONAL MODE ==========
        total_patches = len(balanced_patch_info)
        print(f"  Unconditional mode: Saving all {total_patches} patches without class organization")

        saved_count = 0
        for i, patch_info in enumerate(balanced_patch_info):
            try:
                if is_3d:
                    patch_3d = read_patch_3D(original_images, patch_info, patch_size, n_slices)
                    patch = patch_3d[:, :, middle_slice_idx]
                else:
                    patch = read_patch(original_images, patch_info, patch_size)
                    if len(patch.shape) == 3 and patch.shape[2] == 1:
                        patch = patch[:, :, 0]

                filename = f"patch_{i:04d}.png"
                filepath = os.path.join(training_dir, filename)
                cv2.imwrite(filepath, patch)
                saved_count += 1

                if (i + 1) % 50 == 0:
                    print(f"    Progress: {i + 1}/{total_patches} patches saved...")

            except Exception as e:
                print(f"  Warning: Failed to save patch {i}: {e}")

        print(f"  ✓ Saved {saved_count} patches")

    elif len(active_conditions) == 1:
        # ========== SINGLE CONDITION MODE ==========
        primary_condition = active_conditions[0]
        class_key = f'{primary_condition}_class'

        # Group patches by class
        class_groups = {}
        for patch in balanced_patch_info:
            if class_key in patch:
                class_idx = patch[class_key]
                if class_idx not in class_groups:
                    class_groups[class_idx] = []
                class_groups[class_idx].append(patch)

        # Save patches for each class
        for class_idx, patches in class_groups.items():
            class_dir = os.path.join(training_dir, f"{primary_condition}_class_{class_idx}")
            os.makedirs(class_dir, exist_ok=True)

            for i, patch_info in enumerate(patches):
                if is_3d:
                    patch_3d = read_patch_3D(original_images, patch_info, patch_size, n_slices)
                    patch = patch_3d[:, :, middle_slice_idx]
                else:
                    patch = read_patch(original_images, patch_info, patch_size)
                    if len(patch.shape) == 3 and patch.shape[2] == 1:
                        patch = patch[:, :, 0]

                filename = f"patch_{i:04d}.png"
                filepath = os.path.join(class_dir, filename)
                cv2.imwrite(filepath, patch)

            print(f"  Saved {len(patches)} patches for {primary_condition} class {class_idx}")

    else:
        # ========== MULTI-CONDITION MODE (NEW!) ==========
        print(f"  Multi-condition mode: Organizing by {len(active_conditions)} conditions")

        # Group patches by composite class keys
        composite_groups = {}
        for patch in balanced_patch_info:
            # Build composite key from all active condition classes
            composite_key = tuple(patch.get(f'{cond}_class', -1) for cond in active_conditions)

            # Skip if any condition class is missing
            if -1 in composite_key:
                continue

            if composite_key not in composite_groups:
                composite_groups[composite_key] = []
            composite_groups[composite_key].append(patch)

        print(f"  Found {len(composite_groups)} unique composite classes")

        # Save patches for each composite class
        for composite_key, patches in sorted(composite_groups.items()):
            # Create folder name from composite key
            # E.g., (4, 7) -> "fvf_4_alignment_7"
            folder_parts = []
            for cond_idx, cond_name in enumerate(active_conditions):
                class_val = composite_key[cond_idx]
                folder_parts.append(f"{cond_name}_{class_val}")

            class_dir_name = "_".join(folder_parts)
            class_dir = os.path.join(training_dir, class_dir_name)
            os.makedirs(class_dir, exist_ok=True)

            # Save patches
            for i, patch_info in enumerate(patches):
                if is_3d:
                    patch_3d = read_patch_3D(original_images, patch_info, patch_size, n_slices)
                    patch = patch_3d[:, :, middle_slice_idx]
                else:
                    patch = read_patch(original_images, patch_info, patch_size)
                    if len(patch.shape) == 3 and patch.shape[2] == 1:
                        patch = patch[:, :, 0]

                filename = f"patch_{i:04d}.png"
                filepath = os.path.join(class_dir, filename)
                cv2.imwrite(filepath, patch)

            # Print summary (show first few and total)
            if len(composite_groups) <= 20:
                # Show all if not too many
                print(f"  Saved {len(patches)} patches for {class_dir_name}")
            elif list(sorted(composite_groups.keys())).index(composite_key) < 5:
                # Show first 5
                print(f"  Saved {len(patches)} patches for {class_dir_name}")

        # Summary for many classes
        if len(composite_groups) > 20:
            print(f"  ... (showing first 5 of {len(composite_groups)} composite classes)")
            total_saved = sum(len(patches) for patches in composite_groups.values())
            print(f"  ✓ Total: Saved {total_saved} patches across {len(composite_groups)} composite classes")

def create_models_subdirectory(main_directory):
    """Create models subdirectory."""
    models_dir = os.path.join(main_directory, "Saved_Models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir