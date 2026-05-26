import numpy as np
import random
import sys
import os


# Simple data balancing without augmentation for meniscus images

def balance_dataset_with_repetition(patch_info, desired_images_per_class, min_images_per_class,
                                    condition_manager):
    """
    Balance dataset using simple repetition for underrepresented classes.
    Handles multi-condition scenarios by creating composite class keys.

    Args:
        patch_info: List of patch information dictionaries
        desired_images_per_class: Target number of patches per class
        min_images_per_class: Minimum patches required per class
        condition_manager: Manager containing active conditions

    Returns:
        Tuple of (balanced_patch_info, repetition_percentages, empty_classes, class_counts, non_empty_classes)
    """
    balanced_patch_info = []
    repetition_percentages = {}
    empty_classes = []
    class_counts = {}

    # Get active conditions
    active_conditions = [cond.name for cond in condition_manager.active_conditions]

    print("\nBalancing dataset using repetition...")
    print(f"Active conditions for balancing: {active_conditions}")

    if not active_conditions:
        # No conditions active, just return original data
        balanced_patch_info = patch_info
        class_counts = {0: len(balanced_patch_info)}
        non_empty_classes = {0}
    elif len(active_conditions) == 1:
        # Single condition balancing
        primary_condition = active_conditions[0]
        class_key = f'{primary_condition}_class'

        # Group patches by class
        classes_dict = {}
        for patch in patch_info:
            if class_key in patch:
                class_idx = patch[class_key]
                if class_idx not in classes_dict:
                    classes_dict[class_idx] = []
                classes_dict[class_idx].append(patch)

        # Process each class
        for class_idx, class_patches in sorted(classes_dict.items()):
            num_original = len(class_patches)

            if num_original < min_images_per_class:
                # Not enough patches in this class
                empty_classes.append(class_idx)
                print(f"  {primary_condition} Class {class_idx}: {num_original} patches (below minimum, skipping)")
                continue

            if num_original >= desired_images_per_class:
                # We have enough or too many, sample
                selected_patches = random.sample(class_patches, desired_images_per_class)
                rep_percentage = 0
            else:
                # Need to repeat some patches - EXACT COPIES
                selected_patches = class_patches.copy()
                repetitions_needed = desired_images_per_class - num_original

                for _ in range(repetitions_needed):
                    selected_patches.append(random.choice(class_patches))

                rep_percentage = (repetitions_needed / desired_images_per_class) * 100

            balanced_patch_info.extend(selected_patches)
            repetition_percentages[class_idx] = rep_percentage
            class_counts[class_idx] = len(selected_patches)

            print(f"  {primary_condition} Class {class_idx}: {num_original} -> {len(selected_patches)} patches "
                  f"({rep_percentage:.1f}% repetition)")
    else:
        # Multi-condition balancing - create composite classes
        print(f"Multi-condition balancing with {len(active_conditions)} conditions")

        # Create composite class keys for each patch
        composite_classes = {}
        for patch in patch_info:
            # Build composite key from all active condition classes
            composite_key = tuple(patch.get(f'{cond}_class', 0) for cond in active_conditions)
            if composite_key not in composite_classes:
                composite_classes[composite_key] = []
            composite_classes[composite_key].append(patch)

        print(f"Found {len(composite_classes)} unique composite classes")

        # Balance each composite class
        for composite_key, class_patches in composite_classes.items():
            num_original = len(class_patches)

            if num_original < min_images_per_class:
                empty_classes.append(composite_key)
                continue

            # Determine how many patches we need
            target = desired_images_per_class

            if num_original >= target:
                selected_patches = random.sample(class_patches, target)
                rep_percentage = 0
            else:
                selected_patches = class_patches.copy()
                repetitions_needed = target - num_original

                for _ in range(repetitions_needed):
                    selected_patches.append(random.choice(class_patches))

                rep_percentage = (repetitions_needed / target) * 100

            balanced_patch_info.extend(selected_patches)
            repetition_percentages[composite_key] = rep_percentage
            class_counts[composite_key] = len(selected_patches)

    # Shuffle the balanced dataset
    random.shuffle(balanced_patch_info)

    # Get non-empty classes
    non_empty_classes = set(class_counts.keys()) - set(empty_classes)

    # Print statistics
    print("\nBalanced Dataset Statistics:")
    print(f"Total patches: {len(balanced_patch_info)}")
    print(f"Empty classes: {len(empty_classes)}")
    print(f"Non-empty classes: {len(non_empty_classes)}")

    return balanced_patch_info, repetition_percentages, empty_classes, class_counts, non_empty_classes


def balance_and_analyze_dataset(patch_info, desired_images_per_class, min_images_per_class,
                                condition_manager):
    """
    Wrapper function for dataset balancing that maintains compatibility with the pipeline.

    Args:
        patch_info: List of patch information dictionaries
        desired_images_per_class: Target number of patches per class
        min_images_per_class: Minimum patches required per class
        condition_manager: Manager containing active conditions

    Returns:
        Tuple of balanced data and statistics
    """
    return balance_dataset_with_repetition(
        patch_info,
        desired_images_per_class,
        min_images_per_class,
        condition_manager
    )