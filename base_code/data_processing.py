import cv2
import numpy as np
import os
import random
from enum import Enum
from skimage import filters, morphology
from scipy.ndimage import distance_transform_edt
from scipy import ndimage
import matplotlib.pyplot as plt

class ImageType(Enum):
    GRAYSCALE = "grayscale"
    BINARY = "binary"


def load_images_meniscus(directory_path, n_channels=1):
    """
    Load grayscale meniscus images from a single directory.

    Args:
        directory_path: Path to directory containing meniscus images
        n_channels: Number of channels (1 for grayscale)

    Returns:
        List of loaded images
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"The directory {directory_path} does not exist.")

    print(f"Using directory: {directory_path}")
    original_images = []

    # Load images directly from the directory
    image_filenames = [f for f in os.listdir(directory_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'))]

    print(f"Found {len(image_filenames)} images")

    for filename in image_filenames:
        file_path = os.path.join(directory_path, filename)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        if image is not None:
            if n_channels == 1:
                image = image[..., np.newaxis]
            original_images.append(image)
        else:
            print(f"Warning: Could not load {filename}")

    return original_images


def load_image_stacks_meniscus(directory_path, n_slices=None):
    """
    Load z-stack volumes from sample subdirectories.

    Supports two modes:
    1. Single sample: Load n_slices from directory
    2. Multiple samples: Load all slices from each subdirectory

    Args:
        directory_path: Path to directory containing samples
        n_slices: Number of slices to load per sample
                  If None, loads ALL slices found

    Returns:
        List[np.ndarray]: List of 3D volumes, shape (H, W, D)

    Example structures:
        Single sample:
            directory/
            ├── slice_0000.tif
            ├── slice_0001.tif
            └── ...

        Multiple samples:
            directory/
            ├── Sample_1/
            │   ├── slice_0000.tif
            │   └── ...
            ├── Sample_2/
            └── Sample_3/
    """
    import re
    from pathlib import Path

    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"The directory {directory_path} does not exist.")

    print(f"Loading z-stack volumes from: {directory_path}")

    # Check if directory contains subdirectories (multiple samples)
    subdirs = [d for d in os.listdir(directory_path)
               if os.path.isdir(os.path.join(directory_path, d))]

    volumes = []

    if subdirs:
        # ========== MULTIPLE SAMPLES MODE ==========
        print(f"Detected {len(subdirs)} sample subdirectories")

        for subdir in sorted(subdirs):
            subdir_path = os.path.join(directory_path, subdir)

            # Find all TIF files in this subdirectory
            tif_files = [f for f in os.listdir(subdir_path)
                         if f.lower().endswith(('.tif', '.tiff', '.png'))]

            if not tif_files:
                print(f"  ⚠️  Warning: No TIF files found in {subdir}, skipping")
                continue

            # Determine how many slices to load
            total_slices = len(tif_files)
            slices_to_load = n_slices if n_slices is not None else total_slices

            if slices_to_load > total_slices:
                print(f"  ⚠️  Warning: {subdir} has only {total_slices} slices, "
                      f"requested {slices_to_load}")
                slices_to_load = total_slices

            print(f"\n  Loading {subdir}: {slices_to_load} slices")

            # Extract numerical part for sorting
            def extract_number(filename):
                match = re.search(r'(\d+)', filename)
                return int(match.group(1)) if match else filename

            # Sort files numerically
            tif_files_sorted = sorted(tif_files, key=extract_number)

            # Load slices
            volume_slices = []
            expected_shape = None

            for idx, filename in enumerate(tif_files_sorted[:slices_to_load]):
                file_path = os.path.join(subdir_path, filename)
                slice_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

                if slice_img is None:
                    raise ValueError(f"Failed to load {filename}")

                # Validate dimensions
                if expected_shape is None:
                    expected_shape = slice_img.shape
                elif slice_img.shape != expected_shape:
                    raise ValueError(
                        f"Dimension mismatch in {subdir} at slice {idx}: "
                        f"expected {expected_shape}, got {slice_img.shape}"
                    )

                volume_slices.append(slice_img)

                # Progress indicator
                if (idx + 1) % 10 == 0 or idx == 0 or (idx + 1) == slices_to_load:
                    print(f"    Loaded slice {idx + 1}/{slices_to_load}")

            # Stack into 3D volume
            volume_3d = np.stack(volume_slices, axis=2)
            volumes.append(volume_3d)

            print(f"  ✅ {subdir}: Created volume {volume_3d.shape}")

    else:
        # ========== SINGLE SAMPLE MODE (Original behavior) ==========
        print("Single sample mode (no subdirectories detected)")

        # Find all TIF files
        tif_files = [f for f in os.listdir(directory_path)
                     if f.lower().endswith(('.tif', '.tiff'))]

        if not tif_files:
            raise ValueError(f"No TIF files found in {directory_path}")

        total_slices = len(tif_files)
        slices_to_load = n_slices if n_slices is not None else total_slices

        if slices_to_load > total_slices:
            raise ValueError(f"Only {total_slices} TIF files found, but need {slices_to_load}")

        print(f"Found {total_slices} TIF files")
        print(f"Loading {slices_to_load} slices")

        # Extract and sort
        def extract_number(filename):
            match = re.search(r'(\d+)', filename)
            return int(match.group(1)) if match else filename

        tif_files_sorted = sorted(tif_files, key=extract_number)

        # Load slices
        volume_slices = []
        expected_shape = None

        for idx, filename in enumerate(tif_files_sorted[:slices_to_load]):
            file_path = os.path.join(directory_path, filename)
            slice_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            if slice_img is None:
                raise ValueError(f"Failed to load {filename}")

            if expected_shape is None:
                expected_shape = slice_img.shape
                print(f"  Slice dimensions: {expected_shape}")
            elif slice_img.shape != expected_shape:
                raise ValueError(
                    f"Dimension mismatch at slice {idx}: "
                    f"expected {expected_shape}, got {slice_img.shape}"
                )

            volume_slices.append(slice_img)

            if (idx + 1) % 10 == 0 or idx == 0 or (idx + 1) == slices_to_load:
                print(f"  Loaded slice {idx + 1}/{slices_to_load}: {filename}")

        # Stack into 3D volume
        volume_3d = np.stack(volume_slices, axis=2)
        volumes.append(volume_3d)

        print(f"\n✅ Successfully created 3D volume:")
        print(f"   Shape: {volume_3d.shape}")
        print(f"   Data type: {volume_3d.dtype}")
        print(f"   Value range: [{volume_3d.min()}, {volume_3d.max()}]")

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"LOADING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total volumes loaded: {len(volumes)}")
    for i, vol in enumerate(volumes):
        print(f"  Volume {i + 1}: {vol.shape}")
    print(f"{'=' * 60}\n")

    return volumes


def visualize_volume_slices(volume, slice_indices=None, title="3D Volume Slices"):
    """
    Visualize selected slices from a 3D volume.

    Args:
        volume: 3D numpy array of shape (H, W, D)
        slice_indices: List of slice indices to show, or None for evenly spaced
        title: Figure title

    Example:
        >>> volume = volumes[0]
        >>> visualize_volume_slices(volume, slice_indices=[0, 4, 9])
    """
    if len(volume.shape) != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape}")

    H, W, D = volume.shape

    # Default: show 5 evenly spaced slices
    if slice_indices is None:
        if D >= 5:
            slice_indices = [0, D // 4, D // 2, 3 * D // 4, D - 1]
        else:
            slice_indices = list(range(D))

    n_show = len(slice_indices)

    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4))

    if n_show == 1:
        axes = [axes]

    for idx, slice_idx in enumerate(slice_indices):
        if slice_idx >= D:
            print(f"Warning: Slice index {slice_idx} exceeds depth {D}")
            continue

        slice_2d = volume[:, :, slice_idx]

        axes[idx].imshow(slice_2d, cmap='gray', vmin=0, vmax=255)
        axes[idx].set_title(f'Slice {slice_idx}/{D - 1}', fontsize=12)
        axes[idx].axis('off')

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    return fig











def detect_circular_region(image, threshold_percentile=5):
    """
    Detect the circular tissue region in the meniscus image.

    Args:
        image: Input image (grayscale)
        threshold_percentile: Percentile for threshold determination

    Returns:
        center_x, center_y, radius, binary_mask
    """
    # Apply slight Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 1)

    # Use percentile-based threshold to handle varying intensities
    threshold_value = np.percentile(image[image > 0], threshold_percentile)
    _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary.astype(np.uint8),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour (should be the circular tissue region)
        largest_contour = max(contours, key=cv2.contourArea)

        # Fit a circle to the contour
        (center_x, center_y), radius = cv2.minEnclosingCircle(largest_contour)

        # Create a clean circular mask
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.circle(mask, (int(center_x), int(center_y)), int(radius), 255, -1)

        return int(center_x), int(center_y), int(radius), mask

    # Fallback: assume centered circle
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    radius = min(h, w) // 2 - 50  # Conservative estimate

    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)

    return center_x, center_y, radius, mask


def is_patch_valid(image, x, y, mask, patch_size, min_tissue_ratio=0.95):
    """
    Check if a patch at position (x, y) contains sufficient tissue.

    Args:
        image: Full image
        x, y: Top-left corner of the patch
        mask: Binary mask of tissue region
        patch_size: Size of the patch
        min_tissue_ratio: Minimum ratio of tissue pixels required

    Returns:
        is_valid, tissue_ratio
    """
    # Extract mask patch
    mask_patch = mask[y:y + patch_size, x:x + patch_size]

    # Calculate tissue ratio
    tissue_pixels = np.sum(mask_patch > 0)
    total_pixels = patch_size * patch_size
    tissue_ratio = tissue_pixels / total_pixels

    # Check if patch meets minimum tissue requirement
    is_valid = tissue_ratio >= min_tissue_ratio

    return is_valid, tissue_ratio


def detect_circular_region_3D(volume_3d, threshold_percentile=5):
    """
    Detect circular tissue region for each slice in a 3D volume.

    Args:
        volume_3d: Input 3D volume (H, W, D) where D is number of slices
        threshold_percentile: Percentile for threshold determination

    Returns:
        List of (center_x, center_y, radius, mask) tuples for each slice
    """
    H, W, D = volume_3d.shape
    regions = []

    print(f"Detecting circular tissue regions for {D} slices...")

    for z in range(D):
        slice_2d = volume_3d[:, :, z]

        # Use existing 2D detection function
        center_x, center_y, radius, mask = detect_circular_region(
            slice_2d, threshold_percentile
        )

        regions.append((center_x, center_y, radius, mask))

    # Find common region across all slices (use median center and minimum radius)
    centers_x = [r[0] for r in regions]
    centers_y = [r[1] for r in regions]
    radii = [r[2] for r in regions]

    # Use median center (most stable across slices)
    common_center_x = int(np.median(centers_x))
    common_center_y = int(np.median(centers_y))

    # Use minimum radius (most conservative - ensures all slices valid)
    common_radius = int(np.min(radii) * 0.95)  # 5% buffer for safety

    print(f"  Common tissue region across all slices:")
    print(f"    Center: ({common_center_x}, {common_center_y})")
    print(f"    Radius: {common_radius}")
    print(f"    Radius variation: {np.min(radii):.1f} - {np.max(radii):.1f}")

    return regions, (common_center_x, common_center_y, common_radius)


def is_patch_valid_3D(volume, x, y, z, masks_3d, patch_size, n_slices,
                      min_tissue_ratio=0.95):
    """
    Check if a 3D patch at position (x, y, z) contains sufficient tissue
    across ALL depth slices.

    Args:
        volume: Full 3D volume (H, W, D)
        x, y: Top-left corner of the patch in XY plane
        z: Starting depth position
        masks_3d: List of tissue masks for each slice
        patch_size: Size of the patch in XY (e.g., 256)
        n_slices: Number of slices in patch (e.g., 10)
        min_tissue_ratio: Minimum ratio of tissue pixels required per slice

    Returns:
        is_valid: Boolean indicating if patch is valid
        min_tissue_ratio_value: Minimum tissue ratio across all slices
    """
    tissue_ratios = []

    # Check each slice in the 3D patch
    for z_offset in range(n_slices):
        slice_idx = z + z_offset

        # Extract mask patch for this slice
        mask_patch = masks_3d[slice_idx][y:y + patch_size, x:x + patch_size]

        # Calculate tissue ratio for this slice
        tissue_pixels = np.sum(mask_patch > 0)
        total_pixels = patch_size * patch_size
        tissue_ratio = tissue_pixels / total_pixels

        tissue_ratios.append(tissue_ratio)

    # Find minimum tissue ratio across all slices
    min_ratio = np.min(tissue_ratios)

    # Patch is valid only if ALL slices meet the threshold
    is_valid = min_ratio >= min_tissue_ratio

    return is_valid, min_ratio













# Meniscus property calculation functions
def calculate_fiber_volume_fraction(patch):
    """
    Calculate the fraction of image occupied by fiber material using Otsu method.

    Args:
        patch: Grayscale image patch

    Returns:
        fvf: Fiber volume fraction in [0, 1] range
    """
    # Handle multi-channel input
    if len(patch.shape) == 3:
        patch = patch[:, :, 0]

    # Ensure patch is in uint8 format
    if patch.dtype != np.uint8:
        if patch.max() <= 1:
            patch = (patch * 255).astype(np.uint8)
        else:
            patch = patch.astype(np.uint8)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(patch, (5, 5), 1)

    # Use Otsu's method for automatic thresholding
    threshold_value = filters.threshold_otsu(blurred)
    binary = blurred > threshold_value

    # Calculate FVF
    fiber_pixels = np.sum(binary)
    total_pixels = binary.size
    fvf = fiber_pixels / total_pixels

    # Ensure it's in [0,1]
    fvf = np.clip(fvf, 0.0, 1.0)

    return float(fvf)


def calculate_fiber_alignment_rose(patch, window_size=7):
    """
    Calculate fiber alignment using Rose Diagram Mean Resultant Length (MRL).

    Args:
        patch: Grayscale image patch
        window_size: Size of local window (default 7 for 256x256 patches)

    Returns:
        alignment: MRL value in [0, 1] range
    """
    # Handle multi-channel input
    if len(patch.shape) == 3:
        patch = patch[:, :, 0]

    # Convert to float for gradient computation
    if patch.dtype == np.uint8:
        patch = patch.astype(np.float32) / 255.0

    # Compute gradients using Sobel operators
    grad_x = ndimage.sobel(patch, axis=1, mode='reflect')
    grad_y = ndimage.sobel(patch, axis=0, mode='reflect')

    # Calculate orientation at each pixel (in degrees)
    orientations = np.arctan2(grad_y, grad_x) * 180 / np.pi
    orientations = orientations % 180  # Convert to 0-180 range (undirected fibers)

    # Filter out background (low gradient regions)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    mask = gradient_magnitude > np.percentile(gradient_magnitude, 10)

    # Get orientations only in fiber regions
    valid_orientations = orientations[mask]
    weights = gradient_magnitude[mask]

    # Create histogram of orientations for rose diagram
    n_bins = 36  # 5-degree bins
    bins = np.linspace(0, 180, n_bins + 1)
    hist, _ = np.histogram(valid_orientations, bins=bins, weights=weights)

    # Normalize histogram
    if np.sum(hist) > 0:
        hist = hist / np.sum(hist)

    # Calculate Mean Resultant Length (MRL) using circular statistics
    bin_centers = (bins[:-1] + bins[1:]) / 2
    theta_rad = bin_centers * np.pi / 180

    # For undirected data, use doubled angles
    x_mean = np.sum(hist * np.cos(2 * theta_rad))
    y_mean = np.sum(hist * np.sin(2 * theta_rad))
    mrl = np.sqrt(x_mean ** 2 + y_mean ** 2)

    # Ensure it's in [0,1]
    mrl = np.clip(mrl, 0.0, 1.0)

    return float(mrl)


def calculate_mean_fiber_thickness(patch):
    """
    Calculate normalized fiber thickness.

    Args:
        patch: Grayscale image patch

    Returns:
        thickness_normalized: Normalized thickness value in [0, 1] range
    """
    # Handle multi-channel input
    if len(patch.shape) == 3:
        patch = patch[:, :, 0]

    # Ensure patch is in uint8 format
    if patch.dtype != np.uint8:
        if patch.max() <= 1:
            patch = (patch * 255).astype(np.uint8)
        else:
            patch = patch.astype(np.uint8)

    # Get binary mask using Otsu
    threshold_value = filters.threshold_otsu(patch)
    binary = patch > threshold_value

    # Clean up binary image
    binary = morphology.remove_small_objects(binary, min_size=20)
    binary = morphology.binary_closing(binary, morphology.disk(2))

    # Compute skeleton and distance transform
    skeleton = morphology.skeletonize(binary)
    distance = distance_transform_edt(binary)

    # Sample distances along skeleton
    skeleton_distances = distance[skeleton]

    if len(skeleton_distances) > 0:
        # Use median for robustness
        mean_thickness = 2 * np.median(skeleton_distances)
    else:
        # Default for thin fibers
        mean_thickness = 3.0

    # Normalize thickness to [0, 1] range
    MIN_THICKNESS = 1.0
    MAX_THICKNESS = 30.0
    thickness_norm = (mean_thickness - MIN_THICKNESS) / (MAX_THICKNESS - MIN_THICKNESS)
    thickness_norm = np.clip(thickness_norm, 0, 1)

    return float(thickness_norm)


def calculate_texture_complexity(patch, window_size=32):
    """
    Calculate texture complexity using Shannon entropy.

    Args:
        patch: Grayscale image patch
        window_size: Size of local windows

    Returns:
        complexity: Normalized texture complexity [0, 1]
    """
    # Handle multi-channel input
    if len(patch.shape) == 3:
        patch = patch[:, :, 0]

    # Ensure patch is uint8
    if patch.dtype != np.uint8:
        if patch.max() <= 1:
            patch = (patch * 255).astype(np.uint8)
        else:
            patch = patch.astype(np.uint8)

    h, w = patch.shape
    complexities = []

    # Slide window across image
    for y in range(0, h - window_size + 1, window_size // 2):
        for x in range(0, w - window_size + 1, window_size // 2):
            window = patch[y:y + window_size, x:x + window_size]

            # Calculate histogram
            hist, _ = np.histogram(window, bins=32, range=(0, 256))
            hist = hist.astype(np.float32)
            hist = hist[hist > 0]  # Remove zero bins

            if len(hist) > 1:
                # Normalize histogram
                hist = hist / hist.sum()
                # Calculate Shannon entropy
                window_entropy = -np.sum(hist * np.log2(hist + 1e-10))
                # Normalize by maximum possible entropy
                window_entropy = window_entropy / np.log2(32)
                complexities.append(window_entropy)

    if complexities:
        mean_complexity = np.mean(complexities)
    else:
        mean_complexity = np.std(patch) / 128.0

    mean_complexity = np.clip(mean_complexity, 0, 1)

    return float(mean_complexity)


# ============================================================================
# 3D CONDITION CALCULATION FUNCTIONS
# ============================================================================

def calculate_fiber_volume_fraction_3D(patch_3d):
    """
    Calculate volumetric fiber volume fraction by averaging across depth slices.

    This function computes FVF for each 2D slice independently, then returns
    the mean FVF across all slices. This provides a volumetric measure of
    fiber density throughout the 3D patch.

    Args:
        patch_3d: 3D numpy array of shape (H, W, n_slices)
                  Example: (256, 256, 10)

    Returns:
        float: Mean FVF value in [0, 1] range
               0 = no fiber content
               1 = 100% fiber content

    Example:
        >>> patch_3d = read_patch_3D(volumes, patch_info, 256, 10)
        >>> fvf_3d = calculate_fiber_volume_fraction_3D(patch_3d)
        >>> print(f"Volumetric FVF: {fvf_3d:.3f}")
    """
    if len(patch_3d.shape) != 3:
        raise ValueError(f"Expected 3D patch, got shape {patch_3d.shape}")

    H, W, n_slices = patch_3d.shape
    fvf_per_slice = []

    # Calculate FVF for each slice
    for z in range(n_slices):
        slice_2d = patch_3d[:, :, z]
        fvf = calculate_fiber_volume_fraction(slice_2d)  # Use existing 2D function
        fvf_per_slice.append(fvf)

    # Return volumetric average
    volumetric_fvf = float(np.mean(fvf_per_slice))

    return volumetric_fvf


def calculate_fiber_alignment_rose_3D(patch_3d):
    """
    Calculate volumetric fiber alignment by averaging MRL across depth slices.

    This function computes the Mean Resultant Length (MRL) for each 2D slice
    independently, then returns the mean MRL across all slices. This provides
    a volumetric measure of fiber alignment throughout the 3D patch.

    Args:
        patch_3d: 3D numpy array of shape (H, W, n_slices)
                  Example: (256, 256, 10)

    Returns:
        float: Mean alignment value in [0, 1] range
               0 = randomly oriented fibers (isotropic)
               1 = perfectly aligned fibers (anisotropic)

    Example:
        >>> alignment_3d = calculate_fiber_alignment_rose_3D(patch_3d)
        >>> print(f"Volumetric Alignment: {alignment_3d:.3f}")
    """
    if len(patch_3d.shape) != 3:
        raise ValueError(f"Expected 3D patch, got shape {patch_3d.shape}")

    H, W, n_slices = patch_3d.shape
    alignment_per_slice = []

    # Calculate alignment for each slice
    for z in range(n_slices):
        slice_2d = patch_3d[:, :, z]
        alignment = calculate_fiber_alignment_rose(slice_2d)  # Use existing 2D function
        alignment_per_slice.append(alignment)

    # Return volumetric average
    volumetric_alignment = float(np.mean(alignment_per_slice))

    return volumetric_alignment


def calculate_mean_fiber_thickness_3D(patch_3d):
    """
    Calculate volumetric fiber thickness by averaging across depth slices.

    This function computes normalized fiber thickness for each 2D slice
    independently, then returns the mean thickness across all slices.
    This provides a volumetric measure of fiber thickness throughout
    the 3D patch.

    Args:
        patch_3d: 3D numpy array of shape (H, W, n_slices)
                  Example: (256, 256, 10)

    Returns:
        float: Mean thickness value in [0, 1] range (normalized)
               0 = thinnest fibers (1-5 pixels)
               1 = thickest fibers (25-30 pixels)

    Example:
        >>> thickness_3d = calculate_mean_fiber_thickness_3D(patch_3d)
        >>> print(f"Volumetric Thickness: {thickness_3d:.3f}")
    """
    if len(patch_3d.shape) != 3:
        raise ValueError(f"Expected 3D patch, got shape {patch_3d.shape}")

    H, W, n_slices = patch_3d.shape
    thickness_per_slice = []

    # Calculate thickness for each slice
    for z in range(n_slices):
        slice_2d = patch_3d[:, :, z]
        thickness = calculate_mean_fiber_thickness(slice_2d)  # Use existing 2D function
        thickness_per_slice.append(thickness)

    # Return volumetric average
    volumetric_thickness = float(np.mean(thickness_per_slice))

    return volumetric_thickness


def calculate_texture_complexity_3D(patch_3d):
    """
    Calculate volumetric texture complexity by averaging across depth slices.

    This function computes Shannon entropy-based complexity for each 2D slice
    independently, then returns the mean complexity across all slices.
    This provides a volumetric measure of texture complexity throughout
    the 3D patch.

    Args:
        patch_3d: 3D numpy array of shape (H, W, n_slices)
                  Example: (256, 256, 10)

    Returns:
        float: Mean complexity value in [0, 1] range
               0 = uniform/simple texture
               1 = highly complex/heterogeneous texture

    Example:
        >>> complexity_3d = calculate_texture_complexity_3D(patch_3d)
        >>> print(f"Volumetric Complexity: {complexity_3d:.3f}")
    """
    if len(patch_3d.shape) != 3:
        raise ValueError(f"Expected 3D patch, got shape {patch_3d.shape}")

    H, W, n_slices = patch_3d.shape
    complexity_per_slice = []

    # Calculate complexity for each slice
    for z in range(n_slices):
        slice_2d = patch_3d[:, :, z]
        complexity = calculate_texture_complexity(slice_2d)  # Use existing 2D function
        complexity_per_slice.append(complexity)

    # Return volumetric average
    volumetric_complexity = float(np.mean(complexity_per_slice))

    return volumetric_complexity


def calculate_conditions_stratified(patch_3d):
    """
    Calculate conditions for three depth zones: superficial, middle, and deep.

    This function divides the 3D patch into three depth zones and calculates
    all four conditions for each zone separately. Useful for analyzing
    depth-dependent tissue properties.

    Args:
        patch_3d: 3D numpy array of shape (H, W, n_slices)
                  Example: (256, 256, 10)

    Returns:
        dict: Nested dictionary with structure:
              {
                  'superficial': {'fvf': 0.x, 'alignment': 0.x, ...},
                  'middle': {'fvf': 0.x, 'alignment': 0.x, ...},
                  'deep': {'fvf': 0.x, 'alignment': 0.x, ...}
              }

    Example:
        >>> stratified = calculate_conditions_stratified(patch_3d)
        >>> print(f"Superficial FVF: {stratified['superficial']['fvf']:.3f}")
        >>> print(f"Deep FVF: {stratified['deep']['fvf']:.3f}")
    """
    if len(patch_3d.shape) != 3:
        raise ValueError(f"Expected 3D patch, got shape {patch_3d.shape}")

    H, W, n_slices = patch_3d.shape

    # Define depth zones (for 10 slices)
    # Superficial: 0-2 (top 30%)
    # Middle: 3-6 (middle 40%)
    # Deep: 7-9 (bottom 30%)
    zones = {
        'superficial': patch_3d[:, :, 0:3],
        'middle': patch_3d[:, :, 3:7],
        'deep': patch_3d[:, :, 7:10]
    }

    results = {}

    for zone_name, zone_patch in zones.items():
        results[zone_name] = {
            'fvf': calculate_fiber_volume_fraction_3D(zone_patch),
            'alignment': calculate_fiber_alignment_rose_3D(zone_patch),
            'thickness': calculate_mean_fiber_thickness_3D(zone_patch),
            'complexity': calculate_texture_complexity_3D(zone_patch)
        }

    return results




def extract_and_save_patch_info_meniscus(original_images, patch_size,
                                         num_patches_total, condition_manager,
                                         min_tissue_ratio=0.95, overlap=0.50):
    """
    Extract patches from meniscus images with tissue validation and condition calculation.
    FIXED: Now uses DISTRIBUTED SAMPLING across all images.
    """
    all_patches = []

    # Get active conditions
    active_conditions = [cond.name for cond in condition_manager.active_conditions]

    print(f"Extracting patches with {min_tissue_ratio:.0%} minimum tissue content")
    print(f"Using DISTRIBUTED SAMPLING across ALL images")
    print(f"Active conditions: {active_conditions}")

    # ✅ CALCULATE PATCHES PER IMAGE (DISTRIBUTED APPROACH)
    n_images = len(original_images)
    base_patches_per_image = num_patches_total // n_images
    extra_patches = num_patches_total % n_images

    print(f"\n📊 Target distribution:")
    print(f"  - Base: {base_patches_per_image} patches per image")
    print(f"  - Extra: {extra_patches} images get +1 patch")
    print(f"  - Total target: {num_patches_total} patches\n")

    # Process each image with its own target
    for p, image in enumerate(original_images):
        # Handle channel dimension if present
        if len(image.shape) == 3:
            img_for_mask = image[:, :, 0]
        else:
            img_for_mask = image

        # Detect circular region
        center_x, center_y, radius, mask = detect_circular_region(img_for_mask)

        height, width = img_for_mask.shape[:2]

        # Calculate valid sampling bounds
        max_i = height - patch_size
        max_j = width - patch_size

        # ✅ CALCULATE TARGET FOR THIS SPECIFIC IMAGE
        target_for_image = base_patches_per_image + (1 if p < extra_patches else 0)

        # ✅ TRACK THIS IMAGE'S PATCHES SEPARATELY (NOT GLOBAL!)
        image_patches_count = 0
        attempts = 0
        max_attempts_per_image = target_for_image * 50  # Safety limit per image

        # ✅ EXTRACT UNTIL THIS IMAGE'S TARGET IS MET
        while image_patches_count < target_for_image and attempts < max_attempts_per_image:
            # Randomly sample position
            i = np.random.randint(0, max_i + 1)
            j = np.random.randint(0, max_j + 1)

            # Check if patch is valid
            is_valid, tissue_ratio = is_patch_valid(img_for_mask, j, i, mask,
                                                    patch_size, min_tissue_ratio)

            if is_valid:
                # Extract patch
                patch = image[i:i + patch_size, j:j + patch_size]

                # Create patch info dictionary
                patch_info = {
                    'p': p,
                    'i': i,
                    'j': j,
                    'tissue_ratio': tissue_ratio
                }

                # Calculate conditions if active
                if 'fvf' in active_conditions:
                    patch_info['fvf_value'] = calculate_fiber_volume_fraction(patch)

                if 'alignment' in active_conditions:
                    patch_info['alignment_value'] = calculate_fiber_alignment_rose(patch)

                if 'thickness' in active_conditions:
                    patch_info['thickness_value'] = calculate_mean_fiber_thickness(patch)

                if 'complexity' in active_conditions:
                    patch_info['complexity_value'] = calculate_texture_complexity(patch)

                all_patches.append(patch_info)
                image_patches_count += 1  # ✅ Count for THIS image only

            attempts += 1

        # Report progress for this image
        success_rate = (image_patches_count / attempts * 100) if attempts > 0 else 0
        status = "✅" if image_patches_count == target_for_image else "⚠️"
        print(f"{status} Image {p + 1:2d}/{n_images}: {image_patches_count:3d}/{target_for_image:3d} patches "
              f"({attempts:4d} attempts, {success_rate:4.1f}% success)")

        # ✅ CRITICAL: NO BREAK HERE - CONTINUE TO ALL IMAGES

    # Shuffle for randomness
    random.shuffle(all_patches)

    print(f"\n✅ Successfully extracted {len(all_patches)} patches from ALL {n_images} images")

    # Print statistics for each condition
    for condition in condition_manager.active_conditions:
        if f'{condition.name}_value' in all_patches[0]:
            values = [p[f'{condition.name}_value'] for p in all_patches]
            print(f"{condition.name}: mean={np.mean(values):.3f}, std={np.std(values):.3f}, "
                  f"range=[{np.min(values):.3f}, {np.max(values):.3f}]")

    return all_patches


def extract_and_save_patch_info_meniscus_3D(original_volumes, patch_size,
                                            num_patches_total, condition_manager,
                                            n_slices=10, min_tissue_ratio=0.95,
                                            overlap=0.50):
    """
    Extract 3D patches from meniscus volumes with tissue validation and condition calculation.
    Uses DISTRIBUTED SAMPLING strategy across all volumes.

    Args:
        original_volumes: List of 3D volumes, each with shape (H, W, D)
        patch_size: Size of patch in XY plane (e.g., 256)
        num_patches_total: Total number of 3D patches to extract
        condition_manager: Manager containing active conditions
        n_slices: Number of depth slices per patch (default: 10)
        min_tissue_ratio: Minimum tissue content required (0.95 = 95%)
        overlap: Overlap ratio for grid extraction (not used in random sampling)

    Returns:
        List of patch_info dictionaries with:
        {'p': volume_idx, 'i': y, 'j': x, 'z': z_start,
         'tissue_ratio': min_across_slices, 'condition_value': ...}
    """
    all_patches = []

    # Get active conditions
    active_conditions = [cond.name for cond in condition_manager.active_conditions]

    print(f"\n{'=' * 60}")
    print(f"EXTRACTING 3D PATCHES")
    print(f"{'=' * 60}")
    print(f"Patch size: ({patch_size}, {patch_size}, {n_slices})")
    print(f"Minimum tissue ratio: {min_tissue_ratio:.0%}")
    print(f"Active conditions: {active_conditions}")
    print(f"Using DISTRIBUTED SAMPLING across ALL volumes")

    # Calculate patches per volume
    n_volumes = len(original_volumes)
    base_patches_per_volume = num_patches_total // n_volumes
    extra_patches = num_patches_total % n_volumes

    print(f"\n📊 Target distribution:")
    print(f"  - Base: {base_patches_per_volume} patches per volume")
    print(f"  - Extra: {extra_patches} volumes get +1 patch")
    print(f"  - Total target: {num_patches_total} patches\n")

    # Process each volume
    for vol_idx, volume in enumerate(original_volumes):
        print(f"\n{'=' * 60}")
        print(f"Processing Volume {vol_idx + 1}/{n_volumes}")
        print(f"{'=' * 60}")

        H, W, D = volume.shape
        print(f"Volume shape: ({H}, {W}, {D})")

        # Detect circular tissue regions for all slices
        slice_regions, common_region = detect_circular_region_3D(volume)
        common_cx, common_cy, common_radius = common_region

        # Create tissue masks for all slices
        print(f"\nCreating tissue masks for {D} slices...")
        masks_3d = []
        for z in range(D):
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.circle(mask, (common_cx, common_cy), common_radius, 255, -1)
            masks_3d.append(mask)

        # Calculate valid sampling bounds
        max_i = H - patch_size
        max_j = W - patch_size

        # Calculate how many different z-positions we can extract from
        # For your data: D=10, n_slices=10, so max_z_positions = 1 (only z=0)
        max_z_positions = D - n_slices + 1

        if max_z_positions <= 0:
            print(f"⚠️  Warning: Volume depth ({D}) < required slices ({n_slices})")
            print(f"   Skipping this volume")
            continue

        print(f"\nSampling space:")
        print(f"  XY range: i ∈ [0, {max_i}], j ∈ [0, {max_j}]")
        print(f"  Z positions available: {max_z_positions}")
        print(f"  (For D={D}, n_slices={n_slices}: can extract from z={list(range(max_z_positions))})")

        # Calculate target for this volume
        target_for_volume = base_patches_per_volume + (1 if vol_idx < extra_patches else 0)

        print(f"\n🎯 Target patches for this volume: {target_for_volume}")

        # Track extraction progress
        volume_patches_count = 0
        attempts = 0
        max_attempts_per_volume = target_for_volume * 100  # Increased safety limit

        # Extract patches until target is met
        while volume_patches_count < target_for_volume and attempts < max_attempts_per_volume:
            # Randomly sample XY position
            i = np.random.randint(0, max_i + 1)
            j = np.random.randint(0, max_j + 1)

            # For now, z is always 0 (since we have exactly 10 slices)
            # If you had more slices, you could randomly sample z too:
            # z = np.random.randint(0, max_z_positions)
            z = np.random.randint(0, max_z_positions)

            # Check if 3D patch is valid (all slices meet tissue requirement)
            is_valid, min_tissue_ratio_value = is_patch_valid_3D(
                volume, j, i, z, masks_3d, patch_size, n_slices, min_tissue_ratio
            )

            if is_valid:
                # Extract 3D patch
                patch_3d = volume[i:i + patch_size, j:j + patch_size, z:z + n_slices]

                # Create patch info dictionary
                patch_info = {
                    'p': vol_idx,
                    'i': i,
                    'j': j,
                    'z': z,  # NEW: Z-position in volume
                    'tissue_ratio': min_tissue_ratio_value
                }

                # Calculate 3D conditions (will be added in Task 1.3)
                # Calculate 3D conditions if active
                if 'fvf' in active_conditions:
                    patch_info['fvf_value'] = calculate_fiber_volume_fraction_3D(patch_3d)

                if 'alignment' in active_conditions:
                    patch_info['alignment_value'] = calculate_fiber_alignment_rose_3D(patch_3d)

                if 'thickness' in active_conditions:
                    patch_info['thickness_value'] = calculate_mean_fiber_thickness_3D(patch_3d)

                if 'complexity' in active_conditions:
                    patch_info['complexity_value'] = calculate_texture_complexity_3D(patch_3d)

                all_patches.append(patch_info)
                volume_patches_count += 1

                # Progress update every 10% of target
                if volume_patches_count % max(1, target_for_volume // 10) == 0:
                    progress_pct = (volume_patches_count / target_for_volume) * 100
                    print(f"  Progress: {volume_patches_count}/{target_for_volume} "
                          f"({progress_pct:.0f}%)")

            attempts += 1

        # Report results for this volume
        success_rate = (volume_patches_count / attempts * 100) if attempts > 0 else 0
        status = "✅" if volume_patches_count == target_for_volume else "⚠️"

        print(f"\n{status} Volume {vol_idx + 1} Summary:")
        print(f"  Extracted: {volume_patches_count}/{target_for_volume} patches")
        print(f"  Attempts: {attempts}")
        print(f"  Success rate: {success_rate:.1f}%")

    # Shuffle for randomness
    random.shuffle(all_patches)

    print(f"\n{'=' * 60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"✅ Successfully extracted {len(all_patches)} 3D patches from {n_volumes} volume(s)")

    # Print statistics summary
    if all_patches:
        tissue_ratios = [p['tissue_ratio'] for p in all_patches]
        print(f"\nTissue Ratio Statistics:")
        print(f"  Mean: {np.mean(tissue_ratios):.3f}")
        print(f"  Std:  {np.std(tissue_ratios):.3f}")
        print(f"  Range: [{np.min(tissue_ratios):.3f}, {np.max(tissue_ratios):.3f}]")

    return all_patches


def categorize_patches_meniscus(patch_info, n_classes_per_condition, condition_manager):
    """
    Categorize patches based on active meniscus conditions.
    Each active condition gets divided into n_classes using quantiles.

    This replaces the porosity-based categorization with multi-condition support.

    Args:
        patch_info: List of patch dictionaries with condition values
        n_classes_per_condition: Either:
            - Integer: Same number of classes for all conditions (e.g., 10)
            - Dict: Different classes per condition (e.g., {'fvf': 10, 'alignment': 5})
        condition_manager: Manager with active conditions

    Returns:
        patch_info with added class labels for each active condition
    """
    active_conditions = [cond.name for cond in condition_manager.active_conditions]

    print(f"\nCategorizing patches for conditions: {active_conditions}")

    # ✅ NEW: Handle both integer and dict input
    if isinstance(n_classes_per_condition, int):
        # Legacy mode: same classes for all conditions
        n_classes_dict = {cond: n_classes_per_condition for cond in active_conditions}
        print(f"Using {n_classes_per_condition} classes for all conditions")
    elif isinstance(n_classes_per_condition, dict):
        # Flexible mode: different classes per condition
        n_classes_dict = n_classes_per_condition
        print(f"Using custom class counts: {n_classes_dict}")
    else:
        raise ValueError(f"n_classes_per_condition must be int or dict, got {type(n_classes_per_condition)}")

    for condition_name in active_conditions:
        if condition_name in ['fvf', 'alignment', 'thickness', 'complexity']:
            # ✅ NEW: Get condition-specific class count
            n_classes = n_classes_dict.get(condition_name, 10)  # Default to 10 if not specified

            # Collect all values for this condition
            values = [patch[f'{condition_name}_value'] for patch in patch_info]

            if not values:
                continue

            # Calculate quantile boundaries for equal distribution
            min_val, max_val = min(values), max(values)

            # Handle case where all values are the same
            if min_val == max_val:
                for patch in patch_info:
                    patch[f'{condition_name}_class'] = 0
                print(f"  {condition_name}: All values identical ({min_val:.3f}), assigned to class 0")
                continue

            # ✅ MODIFIED: Use condition-specific n_classes
            quantiles = np.linspace(min_val, max_val, n_classes + 1)

            # Assign class to each patch
            for patch in patch_info:
                value = patch[f'{condition_name}_value']
                # Find which quantile bin this value falls into
                for i in range(len(quantiles) - 1):
                    if i == len(quantiles) - 2:  # Last bin includes max value
                        if quantiles[i] <= value <= quantiles[i + 1]:
                            patch[f'{condition_name}_class'] = i
                            break
                    else:
                        if quantiles[i] <= value < quantiles[i + 1]:
                            patch[f'{condition_name}_class'] = i
                            break

            # Print distribution info
            class_counts = {}
            for patch in patch_info:
                cls = patch.get(f'{condition_name}_class', 0)
                class_counts[cls] = class_counts.get(cls, 0) + 1

            # ✅ MODIFIED: Use condition-specific n_classes in print
            print(f"  {condition_name}: Range [{min_val:.3f}, {max_val:.3f}] divided into {n_classes} classes")
            print(f"    Class distribution: {dict(sorted(class_counts.items()))}")

    return patch_info


def read_patch(original_images, patch_info, patch_size):
    """
    Read a patch from original images.

    Args:
        original_images: List of original images
        patch_info: Dictionary with patch information
        patch_size: Size of the patch to extract

    Returns:
        The image patch
    """
    p, i, j = patch_info['p'], patch_info['i'], patch_info['j']
    patch = original_images[p][i:i + patch_size, j:j + patch_size]
    return patch


def read_patch_3D(original_volumes, patch_info, patch_size, n_slices=10):
    """
    Read a 3D patch from original volumes.

    Args:
        original_volumes: List of 3D volumes
        patch_info: Dictionary with patch information (p, i, j, z)
        patch_size: Size of the patch in XY plane
        n_slices: Number of depth slices

    Returns:
        3D patch of shape (patch_size, patch_size, n_slices)
    """
    p = patch_info['p']
    i = patch_info['i']
    j = patch_info['j']
    z = patch_info.get('z', 0)  # Default to 0 if not specified

    patch_3d = original_volumes[p][i:i + patch_size, j:j + patch_size, z:z + n_slices]

    return patch_3d




def balance_dataset(patch_info, desired_patches_per_class, condition_manager):
    """
    Balance dataset using simple repetition for underrepresented classes.

    Args:
        patch_info: List of patch information
        desired_patches_per_class: Target number of patches per class
        condition_manager: Manager containing active conditions

    Returns:
        Balanced patch info list
    """
    balanced_patch_info = []

    # For simplicity, balance based on the first active condition
    # In practice, you might want to balance across multiple conditions
    if condition_manager.active_conditions:
        first_condition = condition_manager.active_conditions[0]
        class_key = f'{first_condition.name}_class'

        if class_key in patch_info[0]:
            # Group patches by class
            classes = {}
            for patch in patch_info:
                class_idx = patch[class_key]
                if class_idx not in classes:
                    classes[class_idx] = []
                classes[class_idx].append(patch)

            # Balance each class
            for class_idx, class_patches in classes.items():
                n_patches = len(class_patches)

                if n_patches >= desired_patches_per_class:
                    # Sample if we have too many
                    selected = random.sample(class_patches, desired_patches_per_class)
                else:
                    # Repeat if we have too few
                    selected = class_patches.copy()
                    while len(selected) < desired_patches_per_class:
                        selected.append(random.choice(class_patches))

                balanced_patch_info.extend(selected)
                print(f"Class {class_idx}: {n_patches} -> {len(selected)} patches")
    else:
        # No conditions, just return original
        balanced_patch_info = patch_info

    random.shuffle(balanced_patch_info)
    return balanced_patch_info