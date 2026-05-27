"""
inference.py — Inference script for the Meniscus cGAN
======================================================
Automatically downloads the trained generator from Zenodo,
then generates a 3x3 grid of 3D meniscus tissue volumes
covering the full trained FVF × Alignment condition space.

Paper:
  "Conditional Generation of 3D Meniscus Microstructures from Micro-CT Images"
  Sadeghkhani et al., Biomechanics and Modeling in Mechanobiology, 2026

Zenodo model record:
  https://doi.org/10.5281/zenodo.20399140

Usage:
  python inference.py                         # 3x3 grid (default)
  python inference.py --grid_n 4              # 4x4 grid
  python inference.py --fvf 0.24 --mrl 0.30  # single volume
  python inference.py --seed 123              # reproducible noise
  python inference.py --no_vti               # skip VTI, PNG only
  python inference.py --model_dir ./my_model  # skip download, use local model

Outputs (saved to ./generated_volumes/):
  grid_overview.png        — overview figure of all generated volumes
  volumes/fvf*_mrl*.npy   — 3D arrays (512, 512, 32), uint8 [0-255]
  volumes/fvf*_mrl*.vti   — VTK ImageData files for ParaView
"""

import os
import sys
import json
import struct
import base64
import zipfile
import argparse
import urllib.request

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import filters
import cv2


# ─────────────────────────────────────────────────────────────────────────────
# 0.  GPU setup  (must happen before any TF import)
# ─────────────────────────────────────────────────────────────────────────────

def _setup_gpu():
    """Enable GPU memory growth; fall back to CPU silently if it fails."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  GPU : {gpus[0].name}  (memory growth enabled)")
        else:
            print("  GPU : none found — running on CPU")
    except RuntimeError:
        # Memory growth must be set before GPUs initialise
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("  GPU : memory-growth failed — falling back to CPU")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Zenodo download
# ─────────────────────────────────────────────────────────────────────────────

ZENODO_RECORD_ID = "20399140"
ZENODO_API       = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
ZIP_FILENAME     = "trained_model.zip"
MODEL_SUBPATH    = os.path.join("trained_model", "G.h5")
META_SUBPATH     = os.path.join("trained_model", "metadata", "conditions.json")


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb  = downloaded / 1024 / 1024
        tot = total_size / 1024 / 1024
        print(f"\r  Downloading … {mb:.1f} / {tot:.1f} MB  ({pct:.0f}%)",
              end='', flush=True)


def download_model(dest_dir="."):
    """
    Download and extract trained_model.zip from Zenodo if not already present.

    Returns:
        model_path : path to G.h5
        meta_path  : path to conditions.json
    """
    model_path = os.path.join(dest_dir, MODEL_SUBPATH)
    meta_path  = os.path.join(dest_dir, META_SUBPATH)

    # Already extracted?
    if os.path.isfile(model_path) and os.path.isfile(meta_path):
        print(f"  Model already present at: {model_path}")
        return model_path, meta_path

    zip_path = os.path.join(dest_dir, ZIP_FILENAME)

    # Already downloaded but not extracted?
    if not os.path.isfile(zip_path):
        print(f"\n  Fetching file list from Zenodo record {ZENODO_RECORD_ID} …")
        try:
            with urllib.request.urlopen(ZENODO_API, timeout=30) as resp:
                record = json.loads(resp.read().decode())
        except Exception as e:
            sys.exit(f"\n[ERROR] Could not reach Zenodo API: {e}\n"
                     f"  Check your internet connection or download manually:\n"
                     f"  https://doi.org/10.5281/zenodo.{ZENODO_RECORD_ID}\n")

        # Find the zip file URL
        file_url = None
        for f in record.get("files", []):
            if f.get("key", "") == ZIP_FILENAME:
                file_url = f["links"]["self"]
                break

        if file_url is None:
            sys.exit(f"\n[ERROR] '{ZIP_FILENAME}' not found in Zenodo record.\n"
                     f"  Available files: {[f['key'] for f in record.get('files', [])]}\n")

        print(f"  Downloading {ZIP_FILENAME} from Zenodo …")
        os.makedirs(dest_dir, exist_ok=True)
        urllib.request.urlretrieve(file_url, zip_path, reporthook=_progress_hook)
        print()  # newline after progress

    # Extract
    print(f"  Extracting {zip_path} …")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest_dir)

    if not os.path.isfile(model_path):
        sys.exit(f"\n[ERROR] Expected model at '{model_path}' after extraction.\n"
                 f"  Check zip contents.\n")

    print(f"  Model ready: {model_path}")
    return model_path, meta_path


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Condition calculation  (mirrors data_processing.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def _calc_fvf_slice(gray_slice):
    """Fiber Volume Fraction for one 2D slice via Otsu thresholding."""
    blurred = cv2.GaussianBlur(gray_slice, (5, 5), 1)
    thresh  = filters.threshold_otsu(blurred)
    return float(np.clip(np.sum(blurred > thresh) / blurred.size, 0, 1))


def _calc_mrl_slice(gray_slice):
    """Mean Resultant Length (fiber alignment) for one 2D slice."""
    img = gray_slice.astype(np.float32) / 255.0
    gx  = ndimage.sobel(img, axis=1, mode='reflect')
    gy  = ndimage.sobel(img, axis=0, mode='reflect')
    ori = np.arctan2(gy, gx) * 180.0 / np.pi % 180.0
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mask = mag > np.percentile(mag, 10)
    bins = np.linspace(0, 180, 37)
    hist, _ = np.histogram(ori[mask], bins=bins, weights=mag[mask])
    if np.sum(hist) > 0:
        hist /= np.sum(hist)
    bc  = (bins[:-1] + bins[1:]) / 2 * np.pi / 180.0
    mrl = np.sqrt(np.sum(hist * np.cos(2 * bc)) ** 2 +
                  np.sum(hist * np.sin(2 * bc)) ** 2)
    return float(np.clip(mrl, 0, 1))


def measure_volume(volume):
    """
    Measure actual FVF and MRL of a generated volume
    by averaging per-slice values across all depth slices.

    Args:
        volume : np.ndarray (H, W, D), uint8

    Returns:
        fvf, mrl : float
    """
    n_slices = volume.shape[2]
    fvf = float(np.mean([_calc_fvf_slice(volume[:, :, z]) for z in range(n_slices)]))
    mrl = float(np.mean([_calc_mrl_slice(volume[:, :, z]) for z in range(n_slices)]))
    return fvf, mrl


# ─────────────────────────────────────────────────────────────────────────────
# 3.  VTI writer
# ─────────────────────────────────────────────────────────────────────────────

def save_vti(volume, filepath, voxel_size_um=0.6):
    """
    Save a 3D uint8 volume as a VTK ImageData (.vti) file readable by ParaView.

    Args:
        volume      : np.ndarray (H, W, D), uint8
        filepath    : output .vti path
        voxel_size_um : isotropic voxel spacing in micrometres (default 0.6 µm)
    """
    H, W, D  = volume.shape
    data     = np.ascontiguousarray(volume.transpose(2, 0, 1))  # VTK: (D, H, W)
    raw      = data.tobytes()
    header   = struct.pack('<I', len(raw))
    encoded  = base64.b64encode(header + raw).decode('ascii')
    s        = voxel_size_um

    vti = f"""<?xml version="1.0"?>
<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian" header_type="UInt32">
  <ImageData WholeExtent="0 {W-1} 0 {H-1} 0 {D-1}" Origin="0 0 0" Spacing="{s} {s} {s}">
    <Piece Extent="0 {W-1} 0 {H-1} 0 {D-1}">
      <PointData Scalars="ImageScalars">
        <DataArray type="UInt8" Name="ImageScalars" format="binary" NumberOfComponents="1">
          {encoded}
        </DataArray>
      </PointData>
    </Piece>
  </ImageData>
</VTKFile>
"""
    with open(filepath, 'w') as f:
        f.write(vti)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_volume(generator, noise, fvf, mrl, latent_dim):
    """
    Single forward pass through the generator.

    Args:
        generator : loaded Keras model
        noise     : fixed np.ndarray (1, latent_dim), float32
        fvf       : target FVF  float [0, 1]
        mrl       : target MRL  float [0, 1]
        latent_dim: int

    Returns:
        volume : np.ndarray (H, W, D), uint8 [0, 255]
    """
    fvf_in = np.array([[fvf]], dtype=np.float32)
    mrl_in = np.array([[mrl]], dtype=np.float32)
    raw    = generator.predict([noise, fvf_in, mrl_in], verbose=0)  # (1,H,W,D)
    return ((raw[0] + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)


def warn_oob(value, vmin, vmax, name):
    """Print a warning if a requested condition is outside the trained range."""
    if not (vmin <= value <= vmax):
        print(f"  [WARNING] {name}={value:.4f} is outside the trained range "
              f"[{vmin:.4f}, {vmax:.4f}]. Results may be unreliable.")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Grid mode  (main use-case)
# ─────────────────────────────────────────────────────────────────────────────

def run_grid(generator, fixed_noise, fvf_vals, mrl_vals,
             output_dir, latent_dim, save_vti_flag, voxel_size_um):
    """
    Generate a grid of volumes (rows = MRL, columns = FVF).
    All volumes share the same noise vector so differences are
    purely due to the conditioning parameters.

    Saves:
      - one .npy per volume
      - one .vti per volume  (if save_vti_flag)
      - grid_overview.png    — middle-slice mosaic of all volumes
    """
    vol_dir = os.path.join(output_dir, "volumes")
    os.makedirs(vol_dir, exist_ok=True)

    n_fvf = len(fvf_vals)
    n_mrl = len(mrl_vals)
    total = n_fvf * n_mrl

    # Store middle slices for the overview figure
    mid_slices   = np.zeros((n_mrl, n_fvf, 512, 512), dtype=np.uint8)
    actual_fvfs  = np.zeros((n_mrl, n_fvf))
    actual_mrls  = np.zeros((n_mrl, n_fvf))

    count = 0
    for row, mrl_t in enumerate(mrl_vals):
        for col, fvf_t in enumerate(fvf_vals):
            count += 1
            print(f"\n  [{count}/{total}]  Target FVF={fvf_t:.3f}  MRL={mrl_t:.3f}")

            vol = generate_volume(generator, fixed_noise, fvf_t, mrl_t, latent_dim)

            # Measure actual conditions
            a_fvf, a_mrl = measure_volume(vol)
            actual_fvfs[row, col] = a_fvf
            actual_mrls[row, col] = a_mrl
            print(f"         Actual  FVF={a_fvf:.3f}  MRL={a_mrl:.3f}")

            # File stem
            stem = f"fvf{fvf_t:.3f}_mrl{mrl_t:.3f}"

            # Save NumPy array
            npy_path = os.path.join(vol_dir, f"{stem}.npy")
            np.save(npy_path, vol)
            print(f"         Saved:  {os.path.basename(npy_path)}")

            # Save VTI
            if save_vti_flag:
                vti_path = os.path.join(vol_dir, f"{stem}.vti")
                save_vti(vol, vti_path, voxel_size_um)
                print(f"         Saved:  {os.path.basename(vti_path)}")

            # Store middle slice for overview
            mid = vol.shape[2] // 2
            mid_slices[row, col] = vol[:, :, mid]

    # ── Overview figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        n_mrl, n_fvf,
        figsize=(n_fvf * 3.0, n_mrl * 3.2),
        gridspec_kw={"hspace": 0.35, "wspace": 0.12}
    )
    if n_mrl == 1 and n_fvf == 1:
        axes = np.array([[axes]])
    elif n_mrl == 1:
        axes = axes[np.newaxis, :]
    elif n_fvf == 1:
        axes = axes[:, np.newaxis]

    for row in range(n_mrl):
        for col in range(n_fvf):
            ax = axes[row, col]
            ax.imshow(mid_slices[row, col], cmap='gray', vmin=0, vmax=255)
            ax.set_title(
                f"FVF {actual_fvfs[row,col]:.3f}\nMRL {actual_mrls[row,col]:.3f}",
                fontsize=8, pad=3
            )
            ax.axis('off')

    # Column labels (target FVF)
    for col, fvf_t in enumerate(fvf_vals):
        axes[0, col].set_title(
            f"Target FVF={fvf_t:.3f}\n"
            f"Actual FVF={actual_fvfs[0,col]:.3f} MRL={actual_mrls[0,col]:.3f}",
            fontsize=8, pad=3
        )

    # Row labels (target MRL)
    for row, mrl_t in enumerate(mrl_vals):
        axes[row, 0].set_ylabel(
            f"Target\nMRL={mrl_t:.3f}", fontsize=8, rotation=90, labelpad=6
        )

    fig.suptitle(
        "Generated Meniscus Volumes — FVF × MRL Grid\n"
        "(middle slice shown, same noise vector for all)",
        fontsize=11, y=1.01
    )

    overview_path = os.path.join(output_dir, "grid_overview.png")
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Overview figure: {overview_path}")

    return actual_fvfs, actual_mrls


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Single-volume mode
# ─────────────────────────────────────────────────────────────────────────────

def run_single(generator, fixed_noise, fvf_t, mrl_t,
               output_dir, latent_dim, save_vti_flag, voxel_size_um):
    """Generate one volume at the requested FVF and MRL."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  Target FVF={fvf_t:.4f}  MRL={mrl_t:.4f}")
    vol   = generate_volume(generator, fixed_noise, fvf_t, mrl_t, latent_dim)
    a_fvf, a_mrl = measure_volume(vol)
    print(f"  Actual FVF={a_fvf:.4f}  MRL={a_mrl:.4f}")

    stem    = f"fvf{fvf_t:.3f}_mrl{mrl_t:.3f}"
    npy_out = os.path.join(output_dir, f"{stem}.npy")
    np.save(npy_out, vol)
    print(f"  Saved: {npy_out}")

    if save_vti_flag:
        vti_out = os.path.join(output_dir, f"{stem}.vti")
        save_vti(vol, vti_out, voxel_size_um)
        print(f"  Saved: {vti_out}")

    # Quick overview: 8 evenly spaced slices
    n_slices = vol.shape[2]
    idxs     = np.linspace(0, n_slices - 1, min(8, n_slices), dtype=int)
    fig, axes = plt.subplots(1, len(idxs), figsize=(len(idxs) * 2.4, 2.8))
    if len(idxs) == 1:
        axes = [axes]
    for ax, idx in zip(axes, idxs):
        ax.imshow(vol[:, :, idx], cmap='gray', vmin=0, vmax=255)
        ax.set_title(f"z={idx}", fontsize=8)
        ax.axis('off')
    fig.suptitle(
        f"FVF target={fvf_t:.3f} actual={a_fvf:.3f} | "
        f"MRL target={mrl_t:.3f} actual={a_mrl:.3f}",
        fontsize=10
    )
    plt.tight_layout()
    fig_out = os.path.join(output_dir, f"{stem}_overview.png")
    plt.savefig(fig_out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Overview: {fig_out}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Meniscus cGAN inference — auto-downloads model from Zenodo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Download / paths
    p.add_argument("--model_dir", type=str, default=".",
                   help="Directory to download/find trained_model/ folder. "
                        "If trained_model/G.h5 already exists here, "
                        "the download step is skipped.")
    p.add_argument("--output_dir", type=str, default="./generated_volumes",
                   help="Directory to save generated volumes and figures.")

    # Mode
    p.add_argument("--mode", type=str, default="grid",
                   choices=["grid", "single"],
                   help="grid: generate FVF × MRL grid | "
                        "single: one volume at --fvf / --mrl")

    # Grid options
    p.add_argument("--grid_n", type=int, default=3,
                   help="[grid] Steps along each axis → grid_n² volumes total.")

    # Single options
    p.add_argument("--fvf", type=float, default=0.24,
                   help="[single] Target Fiber Volume Fraction.")
    p.add_argument("--mrl", type=float, default=0.30,
                   help="[single] Target fiber alignment (MRL).")

    # Output options
    p.add_argument("--no_vti", action="store_true",
                   help="Skip saving .vti files (saves .npy and PNG only).")
    p.add_argument("--voxel_size_um", type=float, default=0.6,
                   help="Voxel spacing in µm written into the VTI header.")

    # Model options
    p.add_argument("--latent_dim", type=int, default=100,
                   help="Latent noise dimension (must match training; 100 for the "
                        "released model).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for the fixed noise vector.")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  Meniscus cGAN — Inference")
    print("  Zenodo: https://doi.org/10.5281/zenodo.20399140")
    print("=" * 60)

    # ── GPU setup ─────────────────────────────────────────────────────────────
    print("\n[1] Setting up GPU …")
    _setup_gpu()

    # ── Download model from Zenodo ────────────────────────────────────────────
    print("\n[2] Checking / downloading model from Zenodo …")
    model_path, meta_path = download_model(dest_dir=args.model_dir)

    # ── Load conditions metadata ──────────────────────────────────────────────
    print("\n[3] Loading conditions metadata …")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    fvf_min = meta["condition_stats"]["fvf"]["min"]
    fvf_max = meta["condition_stats"]["fvf"]["max"]
    mrl_min = meta["condition_stats"]["alignment"]["min"]
    mrl_max = meta["condition_stats"]["alignment"]["max"]

    print(f"  Trained FVF range : [{fvf_min:.4f}, {fvf_max:.4f}]")
    print(f"  Trained MRL range : [{mrl_min:.4f}, {mrl_max:.4f}]")

    # ── Load generator ────────────────────────────────────────────────────────
    print("\n[4] Loading generator …")
    from tensorflow.keras.models import load_model
    generator = load_model(model_path, compile=False)
    print(f"  Output shape : {generator.output_shape}")
    print(f"  Parameters   : {generator.count_params():,}")

    # ── Fixed noise vector ────────────────────────────────────────────────────
    np.random.seed(args.seed)
    fixed_noise = np.random.randn(1, args.latent_dim).astype(np.float32)
    print(f"\n  Fixed noise vector (seed={args.seed})  "
          f"mean={fixed_noise.mean():.4f}  std={fixed_noise.std():.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    save_vti_flag = not args.no_vti

    # ── Run ───────────────────────────────────────────────────────────────────
    if args.mode == "grid":
        fvf_vals = np.linspace(fvf_min, fvf_max, args.grid_n).tolist()
        mrl_vals = np.linspace(mrl_min, mrl_max, args.grid_n).tolist()

        print(f"\n[5] Generating {args.grid_n}×{args.grid_n} grid …")
        print(f"  FVF values : {[f'{v:.3f}' for v in fvf_vals]}")
        print(f"  MRL values : {[f'{v:.3f}' for v in mrl_vals]}")

        run_grid(
            generator, fixed_noise,
            fvf_vals, mrl_vals,
            args.output_dir, args.latent_dim,
            save_vti_flag, args.voxel_size_um
        )

    elif args.mode == "single":
        warn_oob(args.fvf, fvf_min, fvf_max, "FVF")
        warn_oob(args.mrl, mrl_min, mrl_max, "MRL")

        print(f"\n[5] Generating single volume …")
        run_single(
            generator, fixed_noise,
            args.fvf, args.mrl,
            args.output_dir, args.latent_dim,
            save_vti_flag, args.voxel_size_um
        )

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Done. Results saved to:", os.path.abspath(args.output_dir))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
