#!/usr/bin/env python3
"""
Deep-dive analysis: follow up on key findings from carrier analysis.

Key findings to investigate:
1. Autocorrelation shows 2x2 tiling pattern at exact image half-dimensions
2. Phase is deterministic at low frequencies, random at high frequencies
3. Spatial cross-sample correlation is 0.97 (very high common structure)
4. Per-image variation is random (independent)
5. Visible watermark is identical across all images (r=0.9999)
6. Carrier energy is detectable on dark channels but swamped on bright channels
"""

import cv2
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import sys

BLACK_DIR = Path("test-images/gemini-3.1-pro/2400x1792/pure-black")
COLOR_DIR = Path("test-images/gemini-3.1-pro/2400x1792")

def load_images(directory, max_n=None):
    images = []
    files = sorted(directory.glob("*.png"))
    if max_n:
        files = files[:max_n]
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is not None:
            images.append((f.name, img.astype(np.float64) / 255.0))
    return images

def fft2_channel(channel):
    f = np.fft.fft2(channel)
    return np.fft.fftshift(f)

# ---- Analysis 1: Spatial pattern comparison ----
def analyze_spatial_common_pattern(images):
    """The spatial cross-sample correlation is 0.97. What's the common part?"""
    print("=" * 70)
    print("DEEP 1: SPATIAL COMMON PATTERN (avg of 30 images)")
    print("=" * 70)

    h, w = images[0][1].shape[:2]
    avg_img = np.zeros((h, w, 3))

    for name, img in images:
        avg_img += img
    avg_img /= len(images)

    # The average image IS the common spatial structure
    print(f"\nAverage spatial pattern (common carrier component):")
    for c, ch_name in enumerate(["Blue", "Green", "Red"]):
        ch = avg_img[:,:,c]
        print(f"\n{ch_name} channel:")
        print(f"  Mean: {ch.mean():.6f}, Std: {ch.std():.6f}")
        print(f"  Min: {ch.min():.6f}, Max: {ch.max():.6f}")

        # Check spatial smoothness - is it a smooth pattern or noisy?
        # Gradient magnitude
        gy, gx = np.gradient(ch)
        grad_mag = np.sqrt(gx**2 + gy**2)
        print(f"  Gradient magnitude: mean={grad_mag.mean():.6f}, max={grad_mag.max():.6f}")

    # Check if the average has a visible structure
    # Save it for visual inspection
    avg_save = (avg_img * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite("analysis_avg_spatial_pattern.png", avg_save)
    print(f"\nSaved average spatial pattern to analysis_avg_spatial_pattern.png")

    # Also look at per-pixel variance across samples
    var_img = np.zeros((h, w, 3))
    for name, img in images:
        var_img += (img - avg_img)**2
    var_img /= len(images)

    print(f"\nPer-pixel variance across samples:")
    for c, ch_name in enumerate(["Blue", "Green", "Red"]):
        ch = var_img[:,:,c]
        print(f"  {ch_name}: mean_var={ch.mean():.8f}, max_var={ch.max():.8f}")
        print(f"    Std of variance map: {ch.std():.8f}")

    # Is variance uniform across the image, or concentrated somewhere?
    cv2.imwrite("analysis_spatial_variance.png", (np.sqrt(var_img) * 255 * 100).clip(0, 255).astype(np.uint8))
    print(f"  Saved variance map to analysis_spatial_variance.png")

    return avg_img, var_img

# ---- Analysis 2: What exactly is the 2x2 structure? ----
def analyze_quadrant_symmetry(images):
    """Autocorrelation showed peaks at half-dimensions. Is the image truly symmetric?"""
    print("\n" + "=" * 70)
    print("DEEP 2: QUADRANT SYMMETRY ANALYSIS")
    print("=" * 70)

    name, img = images[0]
    h, w = img.shape[:2]

    # Split into quadrants
    top_left = img[:h//2, :w//2]
    top_right = img[:h//2, w//2:]
    bot_left = img[h//2:, :w//2]
    bot_right = img[h//2:, w//2:]

    for c, ch_name in enumerate(["Blue", "Green", "Red"]):
        tl = top_left[:,:,c].flatten()
        tr = top_right[:,:,c].flatten()
        bl = bot_left[:,:,c].flatten()
        br = bot_right[:,:,c].flatten()

        print(f"\n{ch_name} channel quadrant correlations:")
        for qa_name, qa in [("TL", tl), ("TR", tr), ("BL", bl), ("BR", br)]:
            for qb_name, qb in [("TL", tl), ("TR", tr), ("BL", bl), ("BR", br)]:
                if qa_name < qb_name:
                    r, _ = pearsonr(qa, qb)
                    print(f"  {qa_name}-{qb_name}: {r:.6f}")

    # Also check if there's exact mirror symmetry
    for c, ch_name in enumerate(["Blue", "Green", "Red"]):
        ch = img[:,:,c]
        # Horizontal flip
        h_flipped = np.flip(ch, axis=1)
        r_h, _ = pearsonr(ch.flatten(), h_flipped.flatten())
        # Vertical flip
        v_flipped = np.flip(ch, axis=0)
        r_v, _ = pearsonr(ch.flatten(), v_flipped.flatten())
        # 180 rotation
        rotated = np.flip(np.flip(ch, axis=0), axis=1)
        r_r, _ = pearsonr(ch.flatten(), rotated.flatten())

        print(f"\n{ch_name} symmetry:")
        print(f"  Horizontal flip correlation: {r_h:.6f}")
        print(f"  Vertical flip correlation:   {r_v:.6f}")
        print(f"  180° rotation correlation:   {r_r:.6f}")

# ---- Analysis 3: Phase determinism gradient ----
def analyze_phase_determinism(images):
    """Phase is deterministic at low freq, random at high freq. Where's the boundary?"""
    print("\n" + "=" * 70)
    print("DEEP 3: PHASE DETERMINISM BOUNDARY")
    print("=" * 70)

    h, w = images[0][1].shape[:2]
    n = len(images)

    all_phases = np.zeros((n, h, w, 3))
    for i, (name, img) in enumerate(images):
        for c in range(3):
            all_phases[i,:,:,c] = np.angle(fft2_channel(img[:,:,c]))

    # For each channel, find the radius where phase goes from deterministic to random
    for c, ch_name in enumerate(["Blue", "Green", "Red"]):
        phases = all_phases[:,:,:,c]
        phase_std = phases.std(axis=0)

        # Compute radial profile of phase std
        center_y, center_x = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)

        max_r = min(center_y, center_x) - 1
        radial_std = np.zeros(max_r)
        for ri in range(max_r):
            mask = (r == ri)
            if mask.any():
                radial_std[ri] = phase_std[mask].mean()

        # Find the radius where phase_std crosses thresholds
        print(f"\n{ch_name} channel - phase determinism:")

        # Phase is "deterministic" if std < 0.3 (highly consistent across samples)
        # Phase is "semi-deterministic" if std < 1.0
        # Phase is "random" if std > 1.4 (close to pi/sqrt(3) = 1.814)
        det_boundary = np.searchsorted(radial_std, 0.3)
        semi_boundary = np.searchsorted(radial_std, 1.0)
        random_boundary = np.searchsorted(radial_std, 1.4)

        print(f"  Deterministic (std<0.3) up to radius: {det_boundary}")
        print(f"  Semi-deterministic (std<1.0) up to radius: {semi_boundary}")
        print(f"  Fully random (std>1.4) from radius: {random_boundary}")
        print(f"  Max radius: {max_r}")
        print(f"  Deterministic fraction: {det_boundary/max_r:.4f}")
        print(f"  Semi-deterministic fraction: {semi_boundary/max_r:.4f}")

        # Print detailed profile at key points
        print(f"  Detailed profile:")
        for ri in range(0, min(max_r, 200), 10):
            print(f"    r={ri:4d}: phase_std={radial_std[ri]:.4f}")
        for ri in range(200, max_r, 50):
            print(f"    r={ri:4d}: phase_std={radial_std[ri]:.4f}")

        # How many frequency bins are in the deterministic zone?
        det_bins = (r < det_boundary).sum()
        total_bins = h * w
        print(f"  Deterministic bins: {det_bins} / {total_bins} ({det_bins/total_bins:.4f})")

# ---- Analysis 4: Carrier magnitude template ----
def analyze_carrier_template(images):
    """Characterize the average magnitude spectrum precisely."""
    print("\n" + "=" * 70)
    print("DEEP 4: CARRIER MAGNITUDE TEMPLATE")
    print("=" * 70)

    h, w = images[0][1].shape[:2]
    n = len(images)

    # Average magnitude spectrum
    avg_mag = np.zeros((h, w, 3))
    for name, img in images:
        for c in range(3):
            avg_mag[:,:,c] += np.abs(fft2_channel(img[:,:,c]))
    avg_mag /= n

    # Compute average magnitude in different frequency bands
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # Frequency bands
    bands = [
        ("DC", 0, 3),
        ("Very low", 3, 10),
        ("Low", 10, 30),
        ("Low-mid", 30, 80),
        ("Mid", 80, 200),
        ("Mid-high", 200, 400),
        ("High", 400, 600),
        ("Very high", 600, 800),
        ("Ultra high", 800, 900),
    ]

    print(f"\nAverage magnitude by frequency band (Green channel):")
    for band_name, r_min, r_max in bands:
        mask = (r >= r_min) & (r < r_max)
        if not mask.any():
            continue
        mag_values = avg_mag[:,:,1][mask]
        print(f"  {band_name:12s} (r={r_min:4d}-{r_max:4d}): "
              f"mean={mag_values.mean():.4f}, std={mag_values.std():.4f}, "
              f"min={mag_values.min():.4f}, max={mag_values.max():.4f}, "
              f"bins={mask.sum()}")

    # Check if magnitude follows a specific distribution (e.g., 1/f)
    print(f"\n1/f characteristic check (Green channel):")
    radial_mag = np.zeros(min(center_y, center_x))
    for ri in range(len(radial_mag)):
        mask = (r >= ri) & (r < ri + 1)
        if mask.any():
            radial_mag[ri] = avg_mag[:,:,1][mask].mean()

    # If magnitude ~ 1/f, then log(magnitude) ~ -log(frequency)
    # Fit a line to log-log plot
    freqs = np.arange(1, len(radial_mag))
    mags = radial_mag[1:]
    valid = mags > 0
    if valid.sum() > 10:
        log_f = np.log10(freqs[valid])
        log_m = np.log10(mags[valid])
        # Linear fit
        coeffs = np.polyfit(log_f, log_m, 1)
        print(f"  Log-log slope: {coeffs[0]:.4f} (1/f noise has slope ≈ -1.0)")
        print(f"  Intercept: {coeffs[1]:.4f}")

        # Check fit quality
        predicted = np.polyval(coeffs, log_f)
        r_sq = 1 - np.sum((log_m - predicted)**2) / np.sum((log_m - log_m.mean())**2)
        print(f"  R² of 1/f fit: {r_sq:.4f}")

    # Check if carrier has a specific magnitude pattern (e.g., flat within a band)
    print(f"\nMagnitude flatness check:")
    for r_min, r_max in [(10, 100), (100, 300), (300, 500), (500, 800)]:
        mask = (r >= r_min) & (r < r_max)
        if not mask.any():
            continue
        mag_values = avg_mag[:,:,1][mask]
        cv = mag_values.std() / mag_values.mean()
        print(f"  r={r_min:4d}-{r_max:4d}: CV={cv:.4f} (CV=0 means perfectly flat)")

# ---- Analysis 5: Extract the visible watermark pattern ----
def analyze_visible_watermark(images):
    """The visible watermark is identical across all images (r=0.9999). Extract it."""
    print("\n" + "=" * 70)
    print("DEEP 5: VISIBLE WATERMARK EXTRACTION")
    print("=" * 70)

    # Since the visible WM is identical, we can extract it from any image
    # The visible WM = original image - cleaned image
    name, img = images[0]
    h, w = img.shape[:2]

    # Load the cleaned version
    cleaned_path = Path("test-images/gemini-3.1-pro/2400x1792/pure-black-cleaned") / name
    cleaned = cv2.imread(str(cleaned_path), cv2.IMREAD_COLOR).astype(np.float64) / 255.0

    visible_wm = img - cleaned

    print(f"Visible watermark properties:")
    for c, ch_name in enumerate(["Blue", "Green", "Red"]):
        ch = visible_wm[:,:,c]
        print(f"\n{ch_name} channel:")
        print(f"  Mean: {ch.mean():.6f}")
        print(f"  Std: {ch.std():.6f}")
        print(f"  Min: {ch.min():.6f}, Max: {ch.max():.6f}")

    # Where is the visible watermark located?
    wm_energy = (visible_wm**2).sum(axis=2)
    print(f"\nVisible watermark spatial extent:")
    print(f"  Non-zero pixels: {(wm_energy > 1e-8).sum()} / {h*w}")
    print(f"  Energy concentration: top-left quadrant = {(wm_energy[:h//2,:w//2]**2).sum() / (wm_energy**2).sum():.4f}")

    # The visible watermark is in the top-left corner
    # Check exact extent
    rows_with_energy = np.any(wm_energy > 1e-8, axis=1)
    cols_with_energy = np.any(wm_energy > 1e-8, axis=0)
    row_extent = np.where(rows_with_energy)[0]
    col_extent = np.where(cols_with_energy)[0]
    if len(row_extent) > 0 and len(col_extent) > 0:
        print(f"  Extent: rows [{row_extent[0]}:{row_extent[-1]}], cols [{col_extent[0]}:{col_extent[-1]}]")
        print(f"  Size: {col_extent[-1]-col_extent[0]+1}x{row_extent[-1]-row_extent[0]+1}")

# ---- Analysis 6: Carrier amplitude and operating parameters ----
def analyze_carrier_params(images):
    """Quantify the carrier's operating parameters precisely."""
    print("\n" + "=" * 70)
    print("DEEP 6: CARRIER OPERATING PARAMETERS")
    print("=" * 70)

    name, img = images[0]
    h, w = img.shape[:2]
    n = len(images)

    # For pure-black images, the carrier IS the image
    # So the pixel values directly represent the carrier amplitude

    print(f"\nCarrier amplitude (from pure-black image):")
    for c, ch_name in enumerate(["Blue", "Green", "Red"]):
        all_maxes = []
        all_stds = []
        for _, im in images:
            ch = im[:,:,c]
            all_maxes.append(ch.max())
            all_stds.append(ch.std())

        print(f"\n{ch_name} channel:")
        print(f"  Peak amplitude: mean={np.mean(all_maxes):.6f} (std={np.std(all_maxes):.6f})")
        print(f"  RMS amplitude:  mean={np.mean(all_stds):.6f} (std={np.std(all_stds):.6f})")
        print(f"  Peak in 0-255: {np.mean(all_maxes)*255:.2f}")

    # Carrier as fraction of full scale
    print(f"\nCarrier strength as fraction of 8-bit range:")
    for c, ch_name in enumerate(["Blue", "Green", "Red"]):
        rms_values = [im[:,:,c].std() for _, im in images]
        peak_values = [im[:,:,c].max() for _, im in images]
        print(f"  {ch_name}: RMS = {np.mean(rms_values)*255:.2f}/255 ({np.mean(rms_values)*100:.2f}%), "
              f"Peak = {np.mean(peak_values)*255:.2f}/255 ({np.mean(peak_values)*100:.2f}%)")

    # Carrier spectral flatness - is it white noise or colored?
    print(f"\nCarrier noise color analysis:")
    for c, ch_name in enumerate(["Blue", "Green", "Red"]):
        # Average power spectrum
        avg_power = np.zeros((h, w))
        for _, im in images[:10]:
            fft = fft2_channel(im[:,:,c])
            power = np.abs(fft)**2
            avg_power += power
        avg_power /= 10

        # Radial profile of power spectrum
        center_y, center_x = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        rr = np.sqrt((xx - center_x)**2 + (yy - center_y)**2).astype(int)
        max_r = min(center_y, center_x) - 1

        radial_power = np.zeros(max_r)
        for ri in range(max_r):
            mask = (rr == ri)
            if mask.any():
                radial_power[ri] = avg_power[mask].mean()

        # White noise: flat power spectrum (slope ≈ 0)
        # Pink noise: slope ≈ -1
        # Brown noise: slope ≈ -2
        freqs = np.arange(2, min(500, max_r))
        powers = radial_power[2:min(500, max_r)]
        valid = powers > 0
        if valid.sum() > 10:
            log_f = np.log10(freqs[valid].astype(float))
            log_p = np.log10(powers[valid])
            coeffs = np.polyfit(log_f, log_p, 1)
            slope = coeffs[0]
            noise_type = "white" if abs(slope) < 0.3 else "pink" if abs(slope - (-1)) < 0.3 else "brown" if abs(slope - (-2)) < 0.3 else "unknown"
            print(f"  {ch_name}: log-log slope = {slope:.3f} ({noise_type} noise-like)")

# ---- Analysis 7: Per-bin magnitude distribution ----
def analyze_magnitude_distribution(images):
    """Check if per-frequency-bin magnitude follows a specific distribution."""
    print("\n" + "=" * 70)
    print("DEEP 7: PER-BIN MAGNITUDE DISTRIBUTION")
    print("=" * 70)

    h, w = images[0][1].shape[:2]
    n = len(images)

    # Collect magnitude values at specific frequency bins across samples
    center_y, center_x = h // 2, w // 2

    # Pick a few representative frequency bins
    test_bins = [
        (center_y, center_x + 10, "r=10"),
        (center_y, center_x + 50, "r=50"),
        (center_y, center_x + 100, "r=100"),
        (center_y, center_x + 200, "r=200"),
        (center_y, center_x + 400, "r=400"),
        (center_y, center_x + 600, "r=600"),
    ]

    for c, ch_name in enumerate(["Green"]):
        print(f"\n{ch_name} channel - magnitude distribution at specific frequencies:")
        for y, x, label in test_bins:
            if y >= h or x >= w:
                continue
            mags = []
            for _, img in images:
                fft = fft2_channel(img[:,:,c])
                mags.append(np.abs(fft[y, x]))

            mags = np.array(mags)
            print(f"\n  {label} (bin [{y},{x}]):")
            print(f"    Mean: {mags.mean():.4f}, Std: {mags.std():.4f}")
            print(f"    CV: {mags.std()/mags.mean():.4f}")
            print(f"    Min: {mags.min():.4f}, Max: {mags.max():.4f}")
            print(f"    Range: {mags.max()/mags.min():.2f}x")

            # Is it Rayleigh distributed? (expected for complex Gaussian)
            # Rayleigh: mean = sigma * sqrt(pi/2), std = sigma * sqrt((4-pi)/2)
            # So mean/std = sqrt(pi/(4-pi)) ≈ 1.91
            if mags.std() > 0:
                mean_over_std = mags.mean() / mags.std()
                print(f"    Mean/Std ratio: {mean_over_std:.2f} (Rayleigh ≈ 1.91)")

def main():
    images = load_images(BLACK_DIR)
    print(f"Loaded {len(images)} images for deep analysis\n")

    analyze_spatial_common_pattern(images)
    analyze_quadrant_symmetry(images)
    analyze_phase_determinism(images)
    analyze_carrier_template(images)
    analyze_visible_watermark(images)
    analyze_carrier_params(images)
    analyze_magnitude_distribution(images)

    print("\n" + "=" * 70)
    print("DEEP ANALYSIS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
