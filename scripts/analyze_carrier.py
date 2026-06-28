#!/usr/bin/env python3
"""
SynthID Carrier Signal Analysis
================================
Systematic characterization of the SynthID watermark carrier using
pure-black watermarked images where the carrier is the only signal present.

Research questions:
1. What is the common structure vs per-image variation?
2. Is the carrier tiled or repeating?
3. What frequency range does it occupy?
4. How does it behave per channel (B/G/R)?
5. What is the phase structure?
6. Can we reproduce it procedurally?
"""

import cv2
import numpy as np
from pathlib import Path
from scipy import fftpack
from scipy.stats import pearsonr
import json
import sys

BLACK_DIR = Path("test-images/gemini-3.1-pro/2400x1792/pure-black")
CLEANED_DIR = Path("test-images/gemini-3.1-pro/2400x1792/pure-black-cleaned")
COLOR_DIR = Path("test-images/gemini-3.1-pro/2400x1792")

def load_images(directory, max_n=None):
    """Load all PNG images from directory, return list of (filename, BGR float array)."""
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
    """Compute 2D FFT of a single channel, return shifted spectrum."""
    f = np.fft.fft2(channel)
    return np.fft.fftshift(f)

def magnitude_spectrum(fft2):
    return np.abs(fft2)

def phase_spectrum(fft2):
    return np.angle(fft2)

def radial_profile(data, center=None):
    """Compute radial average profile from center of 2D array."""
    h, w = data.shape
    if center is None:
        center = (h // 2, w // 2)
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
    max_r = min(center[0], center[1], h - center[0], w - center[1])
    profile = np.zeros(max_r)
    counts = np.zeros(max_r)
    for ri in range(max_r):
        mask = (r == ri)
        if mask.any():
            profile[ri] = data[mask].mean()
            counts[ri] = mask.sum()
    return profile, counts

def analyze_basic_stats(images):
    """Section 1: Basic pixel statistics across all samples."""
    print("=" * 70)
    print("SECTION 1: BASIC PIXEL STATISTICS")
    print("=" * 70)

    all_means = []
    all_stds = []
    all_pixel_ranges = []

    for name, img in images:
        ch_means = [img[:,:,c].mean() for c in range(3)]
        ch_stds = [img[:,:,c].std() for c in range(3)]
        all_means.append(ch_means)
        all_stds.append(ch_stds)
        all_pixel_ranges.append((img.min(), img.max()))

    all_means = np.array(all_means)
    all_stds = np.array(all_stds)

    print(f"\nSamples: {len(images)}")
    print(f"Image size: {images[0][1].shape[:2]}")
    print(f"\nPixel value range across all samples:")
    print(f"  Min: {min(r[0] for r in all_pixel_ranges):.6f}")
    print(f"  Max: {max(r[1] for r in all_pixel_ranges):.6f}")

    for c, name in enumerate(["Blue", "Green", "Red"]):
        print(f"\n{name} channel:")
        print(f"  Mean intensity: {all_means[:,c].mean():.6f} (std across samples: {all_means[:,c].std():.6f})")
        print(f"  Std deviation:  {all_stds[:,c].mean():.6f} (std across samples: {all_stds[:,c].std():.6f})")

    return all_means, all_stds

def analyze_spectral_common_structure(images):
    """Section 2: Common spectral structure - average FFT across all samples."""
    print("\n" + "=" * 70)
    print("SECTION 2: COMMON SPECTRAL STRUCTURE (Average across all samples)")
    print("=" * 70)

    # Accumulate FFTs
    h, w = images[0][1].shape[:2]
    avg_mag = np.zeros((h, w, 3))
    avg_phase = np.zeros((h, w, 3))
    all_mags = []

    for name, img in images:
        for c in range(3):
            fft = fft2_channel(img[:,:,c])
            mag = magnitude_spectrum(fft)
            phase = phase_spectrum(fft)
            avg_mag[:,:,c] += mag
            avg_phase[:,:,c] += phase
            all_mags.append(mag)

    n = len(images)
    avg_mag /= n
    avg_phase /= n

    # Per-sample deviation from average
    print(f"\nMagnitude spectrum statistics (averaged over {n} samples):")
    for c, name in enumerate(["Blue", "Green", "Red"]):
        ch_mag = avg_mag[:,:,c]
        print(f"\n{name} channel magnitude:")
        print(f"  Mean: {ch_mag.mean():.4f}")
        print(f"  Std:  {ch_mag.std():.4f}")
        print(f"  Min:  {ch_mag.min():.4f}")
        print(f"  Max:  {ch_mag.max():.4f}")
        print(f"  Median: {np.median(ch_mag):.4f}")

    # Compute variance: how much do individual spectra deviate from the mean?
    mag_var = np.zeros((h, w, 3))
    for i, (name, img) in enumerate(images):
        for c in range(3):
            fft = fft2_channel(img[:,:,c])
            mag = magnitude_spectrum(fft)
            mag_var[:,:,c] += (mag - avg_mag[:,:,c])**2
    mag_var /= n

    print(f"\nSpectral variance (per-sample deviation from mean magnitude):")
    for c, name in enumerate(["Blue", "Green", "Red"]):
        var = mag_var[:,:,c]
        mean_mag = avg_mag[:,:,c]
        # Signal-to-noise ratio: how stable is each frequency bin?
        snr = np.where(var > 0, mean_mag**2 / var, 0)
        print(f"\n{name} channel:")
        print(f"  Mean variance: {var.mean():.4f}")
        print(f"  Max variance:  {var.max():.4f}")
        print(f"  SNR (mean):    {snr[snr > 0].mean():.2f}")
        print(f"  SNR median:    {np.median(snr[snr > 0]):.2f}")

        # What fraction of bins are highly stable (SNR > 10)?
        stable_frac = (snr > 10).sum() / snr.size
        very_stable_frac = (snr > 100).sum() / snr.size
        print(f"  Fraction SNR>10:  {stable_frac:.4f}")
        print(f"  Fraction SNR>100: {very_stable_frac:.4f}")

    return avg_mag, avg_phase, mag_var

def analyze_frequency_range(avg_mag):
    """Section 3: Frequency range and bandwidth analysis."""
    print("\n" + "=" * 70)
    print("SECTION 3: FREQUENCY RANGE AND BANDWIDTH")
    print("=" * 70)

    h, w = avg_mag.shape[:2]

    for c, name in enumerate(["Blue", "Green", "Red"]):
        profile, counts = radial_profile(avg_mag[:,:,c])
        total_energy = (profile * counts).sum()

        # Cumulative energy distribution
        cum_energy = np.cumsum(profile * counts) / total_energy

        # Find frequency ranges
        f10 = np.searchsorted(cum_energy, 0.10)
        f25 = np.searchsorted(cum_energy, 0.25)
        f50 = np.searchsorted(cum_energy, 0.50)
        f75 = np.searchsorted(cum_energy, 0.75)
        f90 = np.searchsorted(cum_energy, 0.90)
        f95 = np.searchsorted(cum_energy, 0.95)
        f99 = np.searchsorted(cum_energy, 0.99)

        print(f"\n{name} channel - radial energy distribution:")
        print(f"  10% energy below radius: {f10}")
        print(f"  25% energy below radius: {f25}")
        print(f"  50% energy below radius: {f50}")
        print(f"  75% energy below radius: {f75}")
        print(f"  90% energy below radius: {f90}")
        print(f"  95% energy below radius: {f95}")
        print(f"  99% energy below radius: {f99}")

        # Check if carrier avoids DC (low frequencies)
        dc_energy = (profile[:5] * counts[:5]).sum() / total_energy
        low_freq_energy = (profile[:20] * counts[:20]).sum() / total_energy
        high_freq_energy = (profile[500:] * counts[500:]).sum() / total_energy
        print(f"  DC (r<5) energy:      {dc_energy:.6f}")
        print(f"  Low (r<20) energy:    {low_freq_energy:.6f}")
        print(f"  High (r>500) energy:  {high_freq_energy:.6f}")

        # Peak radial frequency
        peak_r = np.argmax(profile[1:]) + 1  # skip DC
        print(f"  Peak magnitude at radius: {peak_r}")

    return avg_mag

def analyze_tiling_and_repetition(images):
    """Section 4: Detect tiling, repetition, periodic patterns."""
    print("\n" + "=" * 70)
    print("SECTION 4: TILING AND REPETITION ANALYSIS")
    print("=" * 70)

    # Use the first image's spatial domain signal
    name, img = images[0]
    h, w = img.shape[:2]

    for c, ch_name in enumerate(["Blue", "Green", "Red"]):
        ch = img[:,:,c]
        fft = fft2_channel(ch)
        mag = magnitude_spectrum(fft)

        # Autocorrelation of magnitude spectrum (detects periodic patterns)
        mag_log = np.log1p(mag)
        fft_auto = np.fft.fft2(mag_log)
        autocorr = np.real(np.fft.ifft2(fft_auto * np.conj(fft_auto)))
        autocorr /= autocorr.max()

        # Find peaks in autocorrelation (excluding center)
        center_y, center_x = h // 2, w // 2
        # Zero out center region
        autocorr_no_center = autocorr.copy()
        autocorr_no_center[center_y-5:center_y+5, center_x-5:center_x+5] = 0

        # Find top peaks
        flat = autocorr_no_center.flatten()
        top_idx = np.argsort(flat)[-20:][::-1]
        top_peaks = []
        for idx in top_idx:
            py, px = divmod(idx, w)
            val = flat[idx]
            if val > 0.1:  # Only significant peaks
                top_peaks.append((py, px, val))

        print(f"\n{ch_name} channel - magnitude spectrum autocorrelation peaks:")
        if top_peaks:
            for py, px, val in top_peaks[:10]:
                dy = py - center_y
                dx = px - center_x
                print(f"  Offset ({dx:+5d}, {dy:+5d}): correlation={val:.4f}")
                # Periodicity = image_size / offset
                if abs(dx) > 5 and abs(dy) > 5:
                    tile_w = w / abs(dx) if abs(dx) > 5 else float('inf')
                    tile_h = h / abs(dy) if abs(dy) > 5 else float('inf')
                    print(f"    → Suggests tile size: {tile_w:.0f}x{tile_h:.0f}")
        else:
            print("  No significant periodic peaks detected (no tiling)")

        # Check spatial domain for block structure
        # Variance in non-overlapping blocks
        block_sizes = [8, 16, 32, 64, 128, 256]
        print(f"\n{ch_name} - block variance analysis (spatial domain):")
        for bs in block_sizes:
            if bs > min(h, w):
                continue
            variances = []
            for y in range(0, h - bs, bs):
                for x in range(0, w - bs, bs):
                    block = ch[y:y+bs, x:x+bs]
                    variances.append(block.var())
            variances = np.array(variances)
            print(f"  Block {bs:3d}x{bs:<3d}: var_mean={variances.mean():.6f} var_std={variances.std():.6f} cv={variances.std()/max(variances.mean(),1e-10):.4f}")

def analyze_channel_relationships(images):
    """Section 5: Cross-channel analysis."""
    print("\n" + "=" * 70)
    print("SECTION 5: CROSS-CHANNEL RELATIONSHIPS")
    print("=" * 70)

    # Average magnitude per channel
    h, w = images[0][1].shape[:2]
    avg_mag = np.zeros((h, w, 3))

    for name, img in images:
        for c in range(3):
            fft = fft2_channel(img[:,:,c])
            avg_mag[:,:,c] += magnitude_spectrum(fft)

    avg_mag /= len(images)

    # Channel ratios
    blue = avg_mag[:,:,0]
    green = avg_mag[:,:,1]
    red = avg_mag[:,:,2]

    # Global energy ratios
    blue_energy = (blue**2).sum()
    green_energy = (green**2).sum()
    red_energy = (red**2).sum()
    total = blue_energy + green_energy + red_energy

    print(f"\nChannel energy distribution:")
    print(f"  Blue:  {blue_energy/total:.4f} ({blue_energy:.0f})")
    print(f"  Green: {green_energy/total:.4f} ({green_energy:.0f})")
    print(f"  Red:   {red_energy/total:.4f} ({red_energy:.0f})")

    print(f"\nChannel ratios (relative to Green):")
    print(f"  B/G: {blue_energy/green_energy:.4f}")
    print(f"  R/G: {red_energy/green_energy:.4f}")

    # Per-frequency channel correlation
    print(f"\nPer-frequency channel magnitude correlations:")
    b_flat = blue.flatten()
    g_flat = green.flatten()
    r_flat = red.flatten()

    corr_bg, _ = pearsonr(b_flat, g_flat)
    corr_br, _ = pearsonr(b_flat, r_flat)
    corr_gr, _ = pearsonr(g_flat, r_flat)
    print(f"  B-G correlation: {corr_bg:.6f}")
    print(f"  B-R correlation: {corr_br:.6f}")
    print(f"  G-R correlation: {corr_gr:.6f}")

    # Is the pattern shape identical across channels?
    # Normalize each channel and check if they're the same
    blue_norm = blue / (blue.mean() + 1e-10)
    green_norm = green / (green.mean() + 1e-10)
    red_norm = red / (red.mean() + 1e-10)

    corr_shape_bg, _ = pearsonr(blue_norm.flatten(), green_norm.flatten())
    corr_shape_br, _ = pearsonr(blue_norm.flatten(), red_norm.flatten())
    corr_shape_gr, _ = pearsonr(green_norm.flatten(), red_norm.flatten())
    print(f"\nShape correlation (normalized magnitude):")
    print(f"  B-G shape correlation: {corr_shape_bg:.6f}")
    print(f"  B-R shape correlation: {corr_shape_br:.6f}")
    print(f"  G-R shape correlation: {corr_shape_gr:.6f}")

    if corr_shape_bg > 0.99 and corr_shape_br > 0.99:
        print("  → Carrier SHAPE is identical across channels (only strength differs)")
    elif corr_shape_bg > 0.95:
        print("  → Carrier SHAPE is very similar across channels")
    else:
        print("  → Carrier SHAPE differs between channels (different frequency patterns)")

def analyze_phase_structure(images):
    """Section 6: Phase structure analysis."""
    print("\n" + "=" * 70)
    print("SECTION 6: PHASE STRUCTURE")
    print("=" * 70)

    h, w = images[0][1].shape[:2]
    n = len(images)

    # Collect all phases
    all_phases = np.zeros((n, h, w, 3))

    for i, (name, img) in enumerate(images):
        for c in range(3):
            fft = fft2_channel(img[:,:,c])
            all_phases[i,:,:,c] = phase_spectrum(fft)

    # Phase statistics
    print(f"\nPhase distribution per channel:")
    for c, ch_name in enumerate(["Blue", "Green", "Red"]):
        phases = all_phases[:,:,:,c]
        # Phase should be uniformly distributed if random
        print(f"\n{ch_name} channel:")
        print(f"  Mean phase: {phases.mean():.6f}")
        print(f"  Std phase:  {phases.std():.6f}")
        print(f"  (Uniform random would have mean≈0, std≈{np.pi/np.sqrt(3):.4f})")

        # Phase consistency across samples at each frequency
        phase_std_per_bin = phases.std(axis=0)
        print(f"  Per-bin phase std: mean={phase_std_per_bin.mean():.4f}, "
              f"median={np.median(phase_std_per_bin):.4f}")
        print(f"  (Low std = same phase across samples = deterministic)")

        # Fraction of bins with stable phase (std < 0.1)
        stable_frac = (phase_std_per_bin < 0.1).sum() / phase_std_per_bin.size
        semi_stable = (phase_std_per_bin < 0.5).sum() / phase_std_per_bin.size
        very_stable = (phase_std_per_bin < 0.05).sum() / phase_std_per_bin.size
        print(f"  Fraction with phase std < 0.05: {very_stable:.4f}")
        print(f"  Fraction with phase std < 0.10: {stable_frac:.4f}")
        print(f"  Fraction with phase std < 0.50: {semi_stable:.4f}")

        # Is there spatial structure in the phase consistency?
        # Check if certain frequency regions have more stable phase
        profile_consistency, _ = radial_profile(phase_std_per_bin)
        print(f"\n  Phase consistency by frequency (radial):")
        for radius in [5, 10, 20, 50, 100, 200, 300, 500, 700, 896]:
            if radius < len(profile_consistency):
                print(f"    r={radius:4d}: phase_std={profile_consistency[radius]:.4f}")

def analyze_spatial_structure(images):
    """Section 7: Spatial domain structure of the carrier."""
    print("\n" + "=" * 70)
    print("SECTION 7: SPATIAL DOMAIN CARRIER STRUCTURE")
    print("=" * 70)

    name, img = images[0]
    h, w = img.shape[:2]

    # The carrier in spatial domain (the image itself for pure black)
    print(f"\nSpatial carrier properties (pure-black image = carrier in spatial domain):")

    for c, ch_name in enumerate(["Blue", "Green", "Red"]):
        ch = img[:,:,c]
        print(f"\n{ch_name} channel:")
        print(f"  Value range: [{ch.min():.6f}, {ch.max():.6f}]")
        print(f"  Mean: {ch.mean():.6f}, Std: {ch.std():.6f}")

        # Check if carrier is spatially uniform (same statistics everywhere)
        # Divide into quadrants
        q1 = ch[:h//2, :w//2]
        q2 = ch[:h//2, w//2:]
        q3 = ch[h//2:, :w//2]
        q4 = ch[h//2:, w//2:]

        print(f"  Quadrant stds: Q1={q1.std():.6f}, Q2={q2.std():.6f}, "
              f"Q3={q3.std():.6f}, Q4={q4.std():.6f}")
        print(f"  Quadrant means: Q1={q1.mean():.6f}, Q2={q2.mean():.6f}, "
              f"Q3={q3.mean():.6f}, Q4={q4.mean():.6f}")

    # Cross-sample correlation: is the spatial pattern identical across samples?
    print(f"\nCross-sample spatial correlation (Green channel):")
    greens = [img[:,:,1].flatten() for _, img in images[:10]]
    corrs = []
    for i in range(len(greens)):
        for j in range(i+1, min(len(greens), i+5)):
            r, _ = pearsonr(greens[i], greens[j])
            corrs.append(r)
    print(f"  Pairwise spatial correlations: mean={np.mean(corrs):.6f}, "
          f"min={min(corrs):.6f}, max={max(corrs):.6f}")
    if np.mean(corrs) < 0.1:
        print("  → Spatial pattern is DIFFERENT per image (carrier encodes unique data)")
    else:
        print("  → Spatial pattern has COMMON structure across images")

def analyze_differential_signal(images):
    """Section 8: Analyze the differential signal between samples."""
    print("\n" + "=" * 70)
    print("SECTION 8: DIFFERENTIAL ANALYSIS (per-image variation)")
    print("=" * 70)

    # Compute average magnitude
    h, w = images[0][1].shape[:2]
    n = len(images)
    avg_mag = np.zeros((h, w, 3))

    for name, img in images:
        for c in range(3):
            fft = fft2_channel(img[:,:,c])
            avg_mag[:,:,c] += magnitude_spectrum(fft)
    avg_mag /= n

    # For each image, compute the residual from the average
    print(f"\nPer-image magnitude residual analysis (Green channel):")
    residuals = []
    for name, img in images[:10]:
        fft = fft2_channel(img[:,:,1])
        mag = magnitude_spectrum(fft)
        residual = mag - avg_mag[:,:,1]
        residuals.append(residual)
        print(f"  {name[:35]:35s}: residual_std={residual.std():.4f}, "
              f"residual_mean={residual.mean():.4f}")

    residuals = np.array(residuals)

    # Is the residual structured or random?
    avg_residual = residuals.mean(axis=0)
    print(f"\n  Average residual std: {residuals.std():.4f}")
    print(f"  Average residual has structure? (std/mean of avg_residual): "
          f"{avg_residual.std():.4f} / {avg_residual.mean():.4f}")

    # Check if residuals correlate with each other
    flat_residuals = residuals.reshape(n, -1)[:min(n, 10)]
    corrs = []
    for i in range(len(flat_residuals)):
        for j in range(i+1, len(flat_residuals)):
            r, _ = pearsonr(flat_residuals[i], flat_residuals[j])
            corrs.append(r)
    print(f"  Cross-residual correlations: mean={np.mean(corrs):.6f}, "
          f"min={min(corrs):.6f}, max={max(corrs):.6f}")
    if abs(np.mean(corrs)) < 0.05:
        print("  → Per-image variation is RANDOM (independent per image)")
    else:
        print("  → Per-image variation has CORRELATED structure")

def analyze_color_images(images, color_dir):
    """Section 9: Compare carrier on uniform color images."""
    print("\n" + "=" * 70)
    print("SECTION 9: CARRIER ON DIFFERENT UNIFORM BACKGROUNDS")
    print("=" * 70)

    color_images = {
        "pure-black": "2400x1792-pure-black-gemini.png",
        "pure-white": "2400x1792-pure-white-gemini.png",
        "pure-red": "2400x1792-pure-red-gemini.png",
        "pure-green": "2400x1792-pure-green-gemini.png",
        "pure-blue": "2400x1792-pure-blue-gemini.png",
        "pure-grey": "2400x1792-pure-grey-gemini.png",
    }

    print(f"\nCarrier spectral energy by background color:")
    for color_name, filename in color_images.items():
        path = color_dir / filename
        if not path.exists():
            print(f"  {color_name}: file not found")
            continue

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  {color_name}: failed to load")
            continue

        img_f = img.astype(np.float64) / 255.0
        print(f"\n  {color_name} ({filename}):")
        print(f"    Pixel mean: B={img_f[:,:,0].mean():.4f} G={img_f[:,:,1].mean():.4f} R={img_f[:,:,2].mean():.4f}")

        # Compare magnitude spectrum shape with black image average
        for c, ch_name in enumerate(["Blue", "Green", "Red"]):
            fft = fft2_channel(img_f[:,:,c])
            mag = magnitude_spectrum(fft)
            total_energy = (mag**2).sum()
            print(f"    {ch_name} spectral energy: {total_energy:.0f}, "
                  f"DC component: {mag[mag.shape[0]//2, mag.shape[1]//2]:.0f}")

            # Remove DC, compute carrier-only energy
            mag_no_dc = mag.copy()
            cy, cx = mag.shape[0]//2, mag.shape[1]//2
            mag_no_dc[cy-2:cy+3, cx-2:cx+3] = 0
            carrier_energy = (mag_no_dc**2).sum()
            print(f"    {ch_name} carrier energy (no DC): {carrier_energy:.0f} "
                  f"({carrier_energy/total_energy*100:.2f}% of total)")

def analyze_synthid_vs_cleaned(black_dir, cleaned_dir):
    """Section 10: Compare original black images with cleaned versions."""
    print("\n" + "=" * 70)
    print("SECTION 10: ORIGINAL vs CLEANED (visible WM removed, SynthID intact)")
    print("=" * 70)

    black_files = sorted(black_dir.glob("*.png"))
    cleaned_files = sorted(cleaned_dir.glob("*.png"))

    # Match by filename
    matched = []
    for bf in black_files:
        for cf in cleaned_files:
            if bf.name == cf.name:
                matched.append((bf, cf))
                break

    print(f"\nMatched pairs: {len(matched)}")

    if not matched:
        print("No matched pairs found")
        return

    # The "cleaned" images had the visible watermark removed but SynthID is intact
    # The difference between them shows the visible watermark only (not SynthID)
    diffs = []
    for bf, cf in matched[:10]:
        orig = cv2.imread(str(bf), cv2.IMREAD_COLOR).astype(np.float64) / 255.0
        clean = cv2.imread(str(cf), cv2.IMREAD_COLOR).astype(np.float64) / 255.0

        diff = orig - clean
        diffs.append(diff)

        print(f"\n  {bf.name[:35]}:")
        print(f"    Pixel diff: mean={diff.mean():.6f}, max={abs(diff).max():.6f}")
        print(f"    Per-channel diff energy: B={diff[:,:,0].std():.6f} "
              f"G={diff[:,:,1].std():.6f} R={diff[:,:,2].std():.6f}")

    # This tells us: is the visible watermark the same across images?
    # And: does the visible watermark overlap with SynthID's frequency space?
    print(f"\nVisible watermark analysis:")
    for c, ch_name in enumerate(["Blue", "Green", "Red"]):
        ch_diffs = [d[:,:,c] for d in diffs]
        # Check if the visible watermark pattern is the same across images
        corrs = []
        for i in range(len(ch_diffs)):
            for j in range(i+1, len(ch_diffs)):
                r, _ = pearsonr(ch_diffs[i].flatten(), ch_diffs[j].flatten())
                corrs.append(r)
        if corrs:
            print(f"  {ch_name}: cross-image visible WM correlation = {np.mean(corrs):.6f}")

def main():
    print("SynthID Carrier Signal Analysis")
    print("=" * 70)

    # Load images
    images = load_images(BLACK_DIR)
    print(f"Loaded {len(images)} pure-black watermarked images")

    if len(images) < 5:
        print("ERROR: Need at least 5 images for analysis")
        sys.exit(1)

    # Run all analyses
    analyze_basic_stats(images)
    avg_mag, avg_phase, mag_var = analyze_spectral_common_structure(images)
    analyze_frequency_range(avg_mag)
    analyze_tiling_and_repetition(images)
    analyze_channel_relationships(images)
    analyze_phase_structure(images)
    analyze_spatial_structure(images)
    analyze_differential_signal(images)
    analyze_color_images(images, COLOR_DIR)
    analyze_synthid_vs_cleaned(BLACK_DIR, CLEANED_DIR)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
