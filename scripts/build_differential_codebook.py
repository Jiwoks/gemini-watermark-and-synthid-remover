"""
Build a SynthID spectral codebook using the differential approach.

For pure color images, Gemini_image - reference_image = SynthID carrier signal.
We compute FFT of each difference, average across all pairs, and write a .wcb file.

This produces a much cleaner carrier fingerprint than averaging raw spectra,
because the image content cancels out in the subtraction.

Usage:
    python3 scripts/build_differential_codebook.py \
        test-images/gemini-3.1-pro/2400x1792/ \
        test-images/reference-images/2400x1792/ \
        reference-images/codebook/synthid_differential.wcb
"""

import struct
import sys
import os
import numpy as np
from PIL import Image

MAGIC = b"WMRCB01"


def fft2d(channel):
    """Compute 2D FFT of a single-channel float32 array."""
    return np.fft.fft2(channel)


def load_as_float_rgb(path):
    """Load image, convert to RGB float32 normalized to [0, 1]."""
    img = Image.open(path).convert('RGB')
    return np.array(img, dtype=np.float32) / 255.0


def build_differential_codebook(gemini_dir, reference_dir, output_path):
    colors = ['black', 'white', 'blue', 'cyan', 'green', 'grey',
              'magenta', 'red', 'yellow']

    # Collect carrier estimates per pair using circular-mean
    carrier_mags = []       # list of (3, H, W) arrays
    carrier_phase_units = []  # unit complex for circular-mean
    img_shape = None

    for color in colors:
        gem_path = os.path.join(gemini_dir, f"2400x1792-pure-{color}-gemini.png")
        ref_path = os.path.join(reference_dir, f"2400x1792-pure-{color}.png")

        if not os.path.exists(gem_path):
            print(f"  Skipping {color}: Gemini image not found")
            continue
        if not os.path.exists(ref_path):
            print(f"  Skipping {color}: reference image not found")
            continue

        gem = load_as_float_rgb(gem_path)
        ref = load_as_float_rgb(ref_path)

        if gem.shape != ref.shape:
            print(f"  Skipping {color}: shape mismatch {gem.shape} vs {ref.shape}")
            continue

        if img_shape is None:
            img_shape = gem.shape

        print(f"  {color}: shape={gem.shape}")

        # Per-channel differential FFT
        ch_mags = []
        ch_units = []
        for ch in range(3):
            diff = gem[:, :, ch] - ref[:, :, ch]
            carrier_fft = fft2d(diff)
            mag = np.abs(carrier_fft)
            # Unit complex for circular-mean phase
            unit = carrier_fft / (mag + 1e-12)
            ch_mags.append(mag)
            ch_units.append(unit)

        carrier_mags.append(np.stack(ch_mags))           # (3, H, W)
        carrier_phase_units.append(np.stack(ch_units))    # (3, H, W) complex

    if not carrier_mags:
        print("Error: no valid pairs found")
        sys.exit(1)

    n = len(carrier_mags)
    h, w = img_shape[0], img_shape[1]
    print(f"\n  {n} pairs, image size {w}x{h}")

    # Average carrier magnitude across pairs
    avg_mag = np.mean(carrier_mags, axis=0)       # (3, H, W)

    # Circular-mean phase: average unit complex numbers, then extract angle
    avg_phase_units = np.mean(carrier_phase_units, axis=0)  # (3, H, W) complex
    coherence = np.abs(avg_phase_units)                      # (3, H, W)
    avg_phase = np.angle(avg_phase_units)

    print(f"  Carrier mag range: [{avg_mag.min():.6f}, {avg_mag.max():.2f}]")
    print(f"  Coherence range: [{coherence.min():.3f}, {coherence.max():.3f}]")

    high_coh = (coherence > 0.9).sum()
    total_bins = coherence.size
    print(f"  High-coherence bins (>0.9): {high_coh}/{total_bins} ({100*high_coh/total_bins:.1f}%)")

    # Write .wcb file
    with open(output_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", 1))  # 1 profile

        # Profile header
        f.write(struct.pack("<I", w))
        f.write(struct.pack("<I", h))
        f.write(struct.pack("<i", n))  # sample count

        rows, cols = h, w
        for ch in range(3):
            f.write(struct.pack("<I", rows))
            f.write(struct.pack("<I", cols))
            f.write(avg_mag[ch].astype(np.float32).tobytes())
            f.write(avg_phase[ch].astype(np.float32).tobytes())
            f.write(coherence[ch].astype(np.float32).tobytes())

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nWrote {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <gemini_dir> <reference_dir> <output.wcb>")
        sys.exit(1)

    build_differential_codebook(sys.argv[1], sys.argv[2], sys.argv[3])
