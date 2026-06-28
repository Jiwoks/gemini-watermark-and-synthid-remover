"""
Build a SynthID spectral codebook from a folder of pure-color reference images.

For pure black/white images, the FFT directly reveals the SynthID carrier
(no differential subtraction needed). For other pure colors, we use the
carrier signal approach: accumulate FFTs and compute consistency.

The codebook uses the circular-mean for phase (averaging unit complex numbers)
to properly handle phase wrapping, and computes phase coherence as the
consistency metric (magnitude of the mean unit vector).

Usage:
    python3 scripts/build_codebook_from_folder.py <input_dir> <output.wcb>

Examples:
    # Build from pure black images (carrier = FFT of image directly)
    python3 scripts/build_codebook_from_folder.py \
        reference-images/hf-data/gemini_black/ \
        reference-images/codebook/synthid_hf_black_1024.wcb

    # Build from gemini-3.1-flash pure colors at a specific resolution
    python3 scripts/build_codebook_from_folder.py \
        reference-images/hf-data/gemini-3.1-flash-image-preview/black/1024x1024/ \
        reference-images/codebook/synthid_hf_flash_black_1024.wcb
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


def build_codebook(input_dir, output_path):
    """Build codebook by accumulating FFTs from images in input_dir.

    Uses circular-mean phase averaging (unit complex number accumulation)
    for proper phase wrapping. Consistency = |mean unit vector| (coherence).
    """
    # Collect FFT data per resolution
    profiles = {}  # (h, w) -> {mag_sum, phase_unit_sum, count}

    supported_exts = {'.png', '.jpg', '.jpeg', '.webp'}

    for fname in sorted(os.listdir(input_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in supported_exts:
            continue

        fpath = os.path.join(input_dir, fname)
        img = load_as_float_rgb(fpath)
        h, w = img.shape[:2]
        key = (h, w)

        if key not in profiles:
            profiles[key] = {
                'mag_sum': None,
                'phase_unit_sum': None,
                'count': 0,
            }

        prof = profiles[key]

        for ch in range(3):
            ch_fft = fft2d(img[:, :, ch])
            mag = np.abs(ch_fft)
            # Unit complex number for circular-mean phase
            phase_unit = ch_fft / (mag + 1e-12)  # exp(i*phase)

            if prof['mag_sum'] is None:
                prof['mag_sum'] = [np.zeros_like(mag) for _ in range(3)]
                prof['phase_unit_sum'] = [np.zeros_like(phase_unit) for _ in range(3)]

            prof['mag_sum'][ch] += mag
            prof['phase_unit_sum'][ch] += phase_unit

        prof['count'] += 1
        print(f"  {fname}: {w}x{h}")

    if not profiles:
        print("Error: no images found")
        sys.exit(1)

    # Write .wcb file
    with open(output_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(profiles)))

        for (h, w), prof in sorted(profiles.items()):
            n = prof['count']
            print(f"\n  Profile {w}x{h}: {n} samples")

            f.write(struct.pack("<I", w))
            f.write(struct.pack("<I", h))
            f.write(struct.pack("<i", n))

            for ch in range(3):
                avg_mag = prof['mag_sum'][ch] / n
                avg_phase_unit = prof['phase_unit_sum'][ch] / n
                coherence = np.abs(avg_phase_unit)  # [0, 1]
                avg_phase = np.angle(avg_phase_unit)

                mag_range = f"[{avg_mag.min():.6f}, {avg_mag.max():.2f}]"
                coh_range = f"[{coherence.min():.3f}, {coherence.max():.3f}]"
                high_coh = (coherence > 0.9).sum()
                total = coherence.size
                print(f"  Ch{ch}: mag={mag_range}, coherence={coh_range}, "
                      f"high(>0.9)={100*high_coh/total:.1f}%")

                rows, cols = h, w
                f.write(struct.pack("<I", rows))
                f.write(struct.pack("<I", cols))
                f.write(avg_mag.astype(np.float32).tobytes())
                f.write(avg_phase.astype(np.float32).tobytes())
                f.write(coherence.astype(np.float32).tobytes())

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nWrote {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_dir> <output.wcb>")
        sys.exit(1)

    build_codebook(sys.argv[1], sys.argv[2])
