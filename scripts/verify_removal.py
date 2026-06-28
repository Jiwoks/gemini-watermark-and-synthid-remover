#!/usr/bin/env python3
"""Verify SynthID removal by comparing spectral properties of original vs cleaned images."""

import cv2
import numpy as np
from pathlib import Path
import sys

def fft2_channel(channel):
    f = np.fft.fft2(channel)
    return np.fft.fftshift(f)

def radial_mag_profile(img_f, channel=1, max_r=200):
    h, w = img_f.shape[:2]
    fft = fft2_channel(img_f[:,:,channel])
    mag = np.abs(fft)
    cy, cx = h//2, w//2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    radii = np.arange(1, min(max_r, int(min(h,w)/2)), 1)
    profile = []
    for rad in radii:
        mask = (r >= rad - 0.5) & (r < rad + 0.5)
        if mask.any():
            profile.append(mag[mask].mean())
        else:
            profile.append(0)
    return radii, np.array(profile)

def noise_residual(img_f, channel=1):
    denoised = cv2.bilateralFilter(img_f[:,:,channel].astype(np.float32), 9, 75.0, 75.0).astype(np.float64)
    return img_f[:,:,channel] - denoised

def main():
    test_cases = [
        ("test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-black-gemini.png",
         "/tmp/test-codebook-free-black.png", "Black"),
        ("test-images/2400x1792-test1-gemini.png",
         "/tmp/test-codebook-free-popart.png", "PopArt"),
        ("test-images/2400x1792-test2-gemini.png",
         None, "Photo (no removal)"),
    ]

    # Also build a non-Gemini control
    print("=" * 80)
    print("SYNTHID REMOVAL VERIFICATION")
    print("=" * 80)

    for orig_path, clean_path, label in test_cases:
        orig = cv2.imread(orig_path, cv2.IMREAD_COLOR)
        if orig is None:
            print(f"\n{label}: Cannot load {orig_path}")
            continue
        orig_f = orig.astype(np.float64) / 255.0

        print(f"\n{'─' * 60}")
        print(f"{label}: {orig_path}")
        print(f"  Original: mean={orig_f.mean():.4f}, std={orig_f.std():.4f}")

        if clean_path:
            clean = cv2.imread(clean_path, cv2.IMREAD_COLOR)
            if clean is None:
                print(f"  Clean: Cannot load {clean_path}")
                continue
            clean_f = clean.astype(np.float64) / 255.0
            print(f"  Clean:   mean={clean_f.mean():.4f}, std={clean_f.std():.4f}")

            # Pixel-level difference
            diff = np.abs(orig_f - clean_f)
            print(f"  Difference: mean={diff.mean():.6f}, max={diff.max():.4f}, rms={np.sqrt((diff**2).mean()):.6f}")

            # PSNR
            mse = (diff**2).mean()
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
            print(f"  PSNR: {psnr:.1f} dB")

            # Noise residual comparison (green channel)
            orig_noise = noise_residual(orig_f, channel=1)
            clean_noise = noise_residual(clean_f, channel=1)
            print(f"  Noise residual (green): orig_std={orig_noise.std():.6f}, clean_std={clean_noise.std():.6f}")

            # FFT magnitude at low frequencies
            _, orig_prof = radial_mag_profile(orig_f, max_r=200)
            _, clean_prof = radial_mag_profile(clean_f, max_r=200)

            # Carrier band energy (r=50-150 where SynthID is strongest)
            carrier_band = slice(50, 150)
            orig_carrier = orig_prof[carrier_band].mean()
            clean_carrier = clean_prof[carrier_band].mean()
            reduction = (1.0 - clean_carrier / orig_carrier) * 100 if orig_carrier > 0 else 0
            print(f"  Carrier band (r=50-150) mag: orig={orig_carrier:.1f}, clean={clean_carrier:.1f}, reduction={reduction:.1f}%")

        # Show original spectral signature
        radii, prof = radial_mag_profile(orig_f, max_r=200)
        low_freq = prof[:30].mean()
        mid_freq = prof[50:150].mean()
        high_freq = prof[150:].mean()
        print(f"  Orig spectrum: low_r={low_freq:.1f}, mid_r={mid_freq:.1f}, high_r={high_freq:.1f}, ratio={mid_freq/low_freq:.4f}")

    # Control: non-Gemini image
    print(f"\n{'─' * 60}")
    control_dir = Path("test-images/reference-images")
    if control_dir.exists():
        for f in sorted(list(control_dir.glob("*.png"))[:1]):
            ctrl = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if ctrl is None:
                continue
            if ctrl.shape[0] != 1792 or ctrl.shape[1] != 2400:
                ctrl = cv2.resize(ctrl, (2400, 1792))
            ctrl_f = ctrl.astype(np.float64) / 255.0
            print(f"Control ({f.name}): mean={ctrl_f.mean():.4f}, std={ctrl_f.std():.4f}")
            radii, prof = radial_mag_profile(ctrl_f, max_r=200)
            low_freq = prof[:30].mean()
            mid_freq = prof[50:150].mean()
            high_freq = prof[150:].mean()
            print(f"  Spectrum: low_r={low_freq:.1f}, mid_r={mid_freq:.1f}, high_r={high_freq:.1f}, ratio={mid_freq/low_freq:.4f}")

if __name__ == "__main__":
    main()
