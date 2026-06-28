#!/usr/bin/env python3
"""
Per-channel phase coherence analysis.

Key finding from phase_detector: The carrier IS detectable via phase coherence,
but only on channels where the image content is dark. This means:
- Black image: all channels show coherence ~0.98
- Red image: B/G channels (dark) show ~0.96, R channel (bright) shows ~0.03
- White image: all channels bright, all show ~0.01

Hypothesis: The carrier has constant amplitude. On dark channels, carrier/noise
ratio is high (detectable). On bright channels, image content dominates.

Test: Check per-channel coherence, then check if NORMALIZING by image content
magnitude makes the carrier detectable even on bright channels.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

BLACK_DIR = Path("test-images/gemini-3.1-pro/2400x1792/pure-black")

def fft2_channel(channel):
    f = np.fft.fft2(channel)
    return np.fft.fftshift(f)

def main():
    # Load template
    black_images = []
    for f in sorted(BLACK_DIR.glob("*.png")):
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is not None:
            black_images.append((f.name, img.astype(np.float64) / 255.0))

    h, w = black_images[0][1].shape[:2]
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    mask_100 = r <= 100

    # Build phase template
    cos_sum = np.zeros((h, w, 3))
    sin_sum = np.zeros((h, w, 3))
    avg_mag = np.zeros((h, w, 3))

    for name, img in black_images:
        for c in range(3):
            fft = fft2_channel(img[:,:,c])
            phase = np.angle(fft)
            cos_sum[:,:,c] += np.cos(phase)
            sin_sum[:,:,c] += np.sin(phase)
            avg_mag[:,:,c] += np.abs(fft)

    avg_phase = np.arctan2(sin_sum, cos_sum)
    phase_conc = np.sqrt(cos_sum**2 + sin_sum**2) / len(black_images)
    avg_mag /= len(black_images)

    # ---- Test: Per-channel coherence vs channel brightness ----
    print("=" * 70)
    print("PER-CHANNEL PHASE COHERENCE vs CHANNEL BRIGHTNESS")
    print("=" * 70)

    test_cases = [
        ("test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-black-gemini.png", "black"),
        ("test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-white-gemini.png", "white"),
        ("test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-red-gemini.png", "red"),
        ("test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-green-gemini.png", "green"),
        ("test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-blue-gemini.png", "blue"),
        ("test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-grey-gemini.png", "grey"),
        ("test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-cyan-gemini.png", "cyan"),
        ("test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-yellow-gemini.png", "yellow"),
        ("test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-magenta-gemini.png", "magenta"),
        ("test-images/2400x1792-test1-gemini.png", "pop-art"),
        ("test-images/2400x1792-test2-gemini.png", "photo"),
    ]

    print(f"\n{'Image':<12s} {'Ch':<5s} {'Mean':>6s} {'Std':>6s} {'PhaseCoh':>9s} {'SNR_est':>8s}")
    print("-" * 55)

    for path, label in test_cases:
        p = Path(path)
        if not p.exists():
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img_f = img.astype(np.float64) / 255.0

        for c, ch_name in enumerate(["B", "G", "R"]):
            ch = img_f[:,:,c]
            ch_mean = ch.mean()
            ch_std = ch.std()

            # Phase coherence
            fft = fft2_channel(ch)
            phase = np.angle(fft)
            cos_diff = np.cos(phase - avg_phase[:,:,c])
            total_c = (phase_conc[:,:,c] * mask_100).sum()
            if total_c > 0:
                coh = (cos_diff * phase_conc[:,:,c] * mask_100).sum() / total_c
            else:
                coh = 0

            # Estimate carrier SNR: carrier_rms / (image_std + epsilon)
            carrier_rms = 0.014  # Known carrier RMS
            snr_est = carrier_rms / max(ch_std, 0.001)

            print(f"{label:<12s} {ch_name:<5s} {ch_mean:6.3f} {ch_std:6.4f} {coh:9.4f} {snr_est:8.3f}")
        print()

    # ---- Test: Content-normalized phase coherence ----
    print("\n" + "=" * 70)
    print("CONTENT-NORMALIZED PHASE COHERENCE")
    print("=" * 70)
    print("Idea: weight phase comparison by carrier_magnitude / image_magnitude")
    print("This should make carrier detectable even on bright channels.\n")

    for path, label in test_cases:
        p = Path(path)
        if not p.exists():
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img_f = img.astype(np.float64) / 255.0

        print(f"{label}:")

        # Use noise residual
        denoised = np.zeros_like(img_f)
        for c in range(3):
            denoised[:,:,c] = cv2.bilateralFilter(
                img_f[:,:,c].astype(np.float32), 9, 75.0, 75.0).astype(np.float64)
        noise = img_f - denoised

        for c, ch_name in enumerate(["Blue", "Green", "Red"]):
            # Raw noise phase coherence
            noise_fft = fft2_channel(noise[:,:,c])
            noise_phase = np.angle(noise_fft)
            noise_mag = np.abs(noise_fft)

            cos_diff = np.cos(noise_phase - avg_phase[:,:,c])
            total_c = (phase_conc[:,:,c] * mask_100).sum()
            raw_coh = (cos_diff * phase_conc[:,:,c] * mask_100).sum() / total_c if total_c > 0 else 0

            # Magnitude-weighted: weight by (carrier_mag / noise_mag)
            # This amplifies bins where carrier is strong relative to noise
            weight = phase_conc[:,:,c] * mask_100
            # Add carrier-to-noise magnitude weighting
            with np.errstate(divide='ignore', invalid='ignore'):
                cnr_weight = np.where(noise_mag > 0, avg_mag[:,:,c] / noise_mag, 0)
            cnr_weight = np.clip(cnr_weight, 0, 10)  # cap
            total_w = (weight * cnr_weight).sum()
            if total_w > 0:
                weighted_coh = (cos_diff * weight * cnr_weight).sum() / total_w
            else:
                weighted_coh = 0

            # DC-excluded noise magnitude correlation with carrier template
            # (this measures if the noise spectral shape matches the carrier)
            template_mag_norm = avg_mag[:,:,c] / (avg_mag[:,:,c][mask_100].mean() + 1e-10)
            noise_mag_norm = noise_mag / (noise_mag[mask_100].mean() + 1e-10)

            flat_a = template_mag_norm[mask_100].flatten()
            flat_b = noise_mag_norm[mask_100].flatten()
            mag_corr = np.corrcoef(flat_a, flat_b)[0, 1] if flat_a.std() > 0 and flat_b.std() > 0 else 0

            print(f"  {ch_name}: raw_phase_coh={raw_coh:.4f}, weighted_coh={weighted_coh:.4f}, "
                  f"mag_corr={mag_corr:.4f}")

    # ---- Final test: Combined score for Gemini vs non-Gemini ----
    print("\n" + "=" * 70)
    print("COMBINED DETECTION SCORE (all channels, noise residual)")
    print("=" * 70)
    print("Score = max over channels of (low-freq phase coherence on noise residual)\n")

    scores = []
    for path, label in test_cases:
        p = Path(path)
        if not p.exists():
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img_f = img.astype(np.float64) / 255.0

        # Noise residual
        denoised = np.zeros_like(img_f)
        for c in range(3):
            denoised[:,:,c] = cv2.bilateralFilter(
                img_f[:,:,c].astype(np.float32), 9, 75.0, 75.0).astype(np.float64)
        noise = img_f - denoised

        best_ch_score = 0
        best_ch = ""
        ch_scores = []
        for c, ch_name in enumerate(["B", "G", "R"]):
            noise_fft = fft2_channel(noise[:,:,c])
            noise_phase = np.angle(noise_fft)
            cos_diff = np.cos(noise_phase - avg_phase[:,:,c])
            total_c = (phase_conc[:,:,c] * mask_100).sum()
            coh = (cos_diff * phase_conc[:,:,c] * mask_100).sum() / total_c if total_c > 0 else 0
            ch_scores.append((ch_name, coh))
            if coh > best_ch_score:
                best_ch_score = coh
                best_ch = ch_name

        # Also compute the min channel mean (darkest channel)
        min_mean = min(img_f[:,:,c].mean() for c in range(3))
        all_scores = [s[1] for s in ch_scores]
        avg_score = np.mean(all_scores)

        scores.append((label, best_ch_score, best_ch, min_mean, avg_score))
        print(f"  {label:<12s}: best_ch={best_ch} coh={best_ch_score:.4f}, "
              f"avg_coh={avg_score:.4f}, darkest_ch_mean={min_mean:.3f}")

    # Control: random noise
    np.random.seed(42)
    noise_only = np.random.rand(h, w, 3) * 0.02
    for c in range(3):
        noise_fft = fft2_channel(noise_only[:,:,c])
        noise_phase = np.angle(noise_fft)
        cos_diff = np.cos(noise_phase - avg_phase[:,:,c])
        total_c = (phase_conc[:,:,c] * mask_100).sum()
        coh = (cos_diff * phase_conc[:,:,c] * mask_100).sum() / total_c if total_c > 0 else 0
        if coh > best_ch_score:
            best_ch_score = coh

    print(f"\n  Random noise: coh={best_ch_score:.4f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    print("\nThe carrier is detectable via phase coherence when:")
    print("1. The channel has low mean (carrier dominates noise residual)")
    print("2. The phase template at r<=100 is highly concentrated (>0.97)")
    print()
    print("For content images (pop-art, photo), ALL channels have high mean")
    print("so ALL channels show low phase coherence (~0.04-0.08).")
    print("This is because the image content completely swamps the carrier")
    print("in the noise residual.")
    print()
    print("The carrier IS there — but at ~1.4% RMS, it's undetectable against")
    print("content noise that's 10-30x stronger.")

if __name__ == "__main__":
    main()
