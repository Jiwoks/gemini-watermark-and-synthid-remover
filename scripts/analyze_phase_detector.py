#!/usr/bin/env python3
"""
Phase Template Detection: Test if the deterministic low-frequency phase
template can detect SynthID on content images.

Key insight: Low-frequency phase (r < 100) is deterministic across all 30
black images. If content images have SynthID, their low-frequency phase
should match this template (after accounting for image content phase).

Strategy:
1. Extract the average low-frequency phase template from black images
2. For a content image, extract the noise residual (bilateral filter subtraction)
3. Check if the noise residual's low-frequency phase matches the template
4. Compare against non-Gemini images as control
"""

import cv2
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import sys

BLACK_DIR = Path("test-images/gemini-3.1-pro/2400x1792/pure-black")
TEST_IMAGES = [
    ("test-images/2400x1792-test1-gemini.png", "gemini-pop-art"),
    ("test-images/2400x1792-test2-gemini.png", "gemini-photo"),
    ("test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-black-gemini.png", "gemini-black"),
    ("test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-white-gemini.png", "gemini-white"),
    ("test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-red-gemini.png", "gemini-red"),
    ("test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-grey-gemini.png", "gemini-grey"),
]

def fft2_channel(channel):
    f = np.fft.fft2(channel)
    return np.fft.fftshift(f)

def extract_phase_template(images, max_radius=100):
    """Extract the average low-frequency phase template from black images."""
    h, w = images[0][1].shape[:2]
    n = len(images)

    # Accumulate phase at low frequencies only
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    low_freq_mask = r <= max_radius

    # For phase averaging, we need to use circular mean (unit vector averaging)
    cos_sum = np.zeros((h, w, 3))
    sin_sum = np.zeros((h, w, 3))

    for name, img in images:
        for c in range(3):
            fft = fft2_channel(img[:,:,c])
            phase = np.angle(fft)
            cos_sum[:,:,c] += np.cos(phase)
            sin_sum[:,:,c] += np.sin(phase)

    avg_phase = np.arctan2(sin_sum, cos_sum)
    phase_concentration = np.sqrt(cos_sum**2 + sin_sum**2) / n

    return avg_phase, phase_concentration, low_freq_mask

def phase_coherence_score(image_float, template_phase, template_concentration,
                          low_freq_mask, channel=1):
    """Score how well an image's noise residual phase matches the template."""
    h, w = image_float.shape[:2]

    # Extract noise residual
    denoised = np.zeros_like(image_float)
    for c in range(3):
        denoised[:,:,c] = cv2.bilateralFilter(image_float[:,:,c].astype(np.float32),
                                                9, 75.0, 75.0).astype(np.float64)
    noise = image_float - denoised

    # FFT of noise residual
    noise_fft = fft2_channel(noise[:,:,channel])
    noise_phase = np.angle(noise_fft)
    noise_mag = np.abs(noise_fft)

    # Also FFT of the full image
    img_fft = fft2_channel(image_float[:,:,channel])
    img_phase = np.angle(img_fft)
    img_mag = np.abs(img_fft)

    # Compute phase coherence at low frequencies
    # cos(template_phase - image_phase) weighted by template concentration
    template_p = template_phase[:,:,channel]
    template_c = template_concentration[:,:,channel]

    # Score 1: Noise residual phase vs template
    cos_diff_noise = np.cos(noise_phase - template_p)
    weighted_score_noise = (cos_diff_noise * template_c * low_freq_mask).sum()
    total_weight = (template_c * low_freq_mask).sum()
    if total_weight > 0:
        score_noise = weighted_score_noise / total_weight
    else:
        score_noise = 0

    # Score 2: Full image phase vs template
    cos_diff_img = np.cos(img_phase - template_p)
    weighted_score_img = (cos_diff_img * template_c * low_freq_mask).sum()
    if total_weight > 0:
        score_img = weighted_score_img / total_weight
    else:
        score_img = 0

    # Score 3: Magnitude correlation with template
    template_mag = np.abs(fft2_channel(np.zeros((h, w))))  # baseline
    # Actually, use the average magnitude from black images

    return score_noise, score_img

def multi_radius_analysis(image_float, template_phase, template_concentration,
                           channel=1, radii=[10, 20, 30, 50, 80, 100, 150, 200]):
    """Analyze phase coherence at multiple radii."""
    h, w = image_float.shape[:2]
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # Noise residual
    denoised = cv2.bilateralFilter(image_float[:,:,channel].astype(np.float32),
                                    9, 75.0, 75.0).astype(np.float64)
    noise = image_float[:,:,channel] - denoised

    noise_fft = fft2_channel(noise)
    noise_phase = np.angle(noise_fft)

    img_fft = fft2_channel(image_float[:,:,channel])
    img_phase = np.angle(img_fft)

    template_p = template_phase[:,:,channel]
    template_c = template_concentration[:,:,channel]

    results = {}
    for radius in radii:
        mask = r <= radius
        total_c = (template_c * mask).sum()
        if total_c < 1e-6:
            results[radius] = {'noise': 0, 'image': 0, 'concentration': 0}
            continue

        # Noise phase coherence
        cos_noise = np.cos(noise_phase - template_p)
        score_noise = (cos_noise * template_c * mask).sum() / total_c

        # Image phase coherence
        cos_img = np.cos(img_phase - template_p)
        score_img = (cos_img * template_c * mask).sum() / total_c

        # Template concentration at this radius
        avg_c = template_c[mask].mean()

        results[radius] = {
            'noise': score_noise,
            'image': score_img,
            'concentration': avg_c,
        }

    return results

def main():
    print("SynthID Phase Template Detection Analysis")
    print("=" * 70)

    # Load black images for template
    black_images = []
    for f in sorted(BLACK_DIR.glob("*.png")):
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is not None:
            black_images.append((f.name, img.astype(np.float64) / 255.0))

    print(f"Template images: {len(black_images)}")

    # Extract phase template
    print("\nExtracting phase template...")
    template_phase, template_conc, mask = extract_phase_template(black_images, max_radius=100)

    # Template quality
    for c, ch_name in enumerate(["Blue", "Green", "Red"]):
        conc = template_conc[:,:,c]
        print(f"  {ch_name} template concentration at r<=100: mean={conc[mask].mean():.4f}, "
              f"min={conc[mask].min():.4f}")

    # Test on various images
    print("\n" + "=" * 70)
    print("PHASE COHERENCE TESTS (r <= 100)")
    print("=" * 70)

    for path, label in TEST_IMAGES:
        p = Path(path)
        if not p.exists():
            print(f"\n{label}: file not found ({path})")
            continue

        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"\n{label}: failed to load")
            continue

        img_f = img.astype(np.float64) / 255.0

        # Basic stats
        std = img_f.std()
        print(f"\n{label} ({std:.4f} std):")

        for c, ch_name in enumerate(["Blue", "Green", "Red"]):
            score_noise, score_img = phase_coherence_score(
                img_f, template_phase, template_conc, mask, channel=c)
            print(f"  {ch_name}: noise_phase_coherence={score_noise:.4f}, "
                  f"img_phase_coherence={score_img:.4f}")

    # Multi-radius analysis
    print("\n" + "=" * 70)
    print("MULTI-RADIUS PHASE COHERENCE (Green channel)")
    print("=" * 70)

    radii = [10, 20, 30, 50, 80, 100, 150, 200]
    print(f"\n{'Image':<20s}", end="")
    for r in radii:
        print(f"  r<={r:3d}", end="")
    print()
    print("-" * 80)

    for path, label in TEST_IMAGES:
        p = Path(path)
        if not p.exists():
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img_f = img.astype(np.float64) / 255.0

        results = multi_radius_analysis(img_f, template_phase, template_conc,
                                         channel=1, radii=radii)

        print(f"{label:<20s}", end="")
        for r in radii:
            score = results[r]['noise']
            print(f"  {score:6.3f}", end="")
        print()

    # Control test: generate random noise and check
    print("\n" + "=" * 70)
    print("CONTROL: Random noise and non-Gemini images")
    print("=" * 70)

    # Random Gaussian noise
    np.random.seed(42)
    h, w = 1792, 2400
    noise_img = np.random.rand(h, w, 3) * 0.02  # Similar RMS to carrier
    results = multi_radius_analysis(noise_img, template_phase, template_conc,
                                     channel=1, radii=radii)
    print(f"\nRandom noise (rms=0.02):", end="")
    for r in radii:
        print(f"  {results[r]['noise']:6.3f}", end="")
    print()

    # Check if non-Gemini images exist
    non_gemini_dir = Path("test-images/reference-images")
    if non_gemini_dir.exists():
        non_gemini_files = list(non_gemini_dir.glob("*.png"))[:5]
        for f in non_gemini_files:
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is None:
                continue
            # Resize to 2400x1792 if needed
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h))
            img_f = img.astype(np.float64) / 255.0
            results = multi_radius_analysis(img_f, template_phase, template_conc,
                                             channel=1, radii=radii)
            print(f"Non-Gemini ({f.name[:20]:<20s}):", end="")
            for r in radii:
                print(f"  {results[r]['noise']:6.3f}", end="")
            print()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
