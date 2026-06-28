#!/usr/bin/env python3
"""
Generate spectral visualizations of the SynthID carrier.
Produces images we can actually look at to understand the watermark structure.
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLACK_DIR = Path("test-images/gemini-3.1-pro/2400x1792/pure-black")
OUT_DIR = Path("docs/research/images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def fft2_channel(channel):
    f = np.fft.fft2(channel)
    return np.fft.fftshift(f)

def load_black_images():
    images = []
    for f in sorted(BLACK_DIR.glob("*.png")):
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is not None:
            images.append((f.name, img.astype(np.float64) / 255.0))
    return images

def main():
    print("Loading black images...")
    black_images = load_black_images()
    print(f"  Loaded {len(black_images)} images")

    h, w = black_images[0][1].shape[:2]

    # ---- 1. Average spatial pattern (carrier template) ----
    print("1. Computing average spatial pattern...")
    avg_img = np.zeros((h, w, 3))
    for name, img in black_images:
        avg_img += img
    avg_img /= len(black_images)

    # Save individual channel views + composite
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SynthID Carrier - Average Spatial Pattern (30 black images)", fontsize=14)

    ch_names = ['Blue', 'Green', 'Red']
    for i, (ax, ch) in enumerate(zip(axes.flat[:3], ch_names)):
        # Scale to show the subtle pattern
        ch_data = avg_img[:,:,i]
        ax.imshow(ch_data, cmap='inferno', vmin=0, vmax=0.03)
        ax.set_title(f'{ch} channel (0-0.03 range)\nRMS={ch_data.std():.5f}, Mean={ch_data.mean():.5f}')
        ax.axis('off')
        plt.colorbar(ax.images[0], ax=ax, fraction=0.046)

    # Composite RGB
    ax = axes.flat[3]
    composite = np.clip(avg_img * (1.0 / max(avg_img.max(), 0.001)) * 3, 0, 1)
    ax.imshow(composite[:,:,::-1])
    ax.set_title('Composite (enhanced)')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUT_DIR / "carrier_spatial_average.png", dpi=150)
    plt.close()

    # ---- 2. FFT Magnitude spectrum ----
    print("2. Computing FFT magnitude spectra...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("SynthID Carrier - FFT Magnitude Spectrum (log scale)", fontsize=14)

    for c, ch_name in enumerate(ch_names):
        # Average magnitude across all black images
        avg_mag = np.zeros((h, w))
        for name, img in black_images:
            fft = fft2_channel(img[:,:,c])
            avg_mag += np.abs(fft)
        avg_mag /= len(black_images)

        # Log scale
        log_mag = np.log1p(avg_mag)

        # Full spectrum
        ax = axes[0, c]
        im = ax.imshow(log_mag, cmap='magma')
        ax.set_title(f'{ch_name} - Full Spectrum (log)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Zoomed to center (low frequencies, r<=200)
        cy, cx = h//2, w//2
        r = 200
        ax = axes[1, c]
        zoom = log_mag[cy-r:cy+r, cx-r:cx+r]
        im = ax.imshow(zoom, cmap='magma')
        ax.set_title(f'{ch_name} - Low freq zoom (r<=200)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "carrier_fft_magnitude.png", dpi=150)
    plt.close()

    # ---- 3. Phase structure ----
    print("3. Computing phase structure...")
    cos_sum = np.zeros((h, w, 3))
    sin_sum = np.zeros((h, w, 3))
    for name, img in black_images:
        for c in range(3):
            fft = fft2_channel(img[:,:,c])
            phase = np.angle(fft)
            cos_sum[:,:,c] += np.cos(phase)
            sin_sum[:,:,c] += np.sin(phase)

    phase_conc = np.sqrt(cos_sum**2 + sin_sum**2) / len(black_images)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("SynthID Carrier - Phase Coherence Across 30 Images\n(1.0 = identical phase, 0.0 = random)", fontsize=14)

    cy, cx = h//2, w//2
    y, x = np.ogrid[:h, :w]
    r_map = np.sqrt((x - cx)**2 + (y - cy)**2)

    for c, ch_name in enumerate(ch_names):
        conc = phase_conc[:,:,c]

        # Full spectrum
        ax = axes[0, c]
        im = ax.imshow(conc, cmap='hot', vmin=0, vmax=1)
        ax.set_title(f'{ch_name} - Full')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Zoomed to center
        r = 300
        ax = axes[1, c]
        zoom = conc[cy-r:cy+r, cx-r:cx+r]
        im = ax.imshow(zoom, cmap='hot', vmin=0, vmax=1)
        ax.set_title(f'{ch_name} - Zoom (r<=300)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "carrier_phase_coherence.png", dpi=150)
    plt.close()

    # ---- 4. Radial profiles ----
    print("4. Computing radial profiles...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SynthID Carrier - Radial Profiles", fontsize=14)

    max_r = int(min(h, w) / 2)
    radii = np.arange(1, max_r, 2)

    for c, ch_name in enumerate(ch_names):
        avg_mag = np.zeros((h, w))
        for name, img in black_images:
            fft = fft2_channel(img[:,:,c])
            avg_mag += np.abs(fft)
        avg_mag /= len(black_images)

        # Magnitude vs radius
        mag_profile = []
        conc_profile = []
        for rad in radii:
            mask = (r_map >= rad - 1) & (r_map < rad + 1)
            if mask.any():
                mag_profile.append(avg_mag[mask].mean())
                conc_profile.append(phase_conc[:,:,c][mask].mean())
            else:
                mag_profile.append(0)
                conc_profile.append(0)

        # Log magnitude
        ax = axes[0, 0]
        ax.semilogy(radii, mag_profile, label=ch_name, alpha=0.8)

        # Phase coherence
        ax = axes[0, 1]
        ax.plot(radii, conc_profile, label=ch_name, alpha=0.8)

        # Magnitude variance across images
        mag_all = []
        for name, img in black_images:
            fft = fft2_channel(img[:,:,c])
            mag_all.append(np.abs(fft))

        var_profile = []
        for rad in radii:
            mask = (r_map >= rad - 1) & (r_map < rad + 1)
            if mask.any():
                vals = np.array([m[mask].mean() for m in mag_all])
                var_profile.append(vals.std() / (vals.mean() + 1e-10))
            else:
                var_profile.append(0)

        ax = axes[1, 0]
        ax.plot(radii, var_profile, label=ch_name, alpha=0.8)

    axes[0, 0].set_title('Magnitude vs Radius (log)')
    axes[0, 0].set_xlabel('Radius (pixels)')
    axes[0, 0].set_ylabel('Mean |FFT|')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=100, color='red', linestyle='--', alpha=0.5, label='r=100')

    axes[0, 1].set_title('Phase Coherence vs Radius')
    axes[0, 1].set_xlabel('Radius (pixels)')
    axes[0, 1].set_ylabel('Coherence (0=random, 1=deterministic)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
    axes[0, 1].axvline(x=100, color='red', linestyle='--', alpha=0.5)

    axes[1, 0].set_title('Magnitude CV (variability) vs Radius')
    axes[1, 0].set_xlabel('Radius (pixels)')
    axes[1, 0].set_ylabel('Coefficient of Variation')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 1/f power law fit
    ax = axes[1, 1]
    for c, ch_name in enumerate(ch_names):
        avg_mag = np.zeros((h, w))
        for name, img in black_images:
            fft = fft2_channel(img[:,:,c])
            avg_mag += np.abs(fft)
        avg_mag /= len(black_images)

        mag_profile = []
        for rad in radii[:max_r//2]:
            mask = (r_map >= rad - 1) & (r_map < rad + 1)
            if mask.any():
                mag_profile.append(avg_mag[mask].mean())
            else:
                mag_profile.append(0)

        valid = [(r, m) for r, m in zip(radii[:len(mag_profile)], mag_profile) if m > 0]
        if valid:
            rs, ms = zip(*valid)
            log_r = np.log(rs)
            log_m = np.log(ms)
            ax.plot(log_r, log_m, label=f'{ch_name}', alpha=0.8)

    # Reference lines
    ref_r = np.linspace(2, 6, 100)
    ax.plot(ref_r, -1.3 * ref_r + 8, 'k--', alpha=0.5, label='slope=-1.3 (1/f^1.3)')
    ax.plot(ref_r, -1.0 * ref_r + 8, 'k:', alpha=0.5, label='slope=-1.0 (pink noise)')
    ax.set_title('Power Law (log-log magnitude vs radius)')
    ax.set_xlabel('log(radius)')
    ax.set_ylabel('log(mean |FFT|)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "carrier_radial_profiles.png", dpi=150)
    plt.close()

    # ---- 5. Carrier on different backgrounds ----
    print("5. Comparing carrier on different backgrounds...")
    test_cases = [
        ("test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-black-gemini.png", "Black"),
        ("test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-white-gemini.png", "White"),
        ("test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-red-gemini.png", "Red"),
        ("test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-green-gemini.png", "Green"),
        ("test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-blue-gemini.png", "Blue"),
    ]

    fig, axes = plt.subplots(len(test_cases), 4, figsize=(16, 20))
    fig.suptitle("SynthID Carrier - Spectral Comparison Across Backgrounds (Green channel)", fontsize=14)

    for row, (path, label) in enumerate(test_cases):
        p = Path(path)
        if not p.exists():
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        img_f = img.astype(np.float64) / 255.0

        # Noise residual
        denoised = cv2.bilateralFilter(img_f[:,:,1].astype(np.float32), 9, 75.0, 75.0).astype(np.float64)
        noise = img_f[:,:,1] - denoised

        # FFT
        fft = fft2_channel(noise)
        mag = np.abs(fft)
        phase = np.angle(fft)
        log_mag = np.log1p(mag)

        # Spatial noise
        ax = axes[row, 0]
        ax.imshow(noise, cmap='seismic', vmin=-0.02, vmax=0.02)
        ax.set_title(f'{label} - Noise residual' if row == 0 else '')
        ax.set_ylabel(label)
        ax.axis('off')

        # FFT magnitude full
        ax = axes[row, 1]
        ax.imshow(log_mag, cmap='magma')
        ax.set_title('FFT |mag| (log)' if row == 0 else '')
        ax.axis('off')

        # FFT magnitude zoomed
        cy, cx = h//2, w//2
        r = 150
        ax = axes[row, 2]
        ax.imshow(log_mag[cy-r:cy+r, cx-r:cx+r], cmap='magma')
        ax.set_title('Low freq zoom' if row == 0 else '')
        ax.axis('off')

        # Phase zoomed
        ax = axes[row, 3]
        ax.imshow(phase[cy-r:cy+r, cx-r:cx+r], cmap='hsv', vmin=-np.pi, vmax=np.pi)
        ax.set_title('Phase (zoomed)' if row == 0 else '')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUT_DIR / "carrier_backgrounds_comparison.png", dpi=150)
    plt.close()

    # ---- 6. Content image analysis ----
    print("6. Analyzing content images...")
    content_cases = [
        ("test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-black-gemini.png", "Black (reference)"),
        ("test-images/2400x1792-test1-gemini.png", "Pop Art"),
        ("test-images/2400x1792-test2-gemini.png", "Photo"),
    ]

    fig, axes = plt.subplots(len(content_cases), 3, figsize=(16, 16))
    fig.suptitle("SynthID Carrier - Content vs Black Image Spectra (Green channel, noise residual)", fontsize=14)

    for row, (path, label) in enumerate(content_cases):
        p = Path(path)
        if not p.exists():
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        img_f = img.astype(np.float64) / 255.0

        # Noise residual
        denoised = cv2.bilateralFilter(img_f[:,:,1].astype(np.float32), 9, 75.0, 75.0).astype(np.float64)
        noise = img_f[:,:,1] - denoised

        fft = fft2_channel(noise)
        mag = np.abs(fft)
        phase = np.angle(fft)
        log_mag = np.log1p(mag)

        cy, cx = h//2, w//2
        r = 150

        # Noise residual spatial
        ax = axes[row, 0]
        noise_range = max(abs(noise.min()), abs(noise.max()), 0.01)
        ax.imshow(noise, cmap='seismic', vmin=-noise_range, vmax=noise_range)
        ax.set_title(f'Noise residual (std={noise.std():.4f})' if row == 0 else '')
        ax.set_ylabel(label, fontsize=12)
        ax.axis('off')

        # FFT magnitude zoomed
        ax = axes[row, 1]
        ax.imshow(log_mag[cy-r:cy+r, cx-r:cx+r], cmap='magma')
        ax.set_title('FFT magnitude (low freq)' if row == 0 else '')
        ax.axis('off')

        # Radial magnitude profile
        ax = axes[row, 2]
        profile = []
        radii_short = np.arange(1, 200, 1)
        for rad in radii_short:
            mask = (r_map >= rad - 0.5) & (r_map < rad + 0.5)
            if mask.any():
                profile.append(mag[mask].mean())
            else:
                profile.append(0)
        ax.semilogy(radii_short, profile)
        ax.set_title('Magnitude profile' if row == 0 else '')
        ax.set_xlabel('Radius')
        ax.set_ylabel('|FFT|')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 200)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "carrier_content_comparison.png", dpi=150)
    plt.close()

    print(f"\nAll visualizations saved to {OUT_DIR}/")
    for f in sorted(OUT_DIR.glob("*.png")):
        print(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB)")

if __name__ == "__main__":
    main()
