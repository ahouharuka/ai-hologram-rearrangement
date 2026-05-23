# Weighted Gerchberg–Saxton (WGS) for optical tweezer array holography
#
# This script generates a phase-only hologram (SLM mask) that produces
# an array of diffraction-limited traps at specified coordinates using
# a simple weighted GS loop. It also reconstructs the output-plane
# intensity to visualize performance.
#
# How to use (quick):
# 1) Run this cell. It will generate an 8x8 tweezer array example.
# 2) Files saved to /mnt/data:
#    - "wgs_phase_mask.png" (8-bit wrapped phase for SLM)
#    - "wgs_reconstruction.png" (reconstructed intensity)
#    - "wgs_results.npz" (arrays for further analysis)
#
# Notes:
# - This is a compact, dependency-light implementation (numpy + matplotlib).
# - FFT is unitary-normalized to keep sensible magnitudes.
# - Coordinates are pixels; scale them to your optics if needed.
#
# If you want to tweak array size/spacing, scroll to the "Example usage"
# section near the bottom.

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift

# ---------------------------- Utilities ----------------------------

def wrap_to_2pi(phase):
    """Wrap phase to [0, 2π)."""
    return np.mod(phase, 2.0 * np.pi)

def complex_from_amp_phase(amp, phase):
    """Construct complex field from amplitude and phase arrays."""
    return amp * np.exp(1j * phase)

def normalize_to_unit_mean(arr, eps=1e-12):
    """Normalize non-negative array so mean is 1 (avoid divide-by-zero)."""
    m = np.mean(arr)
    if m < eps:
        return arr
    return arr / m

def make_spot_coords(grid_shape, spacing_px, center, count_xy):
    """
    Generate lattice of spot coordinates.
    grid_shape: (H, W) of the target plane
    spacing_px: pixel spacing between adjacent traps (int)
    center: (cy, cx) center pixel of the array
    count_xy: (ny, nx) number of traps along y and x
    Returns: list of (y, x) integer pixel positions.
    """
    H, W = grid_shape
    ny, nx = count_xy
    cy, cx = center
    ys = np.arange(-(ny - 1) // 2, (ny - 1) // 2 + 1) * spacing_px
    xs = np.arange(-(nx - 1) // 2, (nx - 1) // 2 + 1) * spacing_px
    coords = []
    for j in ys:
        for i in xs:
            y = int(np.round(cy + j))
            x = int(np.round(cx + i))
            if 0 <= y < H and 0 <= x < W:
                coords.append((y, x))
    return coords

# ----------------------- Weighted GS Core --------------------------

def weighted_gs(
    target_coords,
    img_shape=(512, 512),
    iters=150,
    beta=0.5,
    gamma_outside=0.0,
    seed=0,
    initial_phase=None,
    input_amp=1.0,
):
    """
    Weighted Gerchberg–Saxton for point-trap arrays.

    Parameters
    ----------
    target_coords : list of (y, x)
        Pixel coordinates for traps in the Fourier/output plane.
    img_shape : (H, W)
        Simulation grid size.
    iters : int
        Number of WGS iterations.
    beta : float
        Weight update exponent: w <- w * (I_target/I_meas)^beta
        Typical range 0.3~0.8.
    gamma_outside : float in [0,1]
        Outside the trap pixels, blend factor for keeping original amplitude:
        U_new_outside = (1 - gamma) * 0 + gamma * |U_meas|
        Set 0 for strict trap-only constraint; increase slightly to stabilize.
    seed : int
        RNG seed for reproducibility.
    initial_phase : ndarray or None
        Optional initial phase on SLM plane. If None, random in [0, 2π).
    input_amp : float or ndarray
        Amplitude on SLM/input plane. Can be scalar or array of shape img_shape.

    Returns
    -------
    phase_slm : ndarray
        Final phase mask on input plane, wrapped to [0, 2π).
    recon_intensity : ndarray
        Reconstructed intensity in output plane using final phase.
    weights : ndarray [n_traps]
        Final per-trap weights.
    """
    rng = np.random.default_rng(seed)
    H, W = img_shape
    # Input amplitude
    if np.isscalar(input_amp):
        Ain = np.full((H, W), float(input_amp), dtype=np.float32)
    else:
        Ain = np.asarray(input_amp, dtype=np.float32)
        assert Ain.shape == (H, W)

    # Initial phase
    if initial_phase is None:
        phase = rng.uniform(0.0, 2.0 * np.pi, size=(H, W)).astype(np.float32)
    else:
        phase = wrap_to_2pi(np.asarray(initial_phase, dtype=np.float32))

    # Target amplitudes at traps (start equal), and per-trap weights
    n_traps = len(target_coords)
    target_amp = np.ones(n_traps, dtype=np.float64)  # desired equal intensity
    weights = np.ones(n_traps, dtype=np.float64)

    # Pre-build index arrays for scatter/gather at trap locations
    ys = np.array([p[0] for p in target_coords], dtype=np.int32)
    xs = np.array([p[1] for p in target_coords], dtype=np.int32)

    eps = 1e-12

    for _ in range(iters):
        # Forward propagate to output plane
        Uin = complex_from_amp_phase(Ain, phase)
        Uout = fftshift(fft2(ifftshift(Uin), norm="ortho"))

        # Measure trap intensities
        trap_field = Uout[ys, xs]
        trap_intensity = np.abs(trap_field) ** 2 + eps

        # Update weights to equalize achieved intensity to target
        # w <- w * (I_target / I_meas)^beta
        # For equal targets, I_target can be 1 for all traps.
        weights *= (1.0 / trap_intensity) ** beta

        # Build target amplitude map (only at trap pixels)
        A_target = np.zeros_like(Uout, dtype=np.float64)
        A_target[ys, xs] = np.sqrt(weights * target_amp)

        # Apply output-plane constraint
        # Inside traps: set amplitude to A_target, keep current phase
        U_phase = np.exp(1j * np.angle(Uout))
        Unew = np.zeros_like(Uout, dtype=np.complex128)
        Unew[ys, xs] = A_target[ys, xs] * U_phase[ys, xs]

        # Outside traps: optional blending (gamma_outside)
        if gamma_outside > 0.0:
            A_out = np.abs(Uout)
            Unew += gamma_outside * (A_out * U_phase) * (A_target == 0.0)

        # Back propagate to input plane
        Uback = fftshift(ifft2(ifftshift(Unew), norm="ortho"))

        # Enforce input-plane constraint: keep amplitude Ain, update phase
        phase = np.angle(Uback).astype(np.float32)

    # Final products
    phase_slm = wrap_to_2pi(phase)

    # Final reconstruction
    Ufinal = fftshift(fft2(ifftshift(complex_from_amp_phase(Ain, phase_slm)), norm="ortho"))
    recon_intensity = np.abs(Ufinal) ** 2
    recon_intensity = recon_intensity / (recon_intensity.max() + eps)

    return phase_slm, recon_intensity, weights


# ------------------------- Example usage ---------------------------

# Simulation grid
H, W = 768, 768

# Build an 8x8 square lattice of traps
spacing_px = 48
ny, nx = 8, 8
center = (H // 2, W // 2)
coords = make_spot_coords((H, W), spacing_px=spacing_px, center=center, count_xy=(ny, nx))

# Run WGS
phase_mask, recon_I, w_final = weighted_gs(
    target_coords=coords,
    img_shape=(H, W),
    iters=200,
    beta=0.6,
    gamma_outside=0.05,
    seed=42,
    input_amp=1.0,
)

# Save outputs
import imageio.v2 as imageio

# Phase mask as 8-bit image
phase_8bit = (phase_mask / (2.0 * np.pi) * 255.0).astype(np.uint8)
imageio.imwrite("/mnt/data/wgs_phase_mask.png", phase_8bit)

# Reconstruction image (normalized)
recon_8bit = (recon_I / np.max(recon_I) * 255.0).astype(np.uint8)
imageio.imwrite("/mnt/data/wgs_reconstruction.png", recon_8bit)

# Save arrays
np.savez_compressed(
    "/mnt/data/wgs_results.npz",
    phase_mask=phase_mask.astype(np.float32),
    recon_intensity=recon_I.astype(np.float32),
    coords=np.array(coords, dtype=np.int32),
    final_weights=w_final.astype(np.float64),
    grid=np.array([H, W], dtype=np.int32),
    spacing_px=np.array([spacing_px], dtype=np.int32),
    lattice=np.array([ny, nx], dtype=np.int32),
)

# Display quicklooks
plt.figure(figsize=(5, 5))
plt.imshow(phase_8bit)
plt.title("SLM Phase Mask (8-bit, 0..255)")
plt.axis("off")
plt.show()

plt.figure(figsize=(5, 5))
plt.imshow(recon_8bit)
plt.title("Reconstructed Intensity (normalized)")
plt.axis("off")
plt.show()

# Print file paths for the user
print("Saved files:")
print(" - Phase mask: /mnt/data/wgs_phase_mask.png")
print(" - Reconstruction: /mnt/data/wgs_reconstruction.png")
print(" - Arrays (npz): /mnt/data/wgs_results.npz")
