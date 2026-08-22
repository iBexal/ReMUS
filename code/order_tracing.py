"""
order_tracing.py

Finding where each echelle order falls on the detector, fitting its trace,
and extracting a 1D spectrum along it. Nothing here knows about
wavelengths -- that is wavelength_solution.py.
"""

import glob
import os

import fitsio
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def gaussian(x, A, mu, sigma, c):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + c


class Order:
    def __init__(self, center_poly, sigma_poly, order_number=None):
        self.center_poly = center_poly
        self.sigma_poly = sigma_poly

        # Physical echelle order number, set by
        # wavelength_solution.assign_order_numbers (or matched to a saved
        # solution by position). Not the trace index.
        self.order_number = order_number

        # Approximate x where this order crosses the middle row. A
        # mechanical property of the instrument, so it is what a later
        # night's traces get matched against to recover order numbers
        # without redoing the whole calibration.
        self.trace_center_pixel = None

        # Extracted spectra
        self.flat_spectrum = None
        self.thar_spectrum = None
        self.science_spectrum = None

        # Calibration products, filled in by wavelength_solution.py
        self.wavelength_poly = None      # callable: pixel -> wavelength
        self.thar_pixels = None
        self.thar_wavelengths = None

    def center(self, y):
        return self.center_poly(y)

    def sigma(self, y):
        # curve_fit has no constraint against a negative sigma (it is
        # squared in the exponential), and a negative one would silently
        # invert the aperture bounds into an empty slice and zero flux.
        return np.abs(self.sigma_poly(y))

    def aperture(self, y, n_sigma=3):
        c = self.center(y)
        s = self.sigma(y)
        return int(np.floor(c - n_sigma * s)), int(np.ceil(c + n_sigma * s))

    def extract_trace(self, image):
        """Sample the image exactly along the trace, with sub-pixel
        interpolation."""
        ny, _ = image.shape
        y = np.arange(ny)
        return map_coordinates(image, np.vstack([y, self.center(y)]), order=3,
                               mode="nearest")

    def extract_sum(self, image, n_sigma=3):
        ny, nx = image.shape
        out = np.zeros(ny)
        for y in range(ny):
            xmin, xmax = self.aperture(y, n_sigma=n_sigma)
            out[y] = np.sum(image[y, max(0, xmin):min(nx, xmax)])
        return out

    def extract_weighted(self, image, n_sigma=3):
        """Gaussian-weighted (near-optimal) extraction."""
        ny, nx = image.shape
        out = np.zeros(ny)
        for y in range(ny):
            xmin, xmax = self.aperture(y, n_sigma=n_sigma)
            xmin, xmax = max(0, xmin), min(nx, xmax)
            if xmax - xmin < 2:
                out[y] = np.nan
                continue
            x = np.arange(xmin, xmax)
            weights = np.exp(-(x - self.center(y)) ** 2 / (2 * self.sigma(y) ** 2))
            total = weights.sum()
            if total <= 0:
                out[y] = np.nan
                continue
            out[y] = np.sum(image[y, xmin:xmax] * (weights / total))
        return out

    def extract_thar(self, thar_image, n_sigma=2.5):
        """ThAr lines are narrow, so a tighter aperture keeps neighbouring
        orders out while still capturing the line flux that matters for
        centroiding."""
        return self.extract_weighted(thar_image, n_sigma=n_sigma)


def find_spurious_peaks(heights, n_neighbors=2, rel_threshold=0.3):
    """Flag interior peaks that are far fainter than their immediate
    neighbours. A real order follows the smooth blaze envelope, so a peak
    well below its neighbours is more likely a duplicate or artefact.
    Edge peaks are never flagged -- the envelope makes those genuinely
    faint by design.
    """
    heights = np.asarray(heights)
    spurious = []
    for i in range(1, len(heights) - 1):
        lo = max(0, i - n_neighbors)
        hi = min(len(heights), i + n_neighbors + 1)
        neighbours = [j for j in range(lo, hi) if j != i]
        if heights[i] < rel_threshold * np.median(heights[neighbours]):
            spurious.append(i)
    return spurious


def trace_single_order(image, start_x, window=8, step=20):
    ny, nx = image.shape
    mid_y = ny // 2

    centers = np.full(ny, np.nan)
    sigmas = np.full(ny, np.nan)
    amplitudes = np.full(ny, np.nan)

    def fit_row(row, guess):
        xmin = max(0, int(np.floor(guess - window)))
        xmax = min(nx, int(np.ceil(guess + window + 1)))
        if xmax - xmin < 5:
            return None
        x = np.arange(xmin, xmax)
        profile = image[row, xmin:xmax]
        try:
            p0 = (profile.max() - np.median(profile), guess, 2, np.median(profile))
            popt, _ = curve_fit(gaussian, x, profile, p0=p0)
            return popt
        except Exception:
            return None

    popt = fit_row(mid_y, start_x)
    if popt is None:
        return centers, sigmas, amplitudes
    centers[mid_y], sigmas[mid_y], amplitudes[mid_y] = popt[1], popt[2], popt[0]

    for direction in (-step, step):
        guess = centers[mid_y]
        rows = range(mid_y + direction, -1 if direction < 0 else ny, direction)
        for row in rows:
            popt = fit_row(row, guess)
            if popt is None:
                continue
            centers[row], sigmas[row], amplitudes[row] = popt[1], popt[2], popt[0]
            guess = popt[1]

    return centers, sigmas, amplitudes


def trace_orders(white_loc, diagnose=False, n_expected=None, auto_exclude=True,
                 exclude_rel_threshold=0.3, extra_exclude_indices=None,
                 prominence_fraction=0.005, min_separation=15):
    """Find and trace every order visible in a coadded white-light flat.

    Returns (orders, coadded_white). Each Order gets trace_center_pixel set
    to the peak x it was started from.
    """
    white_files = glob.glob(os.path.join(white_loc, "*.fits"))
    if not white_files:
        raise FileNotFoundError(f"no FITS files in {white_loc}")
    frames = []
    for path in white_files:
        with fitsio.FITS(path) as f:
            frames.append(f[0].read())
    coadded = np.median(frames, axis=0)

    profile = coadded[coadded.shape[0] // 2, :]
    peaks, _ = find_peaks(profile, prominence=np.max(profile) * prominence_fraction,
                          distance=min_separation)

    exclude = set(extra_exclude_indices or [])
    if auto_exclude:
        flagged = find_spurious_peaks(profile[peaks], rel_threshold=exclude_rel_threshold)
        if flagged:
            print(f"Auto-flagged {len(flagged)} spurious peak(s) at indices {flagged} "
                  f"(x={peaks[flagged].tolist()})")
        exclude |= set(flagged)

    if diagnose:
        plt.figure(figsize=(13, 5))
        plt.plot(profile, lw=0.8)
        plt.plot(peaks, profile[peaks], "rx")
        if exclude:
            idx = sorted(exclude)
            plt.plot(peaks[idx], profile[peaks[idx]], "ko", ms=10, mfc="none",
                     label="excluded")
            plt.legend()
        for i, p in enumerate(peaks):
            plt.annotate(str(i), (p, profile[p]), fontsize=6, rotation=90,
                         xytext=(0, 10), textcoords="offset points")
        plt.title(f"{len(peaks)} peaks found (expected {n_expected}), "
                  f"{len(exclude)} excluded")
        plt.xlabel("x pixel")
        plt.tight_layout()

    if exclude:
        keep = np.ones(len(peaks), bool)
        keep[list(exclude)] = False
        print(f"Excluding {len(exclude)} peak(s) at x={peaks[~keep].tolist()}")
        peaks = peaks[keep]

    orders = []
    for peak in peaks:
        centers, sigmas, _ = trace_single_order(coadded, start_x=peak, step=20)
        good = np.isfinite(centers)
        if np.sum(good) < 4:
            print(f"Skipping order at x={peak}: only {np.sum(good)} traced points")
            continue
        rows = np.where(good)[0]
        order = Order(center_poly=np.poly1d(np.polyfit(rows, centers[good], 3)),
                      sigma_poly=np.poly1d(np.polyfit(rows, sigmas[good], 3)))
        order.trace_center_pixel = float(peak)
        orders.append(order)

    print(f"Traced {len(orders)} orders.")
    return orders, coadded
