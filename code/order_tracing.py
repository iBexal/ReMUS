"""Locate echelle orders on a flat, fit their traces, and extract spectra.

Images are 2D of shape (ny, nx). Rows, the y axis, are the dispersion
direction; columns, the x axis, are the cross dispersion direction.
Wavelengths are handled in wavelength_solution.py.

Main entry points: trace_orders, trace_single_order, find_spurious_peaks,
and the Order class.
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
    """Evaluate a Gaussian profile on a constant offset.

    Parameters
    ----------
    x : ndarray
        Positions at which to evaluate, in pixels.
    A : float
        Peak amplitude above the offset, in counts.
    mu : float
        Center of the Gaussian, in pixels.
    sigma : float
        Standard deviation of the Gaussian, in pixels.
    c : float
        Constant offset added to the profile, in counts.

    Returns
    -------
    values : ndarray
        Profile values, same shape as x.
    """
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + c


class Order:
    """One echelle order: its trace and the products extracted along it.

    The trace is described by polynomials in the row index y. The center
    polynomial gives the cross dispersion position x of the order, and the
    sigma polynomial gives the Gaussian width of its profile there.

    Parameters
    ----------
    center_poly : np.poly1d
        Polynomial mapping row index y to trace center x, in pixels.
    sigma_poly : np.poly1d
        Polynomial mapping row index y to profile sigma, in pixels.
    order_number : int, optional
        Physical echelle order number. Default None, meaning the number is
        assigned later by wavelength_solution.py.
    """

    def __init__(self, center_poly, sigma_poly, order_number=None):
        self.center_poly = center_poly
        self.sigma_poly = sigma_poly

        # Physical echelle order number, set by
        # wavelength_solution.assign_order_numbers (or matched to a saved
        # solution by position). Not the trace index.
        self.order_number = order_number

        # Approximate x where this order crosses the middle row. Used to
        # match traces from a later night against a saved solution and
        # recover their order numbers.
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
        """Return the trace center at one or more rows.

        Parameters
        ----------
        y : int or ndarray
            Row index along the dispersion axis, in pixels.

        Returns
        -------
        x : float or ndarray
            Cross dispersion position of the trace center, in pixels.
        """
        return self.center_poly(y)

    def sigma(self, y):
        """Return the profile width at one or more rows.

        Parameters
        ----------
        y : int or ndarray
            Row index along the dispersion axis, in pixels.

        Returns
        -------
        sigma : float or ndarray
            Gaussian width across dispersion, in pixels, always positive.
        """
        # curve_fit is free to return a negative sigma, which would invert
        # the aperture bounds into an empty slice and zero flux.
        return np.abs(self.sigma_poly(y))

    def aperture(self, y, n_sigma=3):
        """Return the column bounds of the extraction aperture at a row.

        Parameters
        ----------
        y : int
            Row index along the dispersion axis, in pixels.
        n_sigma : float, optional
            Half width of the aperture in units of the profile sigma.
            Default 3.

        Returns
        -------
        xmin : int
            Lowest column of the aperture, in pixels. Not clipped to the
            image.
        xmax : int
            Column just past the highest one in the aperture, in pixels,
            suitable as a slice end. Not clipped to the image.
        """
        c = self.center(y)
        s = self.sigma(y)
        return int(np.floor(c - n_sigma * s)), int(np.ceil(c + n_sigma * s))

    def extract_trace(self, image):
        """Sample the image along the trace with cubic interpolation.

        Parameters
        ----------
        image : ndarray
            2D image of shape (ny, nx), rows along dispersion.

        Returns
        -------
        spectrum : ndarray
            Interpolated value at the trace center for each row, shape
            (ny,). Positions outside the image take the nearest edge
            value.
        """
        ny, _ = image.shape
        y = np.arange(ny)
        return map_coordinates(image, np.vstack([y, self.center(y)]), order=3,
                               mode="nearest")

    def extract_sum(self, image, n_sigma=3):
        """Extract a spectrum by summing counts across the aperture.

        Parameters
        ----------
        image : ndarray
            2D image of shape (ny, nx), rows along dispersion.
        n_sigma : float, optional
            Half width of the aperture in units of the profile sigma.
            Default 3.

        Returns
        -------
        spectrum : ndarray
            Summed counts per row, shape (ny,). The aperture is clipped to
            the image, so rows near an edge sum fewer columns.
        """
        ny, nx = image.shape
        out = np.zeros(ny)
        for y in range(ny):
            xmin, xmax = self.aperture(y, n_sigma=n_sigma)
            out[y] = np.sum(image[y, max(0, xmin):min(nx, xmax)])
        return out

    def extract_weighted(self, image, n_sigma=3):
        """Extract a spectrum with Gaussian weights across the aperture.

        Parameters
        ----------
        image : ndarray
            2D image of shape (ny, nx), rows along dispersion.
        n_sigma : float, optional
            Half width of the aperture in units of the profile sigma.
            Default 3.

        Returns
        -------
        spectrum : ndarray
            Weighted mean counts per row, shape (ny,). A row is set to NaN
            when its clipped aperture spans fewer than 2 columns or when
            the weights sum to zero or less.
        """
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
        """Extract a ThAr arc spectrum with a tighter default aperture.

        ThAr lines are narrow, and the tighter aperture keeps flux from
        neighbouring orders out of the lines used for centroiding.

        Parameters
        ----------
        thar_image : ndarray
            2D arc image of shape (ny, nx), rows along dispersion.
        n_sigma : float, optional
            Half width of the aperture in units of the profile sigma.
            Default 2.5.

        Returns
        -------
        spectrum : ndarray
            Weighted mean counts per row, shape (ny,), NaN where the
            aperture is unusable.
        """
        return self.extract_weighted(thar_image, n_sigma=n_sigma)


def find_spurious_peaks(heights, n_neighbors=2, rel_threshold=0.3):
    """Flag interior peaks that are much fainter than their neighbours.

    Real orders follow the smooth blaze envelope, so a peak well below the
    median of its neighbours is more likely a duplicate or an artefact.
    The first and last peaks are never flagged, since the envelope makes
    them genuinely faint.

    Parameters
    ----------
    heights : ndarray or list of float
        Peak heights, in counts, in order of increasing column.
    n_neighbors : int, optional
        Number of peaks compared on each side. Default 2.
    rel_threshold : float, optional
        A peak is flagged when its height is below this fraction of the
        median neighbour height. Default 0.3.

    Returns
    -------
    spurious : list of int
        Indices into heights of the flagged peaks, in increasing order.
        Empty if nothing is flagged.
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
    """Trace one order by fitting a Gaussian to rows across the image.

    Fitting starts at the middle row with start_x as the first guess, then
    walks outwards in both directions every step rows, each successful fit
    seeding the next guess. Rows that are not sampled or whose fit fails
    stay NaN. All three arrays are entirely NaN if the middle row fit
    fails.

    Parameters
    ----------
    image : ndarray
        2D image of shape (ny, nx), rows along dispersion.
    start_x : float
        Initial guess for the trace position at the middle row, in pixels.
    window : int, optional
        Half width of the column range fitted around each guess, in
        pixels. Default 8. Rows offering fewer than 5 columns are skipped.
    step : int, optional
        Row spacing between fits, in pixels. Default 20.

    Returns
    -------
    centers : ndarray
        Fitted trace center per row, in pixels, shape (ny,).
    sigmas : ndarray
        Fitted Gaussian width per row, in pixels, shape (ny,).
    amplitudes : ndarray
        Fitted peak amplitude per row, in counts, shape (ny,).
    """
    ny, nx = image.shape
    mid_y = ny // 2

    centers = np.full(ny, np.nan)
    sigmas = np.full(ny, np.nan)
    amplitudes = np.full(ny, np.nan)

    def fit_row(row, guess):
        """Fit one row near guess; None if the fit is not possible."""
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
    """Find and trace every order visible in a coadded white light flat.

    All FITS files in white_loc are median coadded, peaks are located in
    the middle row of the coadd, and each surviving peak seeds one trace.
    Progress, exclusions and skipped orders are printed. With diagnose set,
    a matplotlib figure of the middle row profile is created but not shown.

    Parameters
    ----------
    white_loc : str
        Directory searched for "*.fits" white light flat frames.
    diagnose : bool, optional
        When True, plot the middle row profile with peak indices and any
        exclusions marked. Default False.
    n_expected : int, optional
        Expected number of orders. Default None. Used in the diagnostic
        plot title only.
    auto_exclude : bool, optional
        When True, drop the peaks flagged by find_spurious_peaks. Default
        True.
    exclude_rel_threshold : float, optional
        Relative height threshold passed to find_spurious_peaks. Default
        0.3.
    extra_exclude_indices : list of int, optional
        Further peak indices to drop, numbered as in the diagnostic plot.
        Default None, meaning no extra exclusions.
    prominence_fraction : float, optional
        Required peak prominence, as a fraction of the largest value in
        the middle row profile. Default 0.005.
    min_separation : int, optional
        Minimum separation between peaks across dispersion, in pixels.
        Default 15.

    Returns
    -------
    orders : list of Order
        One Order per traced peak, with trace_center_pixel set to the peak
        column it started from. Peaks yielding fewer than 4 traced rows are
        skipped.
    coadded : ndarray
        Median coadded flat, 2D of shape (ny, nx).

    Raises
    ------
    FileNotFoundError
        If white_loc contains no FITS files.
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
