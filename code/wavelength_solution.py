"""Wavelength calibration for an echelle spectrograph.

The module holds the wavelength model and the routines that fit, judge,
save and reuse one. The model works in m * lambda, the product of echelle
order number and wavelength, because a single smooth function of detector
row describes that product for every order at once. A low-degree Chebyshev
correction in (pixel, order number) is fitted on top of it.

Every class that ends up inside a saved master is defined in this module,
because pickle resolves a class by the module it was defined in.

Main entry points
-----------------
assign_order_numbers : label traced orders with the order number m.
seed_from_doublet : build a linear seed from one clicked doublet.
lock_seed : refine the seed against the atlas, all orders at once.
solve : alternate matching and fitting to reach the final solution.
assess : run the quality checks and return a QualityReport.
save_solution, load_solution : write and read a master solution.

Order tracing and extraction are in order_tracing.py.
"""

import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev as C
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# The fitting switches live in config, but this module is also what a
# saved master unpickles into, and that has to work wherever the pickle
# is opened. So config is optional here and every use of it goes through
# getattr with the default written out, which is what getattr(None, ...)
# falls back to.
try:
    import config
except ImportError:                      # pragma: no cover - import path only
    config = None

C_LIGHT_MS = 299792458.0


# ======================================================================
# grating geometry
# ======================================================================

def compute_grating_K(blaze_angle_deg, groove_density_mm):
    """Return the Littrow grating constant K, in Angstrom.

    K = 2 * d * sin(blaze), with d the groove spacing, so that
    m * lambda is close to K at the blaze peak. Used as a starting guess
    and as a sanity check on the fitted solution.

    Parameters
    ----------
    blaze_angle_deg : float
        Blaze angle in degrees.
    groove_density_mm : float
        Groove density in grooves per mm.

    Returns
    -------
    K : float
        Grating constant in Angstrom.
    """
    d_angstrom = 1e7 / groove_density_mm
    return 2.0 * d_angstrom * np.sin(np.deg2rad(blaze_angle_deg))


def assign_order_numbers(orders, K, anchors, direction):
    """Assign the physical echelle order number m to every traced order.

    Each anchor gives m = K / lambda; rounding that to an integer and
    stepping back to trace index 0 gives m0. Anchors far apart in trace
    index must agree on m0. Sets order_number on every element of orders
    in place.

    Parameters
    ----------
    orders : list
        Traced orders in trace index order. Each element gains an integer
        order_number attribute.
    K : float
        Grating constant in Angstrom.
    anchors : list of tuple
        (trace_index, rest_wavelength) for lines of known order, with the
        wavelength in Angstrom.
    direction : int
        Sign in m_i = m0 + direction * i; +1 or -1.

    Returns
    -------
    m0 : int
        Order number of trace index 0.

    Raises
    ------
    ValueError
        If the anchors imply different values of m0.
    """
    implied = []
    print("Order numbering from anchors:")
    for i, wave in anchors:
        m_float = K / wave
        m_round = int(round(m_float))
        m0_i = m_round - direction * i
        implied.append(m0_i)
        print(f"  trace {i:3d}  lambda={wave:9.2f} A  ->  m = K/lambda = {m_float:7.3f}"
              f"  -> m = {m_round}  -> implies m0 = {m0_i}")

    if len(set(implied)) > 1:
        raise ValueError(
            f"Anchors disagree on m0 ({implied}). Either the grating constant is wrong "
            f"for this configuration, `direction` has the wrong sign, or an order is "
            f"missing from the trace list between the two anchor traces (which would "
            f"shift every trace index past it). Fix this before going any further, "
            f"every wavelength downstream is built on it.")

    m0 = implied[0]
    for i, order in enumerate(orders):
        order.order_number = int(m0 + direction * i)

    ms = np.array([o.order_number for o in orders])
    print(f"  -> m0 = {m0}, orders run m={ms.max()} (trace 0) down to m={ms.min()} "
          f"(trace {len(orders) - 1})")
    print(f"  -> nominal central wavelengths K/m: {K / ms.max():.0f} A (bluest) to "
          f"{K / ms.min():.0f} A (reddest)")
    return m0


def check_trace_spacing(orders, tolerance=0.35):
    """Report gaps between adjacent traces that suggest a missed order.

    Order numbering is m0 + direction * trace_index, so an order missed
    between two traced orders shifts every order number past it. The
    separation between adjacent orders varies smoothly, so a gap wider
    than 1 + tolerance times the local median is flagged. Prints a
    warning for each suspicious gap.

    Parameters
    ----------
    orders : list
        Traced orders, each with trace_center_pixel in pixels.
    tolerance : float, optional
        Fractional excess over the local median gap that counts as
        suspicious. Default 0.35.

    Returns
    -------
    bad : list of int
        Trace indices after which the gap looks suspicious. Empty if the
        spacing is smooth, and empty if fewer than six orders are given.
    """
    x = np.array([o.trace_center_pixel for o in orders], dtype=float)
    gaps = np.diff(x)
    if len(gaps) < 5:
        return []
    smooth = median_filter(gaps, size=7, mode="nearest")
    ratio = gaps / smooth
    bad = np.where(ratio > 1.0 + tolerance)[0]
    for b in bad:
        print(f"  WARNING: gap between trace {b} (x={x[b]:.0f}) and {b + 1} (x={x[b + 1]:.0f}) "
              f"is {gaps[b]:.0f} px, {ratio[b]:.2f}x the local trend, so an order may be"
              f"missing here, which would shift every order number past it.")
    if len(bad) == 0:
        print(f"  trace spacing is smooth across all {len(orders)} orders "
              f"(no missing order between them).")
    return bad.tolist()


# ======================================================================
# ThAr atlas
# ======================================================================

class ReferenceLines:
    """Atlas lines usable for calibration, sorted by wavelength.

    wave is the catalogue wavelength; eff_wave is the amplitude-weighted
    centroid of everything the instrument blends into that line at its
    resolution, which is what a centroid measured on the detector should
    be compared against.

    Parameters
    ----------
    wave : ndarray
        Catalogue wavelengths in Angstrom, shape (n,).
    eff_wave : ndarray
        Effective blend-weighted wavelengths in Angstrom, shape (n,).
    amplitude : ndarray
        Catalogue line amplitudes, shape (n,).
    """

    def __init__(self, wave, eff_wave, amplitude):
        order = np.argsort(wave)
        self.wave = np.asarray(wave, float)[order]
        self.eff_wave = np.asarray(eff_wave, float)[order]
        self.amplitude = np.asarray(amplitude, float)[order]

    def __len__(self):
        """Number of reference lines held."""
        return len(self.wave)


def load_atlas(path, ion_prefix="Th", amplitude_min=10.0):
    """Read a ThAr line list in the '|'-delimited. The lines
    are in VACUUM wavelengths (from pypeit).

    Ar lines are dropped by default; they blend and shift more readily
    than Th lines. The complete list is returned as well, because the
    blend and dominance test needs the rejected lines too.

    Parameters
    ----------
    path : str
        Path to the line list.
    ion_prefix : str, optional
        Keep only ions whose name starts with this prefix. Default "Th".
    amplitude_min : float, optional
        Keep only lines with amplitude above this value. Default 10.0.

    Returns
    -------
    wave : ndarray
        Wavelengths of the selected lines, in Angstrom.
    amplitude : ndarray
        Amplitudes of the selected lines.
    full_wave : ndarray
        Wavelengths of every line in the file, in Angstrom.
    full_amplitude : ndarray
        Amplitudes of every line in the file.
    """
    atlas = pd.read_csv(path, delimiter="|")
    atlas.columns = [c.strip() for c in atlas.columns]
    atlas["ion"] = atlas["ion"].str.strip()
    atlas = atlas.sort_values("wave")

    full_wave = atlas["wave"].values.astype(float)
    full_amp = atlas["amplitude"].values.astype(float)

    keep = atlas["ion"].str.startswith(ion_prefix).values & (full_amp > amplitude_min)
    print(f"load_atlas: {len(atlas)} lines, kept {keep.sum()} with ion '{ion_prefix}*' "
          f"and amplitude > {amplitude_min}")
    return full_wave[keep], full_amp[keep], full_wave, full_amp


def select_reference_lines(sel_wave, sel_amp, full_wave, full_amp,
                           resolution_angstrom, amplitude_min=200.0,
                           dominance=5.0):
    """Keep the atlas lines this resolution can measure without blend bias.

    A line survives if it is one of the pre-selected lines, its amplitude
    exceeds amplitude_min, and its amplitude exceeds dominance times the
    summed amplitude of every other line within one resolution width.
    Each survivor is given an effective wavelength, the amplitude-weighted
    centroid of that whole group.

    Parameters
    ----------
    sel_wave : ndarray
        Wavelengths in Angstrom of the pre-selected lines.
    sel_amp : ndarray
        Amplitudes of the pre-selected lines.
    full_wave : ndarray
        Wavelengths in Angstrom of every atlas line, sorted ascending.
    full_amp : ndarray
        Amplitudes of every atlas line.
    resolution_angstrom : callable
        Maps wavelength in Angstrom to resolution width in Angstrom.
    amplitude_min : float, optional
        Least amplitude of a surviving line. Default 200.0.
    dominance : float, optional
        Required ratio of a line's amplitude to the summed amplitude of
        its neighbours within one resolution width. Default 5.0.

    Returns
    -------
    ref : ReferenceLines
        The surviving lines with their effective wavelengths.
    """
    w = np.asarray(full_wave, float)
    a = np.asarray(full_amp, float)
    width = np.asarray(resolution_angstrom(w), float)

    lo = np.searchsorted(w, w - width)
    hi = np.searchsorted(w, w + width)
    cum_a = np.concatenate([[0.0], np.cumsum(a)])
    cum_aw = np.concatenate([[0.0], np.cumsum(a * w)])
    group_amp = cum_a[hi] - cum_a[lo]
    group_awave = cum_aw[hi] - cum_aw[lo]
    neighbours = group_amp - a
    eff = group_awave / np.maximum(group_amp, 1e-30)

    selected = np.isin(w, sel_wave)
    keep = selected & (a > amplitude_min) & (a > dominance * neighbours)

    ref = ReferenceLines(w[keep], eff[keep], a[keep])
    shift = np.abs(ref.wave - ref.eff_wave)
    print(f"select_reference_lines: {len(ref)} isolated, dominant lines "
          f"(amplitude > {amplitude_min}, > {dominance}x its blend neighbours); "
          f"blend correction median {np.median(shift) * 1000:.2f} mA, "
          f"max {shift.max() * 1000:.1f} mA")
    return ref


# ======================================================================
# arc line detection
# ======================================================================

def _gaussian(x, amp, mu, sigma, offset):
    return amp * np.exp(-(x - mu) ** 2 / (2.0 * sigma ** 2)) + offset


def _gaussian_sloped(x, amp, mu, sigma, offset, slope):
    """A Gaussian on a straight line rather than on a constant.

    The line is written about mu so that offset stays the background
    level under the line itself and the two are close to uncorrelated,
    which keeps the fit well conditioned.

    Parameters
    ----------
    x : ndarray
        Positions at which to evaluate, in pixels.
    amp : float
        Peak amplitude above the background, in counts.
    mu : float
        Centre of the Gaussian, in pixels.
    sigma : float
        Standard deviation of the Gaussian, in pixels.
    offset : float
        Background level at mu, in counts.
    slope : float
        Gradient of the background, in counts per pixel.

    Returns
    -------
    values : ndarray
        Profile values, same shape as x.
    """
    return (amp * np.exp(-(x - mu) ** 2 / (2.0 * sigma ** 2))
            + offset + slope * (x - mu))


# columns of the detection array
DET_PIXEL, DET_SIGMA, DET_AMP, DET_SNR, DET_PIXERR, DET_FITRES = range(6)


def reference_lines_for(atlas_path, m_lambda_centre, m_lambda_half_range, n_pixels,
                        line_sigma_pixels, amplitude_min=200.0, dominance=5.0):
    """Load the atlas and keep the lines this instrument can use.

    The isolation test needs the resolution width, which is derived from
    the approximate dispersion implied by m_lambda_centre and
    m_lambda_half_range. A seed supplies those when building a solution,
    the saved surface when applying one.

    Parameters
    ----------
    atlas_path : str
        Path to the ThAr line list.
    m_lambda_centre : float
        m * lambda at the centre of the detector, in Angstrom.
    m_lambda_half_range : float
        Half the span of m * lambda across the detector, in Angstrom.
    n_pixels : int
        Number of pixels along an order.
    line_sigma_pixels : float
        Instrumental line profile sigma, in pixels.
    amplitude_min : float, optional
        Least amplitude of a surviving line. Default 200.0.
    dominance : float, optional
        Required amplitude ratio over blend neighbours. Default 5.0.

    Returns
    -------
    ref : ReferenceLines
        The usable atlas lines with effective wavelengths.
    """
    sel_wave, sel_amp, full_wave, full_amp = load_atlas(atlas_path)
    fwhm_pixels = 2.355 * line_sigma_pixels

    def resolution_angstrom(wave):
        return (fwhm_pixels * wave * 2.0 * m_lambda_half_range
                / ((n_pixels - 1) * m_lambda_centre))

    return select_reference_lines(sel_wave, sel_amp, full_wave, full_amp,
                                  resolution_angstrom, amplitude_min=amplitude_min,
                                  dominance=dominance)


def detect_arc_lines(spectrum, expected_sigma_pixels=3.3, detection_sigma=7.0,
                     saturation=45000.0, min_separation=8, half_window=9,
                     width_tolerance=(0.5, 2.0), max_fit_residual=0.15,
                     continuum_window=201):
    """Find ThAr emission lines in one extracted order and centroid them.

    Peaks are found above a median-filtered continuum and fitted with a
    Gaussian. A line is rejected if it reaches saturation, if its fitted
    sigma falls outside width_tolerance times expected_sigma_pixels, or
    if the RMS of the fit residual exceeds max_fit_residual as a fraction
    of the amplitude. Each of those biases the centroid.

    Parameters
    ----------
    spectrum : ndarray
        Extracted arc spectrum of one order, shape (n_pixels,).
    expected_sigma_pixels : float, optional
        Instrumental line sigma in pixels, used as the fit starting value
        and as the width reference. Default 3.3.
    detection_sigma : float, optional
        Peak prominence threshold, in units of the robust noise.
        Default 7.0.
    saturation : float, optional
        Counts at or above which a peak is discarded. Default 45000.0.
    min_separation : int, optional
        Least separation between detected peaks, in pixels. Default 8.
    half_window : int, optional
        Half width of the fitting window, in pixels. Default 9.
    width_tolerance : tuple of float, optional
        Lower and upper multiples of expected_sigma_pixels that a fitted
        sigma must lie between. Default (0.5, 2.0).
    max_fit_residual : float, optional
        Largest allowed RMS fit residual, as a fraction of the line
        amplitude. Default 0.15.
    continuum_window : int, optional
        Median filter width used for the continuum, in pixels.
        Default 201.

    Returns
    -------
    detections : ndarray or None
        Shape (n, 6), columns (pixel, sigma in pixels, amplitude, signal
        to noise, pixel error, fractional fit residual). Shape (0, 6) if
        no line survives. None if the spectrum has non-finite values.
    """
    s = np.asarray(spectrum, float)

    # A single non-finite sample used to discard the whole order. The
    # weighted extraction returns NaN for any row whose aperture runs off
    # the edge of the detector, which happens on the bluest and reddest
    # orders, so one such row was costing exactly the orders with the
    # fewest lines to spare. Short gaps are interpolated across for the
    # peak finding and the fits, and any line whose window touches a gap
    # is dropped afterwards.
    finite = np.isfinite(s)
    if not finite.any():
        return None
    if not finite.all():
        if finite.sum() < max(continuum_window, 3 * half_window):
            return None
        s = np.interp(np.arange(len(s)), np.flatnonzero(finite), s[finite])

    continuum = median_filter(s, continuum_window)
    resid = s - continuum
    mad = np.median(np.abs(resid - np.median(resid)))
    noise = 1.4826 * mad if mad > 0 else np.std(resid)
    if noise <= 0:
        return np.zeros((0, 6))

    # Local noise, so the threshold follows the blaze. One number for the
    # whole order is too strict where the order is faint and too loose at
    # its peak, and the faint ends are where pixel coverage and order
    # overlap are decided.
    if getattr(config, "ARC_LOCAL_NOISE", True):
        local = 1.4826 * median_filter(np.abs(resid - median_filter(resid, 51)),
                                       size=continuum_window)
        # never let a quiet stretch drive the threshold below the read
        # noise of the order as a whole
        prominence = detection_sigma * np.maximum(local, 0.35 * noise)
    else:
        prominence = detection_sigma * noise

    peaks, _ = find_peaks(resid, prominence=prominence, distance=min_separation)

    sloped = getattr(config, "ARC_LINE_LINEAR_BACKGROUND", True)

    out = []
    for p in peaks:
        if s[p] >= saturation:
            continue
        lo = max(0, p - half_window)
        hi = min(len(s), p + half_window + 1)
        if hi - lo < 7:
            continue
        if not finite[lo:hi].all():
            continue                      # this window was interpolated, not measured
        x = np.arange(lo, hi)
        if sloped:
            # Fit the raw spectrum with the background as a free slope.
            # Subtracting a running median first leaves a tilted pedestal
            # under the line, and a tilt under a symmetric Gaussian moves
            # its fitted centre.
            y = s[lo:hi]
            base = np.median(y)
            p0 = (max(resid[p], 1e-6), float(p), expected_sigma_pixels, base, 0.0)
            model_fn = _gaussian_sloped
        else:
            y = resid[lo:hi]
            p0 = (y.max(), float(p), expected_sigma_pixels, 0.0)
            model_fn = _gaussian
        try:
            popt, pcov = curve_fit(model_fn, x, y, p0=p0)
        except (RuntimeError, ValueError, TypeError):
            continue
        amp, mu, sigma = popt[0], popt[1], abs(popt[2])
        if amp <= 0 or not (lo + 1 < mu < hi - 1):
            continue
        if not (width_tolerance[0] * expected_sigma_pixels < sigma
                < width_tolerance[1] * expected_sigma_pixels):
            continue
        model = model_fn(x, *popt)
        frac_resid = np.sqrt(np.mean((y - model) ** 2)) / amp
        if frac_resid > max_fit_residual:
            continue
        try:
            pixel_err = float(np.sqrt(abs(pcov[1, 1])))
        except (IndexError, TypeError, ValueError):
            pixel_err = np.nan
        if not np.isfinite(pixel_err) or pixel_err <= 0:
            pixel_err = 1.0
        out.append((mu, sigma, amp, amp / noise, pixel_err, frac_resid))

    return np.array(out) if out else np.zeros((0, 6))


def detect_all_orders(orders, **kwargs):
    """Run detect_arc_lines on every order's extracted ThAr spectrum.

    Prints a summary of the line counts, since a collapse in the number
    found is the first sign of a bad extraction.

    Parameters
    ----------
    orders : list
        Traced orders, each with a thar_spectrum attribute holding an
        ndarray or None.
    **kwargs
        Passed through to detect_arc_lines.

    Returns
    -------
    detections : list
        One entry per order, each an (n, 6) detection array, or None for
        a dead or blank trace.
    """
    detections = []
    n_total = 0
    n_dead = 0
    for order in orders:
        if order.thar_spectrum is None:
            detections.append(None)
            n_dead += 1
            continue
        d = detect_arc_lines(order.thar_spectrum, **kwargs)
        if d is None:
            n_dead += 1
        else:
            n_total += len(d)
        detections.append(d)
    counts = [0 if d is None else len(d) for d in detections]
    print(f"detect_all_orders: {n_total} usable ThAr lines over {len(orders) - n_dead} orders "
          f"(median {int(np.median(counts))} per order, "
          f"range {min(counts)}-{max(counts)}, {n_dead} dead traces)")
    return detections


# ======================================================================
# the solution itself
# ======================================================================

class WavelengthSolution:
    """Maps (pixel, order number) to wavelength for the whole detector.

    m * lambda is modelled as three physical camera terms, 1, u / d and
    f / d with u = pixel - pixel_centre and d = sqrt(f**2 + u**2), which
    are the constant, sin(beta) and cos(beta) of a camera of focal length
    f pixels. Each term carries its own low-degree polynomial in
    normalised order number, covering the out-of-plane cos(gamma) term
    and any detector rotation. An optional Chebyshev correction in
    (pixel, order number) is added to that.

    Defined at module level so that it survives pickling into a saved
    master solution.

    Parameters
    ----------
    focal_pixels : float
        Camera focal length, in pixels.
    coefficients : ndarray
        Physical-term coefficients, shape (3 * (m_degree + 1),).
    m_degree : int
        Degree of the polynomial in normalised order number.
    n_pixels : int
        Number of pixels along an order.
    m_min : float
        Lowest order number, used to normalise m.
    m_max : float
        Highest order number, used to normalise m.
    correction : ndarray, optional
        Chebyshev coefficients in (pixel, order number), shape
        (degree_pixel + 1, degree_m + 1). Default None, meaning no
        correction term is applied.
    """

    def __init__(self, focal_pixels, coefficients, m_degree, n_pixels,
                 m_min, m_max, correction=None):
        self.focal_pixels = float(focal_pixels)
        self.coefficients = np.asarray(coefficients, float)
        self.m_degree = int(m_degree)
        self.n_pixels = int(n_pixels)
        self.m_min = float(m_min)
        self.m_max = float(m_max)
        self.correction = None if correction is None else np.asarray(correction, float)

    # -- coordinate normalisation ------------------------------------
    def _y_hat(self, pixel):
        return 2.0 * (np.asarray(pixel, float) - (self.n_pixels - 1) / 2.0) / (self.n_pixels - 1)

    def _m_hat(self, m):
        mid = (self.m_max + self.m_min) / 2.0
        half = max((self.m_max - self.m_min) / 2.0, 1.0)
        return (np.asarray(m, float) - mid) / half

    def design(self, pixel, m):
        """Design matrix of the physical part of the model.

        Parameters
        ----------
        pixel : ndarray
            Pixel positions along the order, shape (n,).
        m : ndarray
            Order number of each position, shape (n,).

        Returns
        -------
        matrix : ndarray
            Shape (n, 3 * (m_degree + 1)).
        """
        u = np.asarray(pixel, float) - (self.n_pixels - 1) / 2.0
        d = np.sqrt(self.focal_pixels ** 2 + u ** 2)
        basis = np.vstack([np.ones_like(u), u / d, self.focal_pixels / d])
        mh = self._m_hat(m)
        cols = [basis[k] * mh ** j for k in range(3) for j in range(self.m_degree + 1)]
        return np.vstack(cols).T

    # -- evaluation --------------------------------------------------
    def m_lambda(self, pixel, m):
        """Order number times wavelength, in Angstrom.

        Parameters
        ----------
        pixel : ndarray or float
            Position along the order, in pixels.
        m : ndarray or float
            Order number, broadcast against pixel.

        Returns
        -------
        m_lambda : ndarray
            m * wavelength in Angstrom, shape of the broadcast inputs.
        """
        pixel = np.asarray(pixel, float)
        m = np.asarray(m, float)
        pixel, m = np.broadcast_arrays(pixel, m)
        flat_p, flat_m = pixel.ravel(), m.ravel()
        value = self.design(flat_p, flat_m) @ self.coefficients
        if self.correction is not None:
            value = value + C.chebval2d(self._y_hat(flat_p), self._m_hat(flat_m),
                                        self.correction)
        return value.reshape(pixel.shape)

    def wavelength(self, pixel, m):
        """Wavelength in Angstrom at a pixel in a given order.

        Parameters
        ----------
        pixel : ndarray or float
            Position along the order, in pixels.
        m : ndarray or float
            Order number, broadcast against pixel.

        Returns
        -------
        wavelength : ndarray
            Wavelength in Angstrom, shape of the broadcast inputs.
        """
        pixel = np.asarray(pixel, float)
        m = np.asarray(m, float)
        pixel, m = np.broadcast_arrays(pixel, m)
        return self.m_lambda(pixel, m) / m

    def dispersion(self, pixel, m):
        """Local dispersion, in Angstrom per pixel.

        Parameters
        ----------
        pixel : ndarray or float
            Pixel position along the order.
        m : ndarray or float
            Order number.

        Returns
        -------
        dispersion : ndarray
            Signed Angstrom per pixel at (pixel, m), measured over one pixel
            centred on it.
        """
        pixel = np.asarray(pixel, float)
        return (self.wavelength(pixel + 0.5, m) - self.wavelength(pixel - 0.5, m))

    def order_axis(self, m, n_pixels=None):
        """Wavelength axis of one whole order.

        Parameters
        ----------
        m : float
            Order number.
        n_pixels : int, optional
            Number of pixels to evaluate. Default None, meaning the
            solution's own n_pixels.

        Returns
        -------
        wavelength : ndarray
            Wavelength in Angstrom at each pixel, shape (n_pixels,).
        """
        n = self.n_pixels if n_pixels is None else n_pixels
        pix = np.arange(n)
        return self.wavelength(pix, np.full(n, float(m)))

    def for_order(self, m):
        """Return a one-argument callable pixel -> wavelength for one order.

        Lets per-order code hold something that behaves like a per-order
        polynomial.

        Parameters
        ----------
        m : float
            Order number.

        Returns
        -------
        axis : OrderWavelength
            Callable, picklable slice of this solution.
        """
        return OrderWavelength(self, m)


class OrderWavelength:
    """One order's slice of a WavelengthSolution. Picklable and callable.

    pixel_shift moves the axis across the detector, which is the
    correction for flexure. velocity_ms scales the axis instead, which is
    the correction for a Doppler shift and, being the same fraction in
    every order, cannot disturb order overlap. diagnose_frame_offset
    says which of the two an exposure needs.

    Parameters
    ----------
    solution : WavelengthSolution
        Surface to slice.
    m : float
        Order number.
    pixel_shift : float, optional
        Shift subtracted from the pixel coordinate, in pixels.
        Default 0.0.
    velocity_ms : float, optional
        Velocity in m/s divided out of the wavelength. Default 0.0.
    """

    def __init__(self, solution, m, pixel_shift=0.0, velocity_ms=0.0):
        self.solution = solution
        self.m = float(m)
        self.pixel_shift = float(pixel_shift)
        self.velocity_ms = float(velocity_ms)

    def __call__(self, pixel):
        """Wavelength in Angstrom at a pixel in this order.

        Parameters
        ----------
        pixel : ndarray or float
            Position along the order, in pixels.

        Returns
        -------
        wavelength : ndarray
            Wavelength in Angstrom, with pixel_shift and velocity_ms
            applied, same shape as pixel.
        """
        pixel = np.asarray(pixel, float) - self.pixel_shift
        w = self.solution.wavelength(pixel, np.full(pixel.shape, self.m))
        return w / (1.0 + self.velocity_ms / C_LIGHT_MS)


class PolynomialSurface:
    """m * lambda as a plain Chebyshev surface in (pixel, order number).

    Used only to warm the solution up. It is well conditioned at low
    degree and tolerates bad starting matches, but it extrapolates poorly
    outside the pixel range where lines were matched, so solve() hands
    over to WavelengthSolution once there are enough matches to pin the
    camera geometry down.

    Parameters
    ----------
    coefficients : ndarray
        Chebyshev coefficients, shape (degree_pixel + 1, degree_m + 1).
    n_pixels : int
        Number of pixels along an order.
    m_min : float
        Lowest order number, used to normalise m.
    m_max : float
        Highest order number, used to normalise m.
    """

    def __init__(self, coefficients, n_pixels, m_min, m_max):
        self.coefficients = np.asarray(coefficients, float)
        self.n_pixels = int(n_pixels)
        self.m_min = float(m_min)
        self.m_max = float(m_max)

    def _y_hat(self, pixel):
        return 2.0 * (np.asarray(pixel, float) - (self.n_pixels - 1) / 2.0) / (self.n_pixels - 1)

    def _m_hat(self, m):
        mid = (self.m_max + self.m_min) / 2.0
        half = max((self.m_max - self.m_min) / 2.0, 1.0)
        return (np.asarray(m, float) - mid) / half

    def m_lambda(self, pixel, m):
        """Order number times wavelength, in Angstrom.

        Parameters
        ----------
        pixel : ndarray or float
            Position along the order, in pixels.
        m : ndarray or float
            Order number, broadcast against pixel.

        Returns
        -------
        m_lambda : ndarray
            m * wavelength in Angstrom.
        """
        pixel, m = np.broadcast_arrays(np.asarray(pixel, float), np.asarray(m, float))
        return C.chebval2d(self._y_hat(pixel), self._m_hat(m), self.coefficients)

    def wavelength(self, pixel, m):
        """Wavelength in Angstrom at a pixel in a given order.

        Parameters
        ----------
        pixel : ndarray or float
            Position along the order, in pixels.
        m : ndarray or float
            Order number, broadcast against pixel.

        Returns
        -------
        wavelength : ndarray
            Wavelength in Angstrom.
        """
        pixel, m = np.broadcast_arrays(np.asarray(pixel, float), np.asarray(m, float))
        return self.m_lambda(pixel, m) / m

    def dispersion(self, pixel, m):
        """Local dispersion, in Angstrom per pixel.

        Parameters
        ----------
        pixel : ndarray or float
            Position along the order, in pixels.
        m : ndarray or float
            Order number, broadcast against pixel.

        Returns
        -------
        dispersion : ndarray
            Angstrom per pixel, from the difference across one pixel.
        """
        pixel = np.asarray(pixel, float)
        return self.wavelength(pixel + 0.5, m) - self.wavelength(pixel - 0.5, m)


def fit_polynomial_surface(matches, n_pixels, degrees, clip_sigma=4.0, iterations=4):
    """Fit the warm-up surface by robust weighted least squares.

    Parameters
    ----------
    matches : MatchSet
        Matched lines to fit.
    n_pixels : int
        Number of pixels along an order.
    degrees : tuple of int
        Chebyshev degrees in (pixel, order number).
    clip_sigma : float, optional
        Residuals beyond this many robust sigma are rejected on the next
        iteration. Default 4.0.
    iterations : int, optional
        Largest number of clip and refit iterations. Default 4.

    Returns
    -------
    solution : PolynomialSurface
        The fitted surface.
    keep : ndarray
        Boolean mask over matches of the lines that survived clipping,
        shape (len(matches),).
    residuals : ndarray
        Residual in m * lambda for every match, in Angstrom, shape
        (len(matches),).
    """
    keep = np.ones(len(matches), bool)
    solution = None
    residuals = np.zeros(len(matches))
    n_free = (degrees[0] + 1) * (degrees[1] + 1)
    for _ in range(iterations):
        V = C.chebvander2d(
            2.0 * (matches.pixel[keep] - (n_pixels - 1) / 2.0) / (n_pixels - 1),
            (matches.m[keep] - (matches.m.max() + matches.m.min()) / 2.0)
            / max((matches.m.max() - matches.m.min()) / 2.0, 1.0),
            degrees)
        sw = np.sqrt(matches.weight[keep])
        coef, *_ = np.linalg.lstsq(V * sw[:, None], matches.m_lambda[keep] * sw, rcond=None)
        solution = PolynomialSurface(coef.reshape(degrees[0] + 1, degrees[1] + 1),
                                     n_pixels, matches.m.min(), matches.m.max())
        residuals = matches.m_lambda - solution.m_lambda(matches.pixel, matches.m)
        scatter = 1.4826 * np.median(np.abs(residuals[keep] - np.median(residuals[keep])))
        new_keep = np.abs(residuals) < clip_sigma * max(scatter, 1e-12)
        if np.array_equal(new_keep, keep) or new_keep.sum() < 3 * n_free:
            break
        keep = new_keep
    return solution, keep, residuals


class LinearSeed:
    """Starting guess m * lambda = A + B * y_hat, shared by every order.

    y_hat runs from -1 at pixel 0 to +1 at the last pixel. This is the
    simplest model that a single clicked doublet pins down, and it is not
    used past the first matching pass.

    Parameters
    ----------
    A : float
        Zero point of m * lambda, in Angstrom.
    B : float
        Half the span of m * lambda across the detector, in Angstrom.
    n_pixels : int
        Number of pixels along an order.
    """

    def __init__(self, A, B, n_pixels):
        self.A = float(A)
        self.B = float(B)
        self.n_pixels = int(n_pixels)

    def _y_hat(self, pixel):
        return 2.0 * (np.asarray(pixel, float) - (self.n_pixels - 1) / 2.0) / (self.n_pixels - 1)

    def m_lambda(self, pixel, m):
        """Order number times wavelength, in Angstrom.

        The value does not depend on m, since the seed is one line shared
        by every order. m is accepted so the seed can stand in for a
        fitted solution.

        Parameters
        ----------
        pixel : ndarray or float
            Position along the order, in pixels.
        m : ndarray or float
            Order number, used only for broadcasting.

        Returns
        -------
        m_lambda : ndarray
            A + B * y_hat, in Angstrom.
        """
        pixel, m = np.broadcast_arrays(np.asarray(pixel, float), np.asarray(m, float))
        return self.A + self.B * self._y_hat(pixel)

    def wavelength(self, pixel, m):
        """Wavelength in Angstrom at a pixel in a given order.

        Parameters
        ----------
        pixel : ndarray or float
            Position along the order, in pixels.
        m : ndarray or float
            Order number, broadcast against pixel.

        Returns
        -------
        wavelength : ndarray
            Wavelength in Angstrom.
        """
        pixel, m = np.broadcast_arrays(np.asarray(pixel, float), np.asarray(m, float))
        return self.m_lambda(pixel, m) / m

    def dispersion(self, pixel, m):
        """Local dispersion, in Angstrom per pixel.

        Parameters
        ----------
        pixel : ndarray or float
            Position along the order, in pixels.
        m : ndarray or float
            Order number, broadcast against pixel.

        Returns
        -------
        dispersion : ndarray
            Angstrom per pixel, from the difference across one pixel.
        """
        return self.wavelength(np.asarray(pixel, float) + 0.5, m) - \
               self.wavelength(np.asarray(pixel, float) - 0.5, m)


# ======================================================================
# seeding
# ======================================================================

def seed_from_doublet(pixels, wavelengths, m, n_pixels, K=None):
    """Turn one clicked doublet in one order into a seed for every order.

    Two lines of known wavelength at known pixels in order m give the
    local dispersion, and because m * lambda is a shared function of
    pixel that single measurement fixes the linear seed for the whole
    detector:

        B = m * (dlambda/dy) * (n_pixels - 1) / 2
        A = m * lambda_1 - B * y_hat(pixel_1)

    Parameters
    ----------
    pixels : ndarray
        The two measured line positions, in pixels.
    wavelengths : ndarray
        The two rest wavelengths, in Angstrom.
    m : int
        Order number the two lines were measured in.
    n_pixels : int
        Number of pixels along an order.
    K : float, optional
        Nominal grating constant in Angstrom, used only to report how far
        the seed zero point lands from it. Default None, meaning that
        comparison is not printed.

    Returns
    -------
    seed : LinearSeed
        Linear seed for the whole detector.

    Raises
    ------
    ValueError
        If exactly two lines are not supplied, or if the two clicked
        positions are less than 5 pixels apart.
    """
    pixels = np.asarray(pixels, float)
    wavelengths = np.asarray(wavelengths, float)
    if len(pixels) != 2 or len(wavelengths) != 2:
        raise ValueError("seed_from_doublet needs exactly two lines")

    order = np.argsort(wavelengths)
    pixels, wavelengths = pixels[order], wavelengths[order]
    d_pixel = pixels[1] - pixels[0]
    if abs(d_pixel) < 5:
        raise ValueError(f"clicked doublet is only {d_pixel:.1f} pixels apart, "
                         f"that is almost certainly the same line clicked twice")

    dispersion = (wavelengths[1] - wavelengths[0]) / d_pixel   # A / pixel
    B = m * dispersion * (n_pixels - 1) / 2.0

    y_hat = 2.0 * (pixels - (n_pixels - 1) / 2.0) / (n_pixels - 1)
    A = float(np.mean(m * wavelengths - B * y_hat))

    print(f"seed_from_doublet: order m={m}, {abs(d_pixel):.1f} px between "
          f"{wavelengths[0]:.2f} and {wavelengths[1]:.2f} A")
    print(f"  -> dispersion {dispersion:.5f} A/px at m={m}; "
          f"m*lambda = {A:.0f} + {B:.0f} * y_hat")
    print(f"  -> implies {2 * B / m:.1f} A of coverage per order at m={m} "
          f"(free spectral range there is {A / m ** 2:.1f} A)")
    if K is not None:
        print(f"  -> nominal grating constant K = {K:.0f} A, seed zero point differs by "
              f"{A - K:+.0f} A ({100 * (A - K) / K:+.2f}%)")
    return LinearSeed(A, B, n_pixels)


def lock_seed(detections, order_numbers, reference, seed, n_pixels,
              zero_point_range=3000.0, slope_fraction=0.06, n_slope=121,
              grid_step=1.0, expected_sigma_pixels=3.3, verbose=True):
    """Refine the linear seed against the atlas, all orders at once.

    Each order is correlated against its own atlas window in m * lambda,
    and the correlations are summed, so the zero point every order agrees
    on wins. A single order carries too few lines to pick its own zero
    point reliably. Orders with fewer than 5 detected lines are ignored.
    The returned SNR is the peak of the summed correlation over its
    robust scatter: tens means locked, single digits means the seed, the
    order numbers or the atlas selection is wrong.

    Parameters
    ----------
    detections : list
        One (n, 6) detection array or None per order.
    order_numbers : list of int
        Order number of each entry in detections.
    reference : ReferenceLines
        Atlas lines to correlate against.
    seed : LinearSeed
        Starting seed, giving the centre of the search.
    n_pixels : int
        Number of pixels along an order.
    zero_point_range : float, optional
        Half width of the zero-point search in m * lambda, in Angstrom.
        Default 3000.0.
    slope_fraction : float, optional
        Fractional half range of the dispersion search around seed.B.
        Default 0.06.
    n_slope : int, optional
        Number of dispersion values tried. Default 121.
    grid_step : float, optional
        Step of the correlation grid in m * lambda, in Angstrom.
        Default 1.0.
    expected_sigma_pixels : float, optional
        Line sigma in pixels, setting the correlation kernel width.
        Default 3.3.
    verbose : bool, optional
        Print the locked seed and its SNR. Default True.

    Returns
    -------
    seed : LinearSeed
        The refined seed.
    snr : float
        Peak of the summed correlation over its robust scatter.

    Raises
    ------
    ValueError
        If no order has at least 5 detected lines.
    """
    y_hat = 2.0 * (np.arange(n_pixels) - (n_pixels - 1) / 2.0) / (n_pixels - 1)

    # observed line combs, one per order, in "m*lambda relative to A" space
    live = [(i, d) for i, d in enumerate(detections) if d is not None and len(d) >= 5]
    if not live:
        raise ValueError("no orders have enough detected ThAr lines to lock a seed")

    best = None
    slopes = seed.B * np.linspace(1.0 - slope_fraction, 1.0 + slope_fraction, n_slope)
    for B in slopes:
        u_lo = seed.A - zero_point_range - B
        u_hi = seed.A + zero_point_range + B
        u_obs = np.arange(-B, B + grid_step, grid_step)
        u_ref = np.arange(u_lo, u_hi + grid_step, grid_step)
        n_obs, n_ref = len(u_obs), len(u_ref)
        sigma_u = expected_sigma_pixels * 2.0 * B / (n_pixels - 1)
        half = max(int(4 * sigma_u / grid_step), 1)
        kernel = np.exp(-0.5 * (np.arange(-half, half + 1) * grid_step / sigma_u) ** 2)

        size = 1
        while size < n_obs + n_ref:
            size *= 2
        obs = np.zeros((len(live), size))
        ref_grid = np.zeros((len(live), size))

        for k, (i, d) in enumerate(live):
            m = float(order_numbers[i])
            # observed comb: unit-height marks at the detected line positions
            pos = np.interp(d[:, DET_PIXEL], np.arange(n_pixels), B * y_hat)
            idx = np.round((pos + B) / grid_step).astype(int)
            idx = idx[(idx >= 0) & (idx < n_obs)]
            row = np.zeros(n_obs)
            np.add.at(row, idx, 1.0)
            row = np.convolve(row, kernel, mode="same")
            if row.std() > 0:
                obs[k, :n_obs] = (row - row.mean()) / row.std()

            j0, j1 = np.searchsorted(reference.wave, [u_lo / m, u_hi / m])
            if j1 - j0 < 3:
                continue
            centres = reference.wave[j0:j1] * m
            amps = np.sqrt(reference.amplitude[j0:j1])
            jdx = np.round((centres - u_lo) / grid_step).astype(int)
            ok = (jdx >= 0) & (jdx < n_ref)
            row = np.zeros(n_ref)
            np.add.at(row, jdx[ok], amps[ok])
            row = np.convolve(row, kernel, mode="same")
            if row.std() > 0:
                ref_grid[k, :n_ref] = (row - row.mean()) / row.std()

        cc = np.fft.irfft(np.conj(np.fft.rfft(obs, size, axis=1))
                          * np.fft.rfft(ref_grid, size, axis=1), size, axis=1)
        total = cc.sum(axis=0)[:n_ref]
        lag = int(np.argmax(total))
        A = u_lo + lag * grid_step + B
        med = np.median(total)
        mad = 1.4826 * np.median(np.abs(total - med))
        snr = (total[lag] - med) / mad if mad > 0 else 0.0
        if best is None or snr > best[2]:
            best = (A, B, snr)

    A, B, snr = best
    if verbose:
        print(f"lock_seed: m*lambda = {A:.0f} + {B:.0f} * y_hat  "
              f"(seed was {seed.A:.0f} + {seed.B:.0f}); "
              f"zero point moved {A - seed.A:+.0f} A, dispersion "
              f"{100 * (B / seed.B - 1):+.2f}%")
        print(f"  correlation SNR = {snr:.1f} over {len(live)} orders "
              f"({'locked' if snr > 15 else 'WEAK, do not trust this'})")
    return LinearSeed(A, B, n_pixels), snr


def check_order_number_offset(detections, order_numbers, reference, locked, n_pixels,
                              offsets=(-2, -1, 0, 1, 2), **lock_kwargs):
    """Confirm the order numbers by re-locking with m shifted by a constant.

    Absolute order number is the one thing overlap agreement cannot
    check: shifting every m by one gives a self-consistent solution with
    every wavelength wrong by roughly one free spectral range. Only the
    atlas separates them. The zero-point search is widened here, because
    relabelling every order moves the best zero point by a whole order's
    worth of m * lambda. Prints a warning if a non-zero offset locks
    better, and treats a margin below 2.0 as no confirmation.

    Parameters
    ----------
    detections : list
        One (n, 6) detection array or None per order.
    order_numbers : list of int
        Current order numbers.
    reference : ReferenceLines
        Atlas lines to correlate against.
    locked : LinearSeed
        Seed from lock_seed, used as the centre of each trial search.
    n_pixels : int
        Number of pixels along an order.
    offsets : tuple of int, optional
        Constant shifts in m to try. Default (-2, -1, 0, 1, 2).
    **lock_kwargs
        Override the lock_seed keywords used for each trial.

    Returns
    -------
    results : dict
        Lock SNR for each offset, keyed by the offset.
    margin : float
        SNR of the current numbering divided by that of the best rival
        offset.
    """
    print("Order-number check (re-locking with m shifted by a constant):")
    median_m = float(np.median([m for m in order_numbers if m is not None]))
    kwargs = dict(slope_fraction=0.04, n_slope=17,
                  zero_point_range=2.6 * locked.A / median_m)
    kwargs.update(lock_kwargs)
    results = {}
    for off in offsets:
        shifted = [m + off for m in order_numbers]
        _, snr = lock_seed(detections, shifted, reference, locked, n_pixels,
                           verbose=False, **kwargs)
        results[off] = snr
        print(f"  m {off:+d}: lock SNR = {snr:6.1f}{'   <-- current numbering' if off == 0 else ''}")
    best = max(results, key=results.get)
    runner_up = max(snr for off, snr in results.items() if off != 0)
    margin = results[0] / max(runner_up, 1e-9)
    if best != 0:
        print(f"  WARNING: offset {best:+d} locks better than the current numbering. "
              f"Either an anchor is wrong or an order is missing from the trace list. "
              f"Every wavelength is out by about one free spectral range until this "
              f"is fixed, and no internal check will notice.")
    elif margin >= 2.0:
        print(f"  current numbering wins by {margin:.1f}x, "
              f"order numbers confirmed against the atlas.")
    else:
        print(f"  current numbering wins, but only by {margin:.1f}x. That is not a "
              f"confirmation: the atlas cannot tell these numberings apart, which "
              f"usually means the lock itself is poor.")
    return results, margin


# ======================================================================
# matching and fitting
# ======================================================================

class MatchSet:
    """Detected lines paired with reference wavelengths, ready to fit.

    All arrays are parallel and have shape (n,).

    Parameters
    ----------
    pixel : ndarray
        Measured line centroids, in pixels.
    m : ndarray
        Order number of each line.
    m_lambda : ndarray
        Reference m * lambda of each line, in Angstrom.
    weight : ndarray
        Least-squares weight, 1 / sigma**2 in m * lambda.
    pixel_err : ndarray
        Centroid uncertainty of each line, in pixels.
    snr : ndarray
        Signal to noise of each detected line.
    order_index : ndarray
        Index into the orders list that each line came from.
    """

    def __init__(self, pixel, m, m_lambda, weight, pixel_err, snr, order_index):
        self.pixel = pixel
        self.m = m
        self.m_lambda = m_lambda
        self.weight = weight
        self.pixel_err = pixel_err
        self.snr = snr
        self.order_index = order_index

    def __len__(self):
        """Number of matched lines held."""
        return len(self.pixel)

    def subset(self, mask):
        """Select a subset of the matches.

        Parameters
        ----------
        mask : ndarray
            Boolean mask or integer index array over the matches.

        Returns
        -------
        matches : MatchSet
            A new MatchSet holding the selected entries.
        """
        return MatchSet(self.pixel[mask], self.m[mask], self.m_lambda[mask],
                        self.weight[mask], self.pixel_err[mask], self.snr[mask],
                        self.order_index[mask])


def match_lines(model, detections, order_numbers, reference, n_pixels,
                tolerance_pixels, ambiguity_factor=2.5, max_pixel_error=1.0):
    """Pair each detected line with a reference line, refusing ambiguity.

    A match is kept only if the nearest reference line is within
    tolerance_pixels and the next-nearest is at least ambiguity_factor
    times the tolerance further away. A match that could plausibly have
    been its neighbour contributes a wrong wavelength as readily as a
    right one, and sigma clipping downstream does not reliably find those
    again. The tolerance is given in pixels and converted per order using
    the model's local dispersion, so one number means the same thing in
    the blue and in the red.

    Parameters
    ----------
    model : object
        Any model exposing wavelength(pixel, m) and dispersion(pixel, m),
        such as LinearSeed, PolynomialSurface or WavelengthSolution.
    detections : list
        One (n, 6) detection array or None per order.
    order_numbers : list of int
        Order number of each entry in detections.
    reference : ReferenceLines
        Atlas lines to match against.
    n_pixels : int
        Number of pixels along an order.
    tolerance_pixels : float
        Largest accepted separation between a detection and its reference
        line, in pixels.
    ambiguity_factor : float, optional
        Required separation of the second-nearest reference line, in
        units of the tolerance. Default 2.5.
    max_pixel_error : float, optional
        Detections with a larger centroid error, in pixels, are
        discarded. Default 1.0.

    Returns
    -------
    matches : MatchSet
        The accepted matches, empty if nothing matched.
    """
    pix, ms, mlam, wgt, perr, snr, oidx = [], [], [], [], [], [], []
    ref_w = reference.wave
    ref_eff = reference.eff_wave

    for i, det in enumerate(detections):
        # An order can be traced and have its lines detected while still
        # carrying no order number, which is what a trace outside the
        # master's range gets. float(None) raises, so this used to take
        # the whole night's reduction down over one extra edge trace.
        if det is None or len(det) == 0 or order_numbers[i] is None:
            continue
        m = float(order_numbers[i])
        mu = det[:, DET_PIXEL]
        m_col = np.full(len(mu), m)
        lam = model.wavelength(mu, m_col)

        disp = np.abs(model.dispersion(mu, m_col))
        tol = tolerance_pixels * disp

        j = np.clip(np.searchsorted(ref_w, lam), 1, len(ref_w) - 1)
        cand = np.vstack([j - 1, j])
        dist = np.abs(ref_w[cand] - lam[None, :])
        which = np.argmin(dist, axis=0)
        cols = np.arange(len(lam))
        nearest = cand[which, cols]
        d1 = dist[which, cols]

        neigh = np.clip(np.vstack([nearest - 2, nearest - 1, nearest + 1, nearest + 2]),
                        0, len(ref_w) - 1)
        d2 = np.min(np.abs(ref_w[neigh] - lam[None, :]), axis=0)

        ok = (d1 < tol) & (d2 > ambiguity_factor * tol) & (det[:, DET_PIXERR] < max_pixel_error)
        if not ok.any():
            continue

        pix.append(mu[ok])
        ms.append(m_col[ok])
        mlam.append(ref_eff[nearest[ok]] * m)
        perr.append(det[ok, DET_PIXERR])
        snr.append(det[ok, DET_SNR])
        oidx.append(np.full(ok.sum(), i))
        # weight in m*lambda space: a pixel error becomes a wavelength error
        # through the local dispersion, and m*lambda scales it by m
        sigma_mlam = det[ok, DET_PIXERR] * disp[ok] * m
        wgt.append(1.0 / np.maximum(sigma_mlam, 1e-6) ** 2)

    if not pix:
        return MatchSet(*[np.array([]) for _ in range(7)])
    return MatchSet(np.concatenate(pix), np.concatenate(ms), np.concatenate(mlam),
                    np.concatenate(wgt), np.concatenate(perr), np.concatenate(snr),
                    np.concatenate(oidx))


FOCAL_LIMITS = (3000.0, 200000.0)   # plausible camera focal lengths, in pixels


def fit_solution(matches, n_pixels, m_degree=2, correction_degree=None,
                 focal_guess=None, clip_sigma=4.0, max_iterations=6):
    """Fit the physical surface, and its correction, to matched lines.

    The camera focal length is the only non-linear parameter. It is
    scanned on a grid over FOCAL_LIMITS with an exact weighted
    least-squares solve inside, then refined on a local grid, so there is
    no optimiser to get stuck and no starting-value sensitivity. Clipping
    and refitting repeat until the kept set stops changing.

    Parameters
    ----------
    matches : MatchSet
        Matched lines to fit.
    n_pixels : int
        Number of pixels along an order.
    m_degree : int, optional
        Degree of the polynomial in normalised order number. Default 2.
    correction_degree : tuple of int, optional
        Chebyshev degrees in (pixel, order number) for the correction
        term. Default None, meaning no correction is fitted.
    focal_guess : float, optional
        Accepted but unused; the focal length is searched over the full
        range on every iteration. Default None.
    clip_sigma : float, optional
        Residuals beyond this many robust sigma are rejected on the next
        iteration. Default 4.0.
    max_iterations : int, optional
        Largest number of clip and refit iterations. Default 6.

    Returns
    -------
    solution : WavelengthSolution
        The fitted surface.
    keep : ndarray
        Boolean mask over matches of the surviving lines, shape
        (len(matches),).
    residuals : ndarray
        Residual in m * lambda for every match, in Angstrom, shape
        (len(matches),).

    Raises
    ------
    ValueError
        If fewer than 6 * (m_degree + 1) matched lines are supplied.
    """
    if len(matches) < 3 * (m_degree + 1) * 2:
        raise ValueError(f"only {len(matches)} matched lines, not enough to fit")

    m_min, m_max = matches.m.min(), matches.m.max()
    # The focal length is the only non-linear parameter. As f grows the
    # basis flattens towards a straight line and the correction term can
    # imitate the lost curvature, so a fit to poor matches drifts to
    # f = infinity. Bounded by FOCAL_LIMITS and searched on a grid.
    keep = np.ones(len(matches), bool)
    solution = None
    residuals = np.zeros(len(matches))

    joint = (correction_degree is not None
             and getattr(config, "JOINT_CORRECTION_FIT", True))
    n_physical = 3 * (m_degree + 1)

    # The correction's own m-only columns duplicate the physical basis.
    # design() emits {1, u/d, f/d} x m_hat**j, and its first group is
    # {m_hat**0 ... m_hat**m_degree}; chebvander2d's first pixel group is
    # T_0(y_hat) * T_j(m_hat) for j = 0 .. degree_m, which spans functions
    # of m_hat alone up to degree_m. Stacking both makes the system rank
    # deficient, so the duplicates are dropped and the physical terms keep
    # them.
    #
    # Only the overlap duplicates: the two groups share polynomials up to
    # min(m_degree, degree_m). Dropping the whole first group instead
    # would throw away degree_m - m_degree columns that nothing else
    # spans, which is real freedom lost, and choose_degrees puts four such
    # combinations on its grid.
    correction_keep = None
    if correction_degree is not None:
        n_correction = (correction_degree[0] + 1) * (correction_degree[1] + 1)
        duplicated = min(correction_degree[1], m_degree)
        correction_keep = np.arange(n_correction) > duplicated

    def correction_matrix(trial, pixel, m):
        return C.chebvander2d(trial._y_hat(pixel), trial._m_hat(m), correction_degree)

    def best_focal(grid):
        """Scan the focal length against the physical terms alone.

        Deliberately without the correction. The correction can imitate
        the curvature the physical basis loses as f grows, so including
        it here leaves f almost unconstrained and it wanders off to the
        bound, taking the meaning of focal_pixels and the FOCAL_LIMITS
        sanity check with it. f is a property of the camera, so it is
        measured from the part of the model that describes the camera.
        """
        best = None
        sw = np.sqrt(matches.weight[keep])
        target = matches.m_lambda[keep] * sw
        for f in grid:
            trial = WavelengthSolution(f, np.zeros(n_physical), m_degree,
                                       n_pixels, m_min, m_max)
            A = trial.design(matches.pixel[keep], matches.m[keep])
            coef, *_ = np.linalg.lstsq(A * sw[:, None], target, rcond=None)
            r = matches.m_lambda[keep] - A @ coef
            chi = np.sqrt(np.average(r ** 2, weights=matches.weight[keep]))
            if best is None or chi < best[0]:
                best = (chi, f, coef)
        return best

    def solve_together(trial):
        """Refit the physical and correction coefficients in one system.

        Fitting the correction to what the physical fit left behind is
        one step of backfitting, which only reaches the least-squares
        optimum when the two bases are orthogonal, and these are not. At
        a fixed focal length both sets of coefficients are linear, so the
        optimum is one lstsq call away.
        """
        A = trial.design(matches.pixel[keep], matches.m[keep])
        V = correction_matrix(trial, matches.pixel[keep], matches.m[keep])
        A = np.hstack([A, V[:, correction_keep]])
        sw = np.sqrt(matches.weight[keep])
        coef, *_ = np.linalg.lstsq(A * sw[:, None],
                                   matches.m_lambda[keep] * sw, rcond=None)
        trial.coefficients = coef[:n_physical]
        flat = np.zeros((correction_degree[0] + 1) * (correction_degree[1] + 1))
        flat[correction_keep] = coef[n_physical:]
        trial.correction = flat.reshape(correction_degree[0] + 1,
                                        correction_degree[1] + 1)

    for _ in range(max_iterations):
        # Full range every time, then a local refinement. Starting from
        # where the previous pass left f strands it at a bound, because
        # the early passes prefer no curvature and a local search cannot
        # walk back from there.
        coarse = best_focal(np.geomspace(*FOCAL_LIMITS, 80))
        best = best_focal(np.clip(np.linspace(0.85 * coarse[1], 1.18 * coarse[1], 40),
                                  *FOCAL_LIMITS))
        if coarse[0] < best[0]:
            best = coarse

        solution = WavelengthSolution(best[1], best[2], m_degree, n_pixels,
                                      m_min, m_max)

        if correction_degree is not None:
            if joint:
                solve_together(solution)
            else:
                # the original two-stage fit: the correction takes what
                # the physical terms left behind, and they are not refitted
                resid = matches.m_lambda - solution.m_lambda(matches.pixel, matches.m)
                V = correction_matrix(solution, matches.pixel[keep], matches.m[keep])
                sw = np.sqrt(matches.weight[keep])
                cc, *_ = np.linalg.lstsq(V * sw[:, None], resid[keep] * sw, rcond=None)
                solution.correction = cc.reshape(correction_degree[0] + 1,
                                                 correction_degree[1] + 1)

        residuals = matches.m_lambda - solution.m_lambda(matches.pixel, matches.m)

        # Clip on the residual divided by its own uncertainty. Residuals
        # are in m*lambda, so at equal wavelength accuracy a line in the
        # bluest order carries 2.6x the residual of one in the reddest
        # here, and a single cut in m*lambda would clip the blue orders
        # hardest while letting the red ones through.
        if getattr(config, "WEIGHTED_CLIPPING", True):
            # About the median rather than about zero. The two agree
            # whenever the model's constant term has pulled the residuals
            # onto zero, which is the usual case, and differ exactly when
            # it has not: a pass where the matches are still one-sided.
            statistic = residuals * np.sqrt(matches.weight)
            centre = np.median(statistic[keep])
        else:
            statistic = residuals
            centre = 0.0
        scatter = 1.4826 * np.median(np.abs(statistic[keep] - np.median(statistic[keep])))
        new_keep = np.abs(statistic - centre) < clip_sigma * max(scatter, 1e-12)
        if new_keep.sum() == keep.sum() and np.array_equal(new_keep, keep):
            break
        keep = new_keep
        if keep.sum() < 3 * (m_degree + 1) * 3:
            break

    return solution, keep, residuals


def _rms_angstrom(matches, residuals, mask):
    return float(np.sqrt(np.mean((residuals[mask] / matches.m[mask]) ** 2)))


def default_schedule(correction_degree=(4, 2)):
    """Return the tightening ladder for the physical-model passes.

    Two uncorrected passes come first, so the camera geometry is fitted
    to the lines alone, then the correction is enabled and the matching
    tolerance closes to 2.0 pixels.

    Parameters
    ----------
    correction_degree : tuple of int, optional
        Chebyshev degrees in (pixel, order number) used from the third
        pass onward. Default (4, 2).

    Returns
    -------
    schedule : list of tuple
        (tolerance_pixels, correction_degree) for each of six passes.
    """
    return [(4.0, None), (3.0, None),
            (2.5, correction_degree), (2.0, correction_degree),
            (2.0, correction_degree), (2.0, correction_degree)]


def solve(orders, detections, reference, seed, n_pixels,
          schedule=None, warmup=None, m_degree=2, correction_degree=None,
          verbose=True):
    """Go from a linear seed to the final wavelength solution.

    Matching and fitting alternate, starting loose and tightening. The
    model is refitted before the tolerance shrinks, never after, so each
    tightening applies to a model that has already improved. The physical
    basis takes over from the warm-up polynomial for the scheduled
    passes; because it predicts the ends of each order correctly from the
    middle, the lines there get matched on the following pass, and the
    ends are where orders overlap.

    Parameters
    ----------
    orders : list
        Traced orders carrying order_number.
    detections : list
        One (n, 6) detection array or None per order.
    reference : ReferenceLines
        Atlas lines to match against.
    seed : LinearSeed
        Starting model for the first matching pass.
    n_pixels : int
        Number of pixels along an order.
    schedule : list of tuple, optional
        (tolerance_pixels, correction_degree) per physical-model pass.
        Default None, meaning default_schedule().
    warmup : list of tuple, optional
        (tolerance_pixels, chebyshev_degrees) per warm-up pass. Default
        None, meaning five passes tightening from 30.0 to 4.0 pixels
        while the degree rises.
    m_degree : int, optional
        Degree of the polynomial in normalised order number. Default 2.
    correction_degree : tuple of int, optional
        Correction degrees used to build the default schedule. Default
        None, meaning (4, 2).
    verbose : bool, optional
        Print one line per pass and the fitted focal length. Default
        True.

    Returns
    -------
    solution : WavelengthSolution
        The final surface.
    matches : MatchSet
        Matches from the final pass.
    keep : ndarray
        Boolean mask over matches of the surviving lines.
    residuals : ndarray
        Residual in m * lambda for every match, in Angstrom.

    Raises
    ------
    RuntimeError
        If a warm-up pass matches fewer than 60 lines.
    """
    if warmup is None:
        # The tolerance comes down slowly and the degree rises slowly. Too
        # large a drop in tolerance discards the lines at the ends of each
        # order before the model can reach them; too high a degree too
        # early lets the polynomial chase badly matched lines.
        warmup = [(30.0, (1, 0)), (15.0, (2, 1)), (8.0, (3, 1)),
                  (5.0, (4, 2)), (4.0, (4, 2))]
    if schedule is None:
        schedule = default_schedule((4, 2) if correction_degree is None
                                    else correction_degree)

    order_numbers = [o.order_number for o in orders]
    model = seed
    solution = matches = keep = residuals = None

    def report(tag, tol, extra):
        rms = _rms_angstrom(matches, residuals, keep)
        lam = matches.m_lambda[keep] / matches.m[keep]
        velocity = np.sqrt(np.mean(((residuals[keep] / matches.m[keep]) / lam
                                    * C_LIGHT_MS) ** 2))
        print(f"  {tag}: tol={tol:4.1f} px  {extra:16s}  matched={len(matches):5d}  "
              f"kept={keep.sum():5d}  rms={rms * 1000:6.2f} mA ({velocity:6.0f} m/s)")

    # --- warm-up on a plain polynomial surface -------------------------
    # The physical basis must not be fitted to bad matches: its curvature
    # is pinned down only by correct lines, and wrong ones drive it to a
    # huge focal length, which the correction term then has to fake. The
    # warm-up surface makes the matches reliable before the hand-over.
    for step, (tolerance, degrees) in enumerate(warmup):
        matches = match_lines(model, detections, order_numbers, reference, n_pixels,
                              tolerance)
        if len(matches) < 60:
            raise RuntimeError(
                f"only {len(matches)} lines matched at tolerance {tolerance} px. "
                f"The seed is not close enough to the truth. Check the lock SNR, the"
                f"order numbers, and the clicked doublet before anything else.")
        solution, keep, residuals = fit_polynomial_surface(matches, n_pixels, degrees)
        model = solution
        if verbose:
            report(f"warm-up {step + 1}", tolerance, f"cheb{degrees}")

    # --- the real fit ---------------------------------------------------
    focal_guess = None
    for step, (tolerance, correction_degree) in enumerate(schedule):
        matches = match_lines(model, detections, order_numbers, reference, n_pixels,
                              tolerance)
        solution, keep, residuals = fit_solution(
            matches, n_pixels, m_degree=m_degree,
            correction_degree=correction_degree, focal_guess=focal_guess)
        focal_guess = solution.focal_pixels
        model = solution
        if verbose:
            report(f"pass    {step + 1}", tolerance,
                   f"corr={correction_degree}")

    if verbose:
        f = solution.focal_pixels
        if FOCAL_LIMITS[0] * 1.02 < f < FOCAL_LIMITS[1] * 0.98:
            note = "inside the plausible range"
        else:
            note = ("AT A BOUND, so the physical basis is not constraining "
                    "here, so treat this as a plain polynomial fit and do not trust "
                    "it beyond the pixels that were matched")
        print(f"  camera focal length fitted at {f:.0f} px ({note})")
    return solution, matches, keep, residuals


# ======================================================================
# quality: does this solution deserve to be believed
# ======================================================================

class QualityReport:
    """The checks measured on a solution, and whether they all passed.

    Held as an object rather than printed, so that a driver can refuse to
    save a solution that fails.

    Attributes
    ----------
    checks : list of tuple
        (name, passed, message) for every check added.
    stats : dict
        Measured quantities, keyed by name.
    """

    def __init__(self):
        self.checks = []      # (name, passed, message)
        self.stats = {}

    def add(self, name, passed, message):
        """Record the outcome of one check.

        Parameters
        ----------
        name : str
            Short name of the check, printed in the report.
        passed : bool
            Whether the check passed.
        message : str
            One line stating the measured value and the threshold.
        """
        self.checks.append((name, bool(passed), message))

    @property
    def passed(self):
        """True if every recorded check passed."""
        return all(p for _, p, _ in self.checks)

    def show(self):
        """Print every check and the overall verdict."""
        print("\n" + "=" * 72)
        print("WAVELENGTH SOLUTION QUALITY")
        print("=" * 72)
        for name, ok, message in self.checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:24s} {message}")
        print("-" * 72)
        print(f"  OVERALL: {'PASS' if self.passed else 'FAIL'}")
        print("=" * 72 + "\n")


def cross_validate(matches, keep, n_pixels, m_degree, correction_degree,
                   folds=5, seed=0):
    """Measure the solution error on lines it was not fitted to.

    The kept matches are split into folds; each fold is predicted by a
    solution fitted to the others. The RMS of a fit against its own
    training lines always improves with more free parameters, so it
    cannot show whether the extra freedom describes the instrument or the
    noise, and this can. Predictions beyond 5 robust sigma are dropped,
    since a fold can leave a gap that has to be extrapolated into.

    Parameters
    ----------
    matches : MatchSet
        All matched lines.
    keep : ndarray
        Boolean mask over matches selecting the lines to use.
    n_pixels : int
        Number of pixels along an order.
    m_degree : int
        Degree of the polynomial in normalised order number.
    correction_degree : tuple of int or None
        Chebyshev degrees for the correction term, or None for none.
    folds : int, optional
        Number of cross-validation folds. Default 5.
    seed : int, optional
        Seed of the shuffle that assigns lines to folds. Default 0.

    Returns
    -------
    rms : float
        Held-out RMS wavelength error in Angstrom, or NaN if every fold
        failed to fit.
    """
    sub = matches.subset(keep)
    idx = np.arange(len(sub))
    np.random.default_rng(seed).shuffle(idx)
    errors = []
    for f in range(folds):
        test = idx[f::folds]
        train = np.setdiff1d(idx, test)
        try:
            sol, _, _ = fit_solution(sub.subset(train), n_pixels, m_degree=m_degree,
                                     correction_degree=correction_degree)
        except ValueError:
            continue
        pred = sol.m_lambda(sub.pixel[test], sub.m[test])
        errors.append((sub.m_lambda[test] - pred) / sub.m[test])
    if not errors:
        return np.nan
    e = np.concatenate(errors)
    scatter = 1.4826 * np.median(np.abs(e - np.median(e)))
    e = e[np.abs(e) < 5 * scatter]          # a fold can leave a gap it must extrapolate into
    return float(np.sqrt(np.mean(e ** 2)))


def choose_degrees(matches, keep, n_pixels,
                   m_degrees=(1, 2, 3), correction_degrees=(None, (2, 1), (3, 2), (4, 2), (5, 3)),
                   folds=5, tolerance=0.05, verbose=True):
    """Pick the model complexity by cross-validation.

    Every combination of m_degree and correction degree is
    cross-validated. Among the models that cross-validate within
    tolerance of the best, the one with the fewest free parameters wins.

    Parameters
    ----------
    matches : MatchSet
        All matched lines.
    keep : ndarray
        Boolean mask over matches selecting the lines to use.
    n_pixels : int
        Number of pixels along an order.
    m_degrees : tuple of int, optional
        Polynomial degrees in order number to try. Default (1, 2, 3).
    correction_degrees : tuple, optional
        Correction degrees to try, each a (pixel, order number) pair or
        None. Default (None, (2, 1), (3, 2), (4, 2), (5, 3)).
    folds : int, optional
        Number of cross-validation folds. Default 5.
    tolerance : float, optional
        Fractional margin on the best cross-validated RMS within which a
        simpler model is preferred. Default 0.05.
    verbose : bool, optional
        Print the result of every model tried. Default True.

    Returns
    -------
    m_degree : int
        Chosen polynomial degree in order number.
    correction_degree : tuple of int or None
        Chosen Chebyshev correction degrees.
    cv : float
        Cross-validated RMS of the chosen model, in Angstrom.

    Raises
    ------
    RuntimeError
        If cross-validation failed for every model tried.
    """
    if verbose:
        print(f"Choosing model complexity by {folds}-fold cross-validation:")
    results = []
    for md in m_degrees:
        for cd in correction_degrees:
            try:
                cv = cross_validate(matches, keep, n_pixels, md, cd, folds=folds)
            except Exception:
                continue
            if not np.isfinite(cv):
                continue
            n_free = 3 * (md + 1) + (0 if cd is None else (cd[0] + 1) * (cd[1] + 1))
            results.append((cv, n_free, md, cd))
            if verbose:
                print(f"    m_degree={md}  correction={str(cd):8s} "
                      f"({n_free:2d} free) -> cross-validated rms {cv * 1000:6.2f} mA")
    if not results:
        raise RuntimeError("cross-validation failed for every model tried")
    best_cv = min(r[0] for r in results)
    within = [r for r in results if r[0] <= best_cv * (1.0 + tolerance)]
    cv, n_free, md, cd = min(within, key=lambda r: (r[1], r[0]))
    if verbose:
        print(f"  -> m_degree={md}, correction={cd}: cross-validated rms "
              f"{cv * 1000:.2f} mA with {n_free} free parameters "
              f"(best was {best_cv * 1000:.2f} mA; the simplest model within "
              f"{tolerance:.0%} of it wins)")
    return md, cd, cv


def overlap_agreement(orders, solution, spectrum_attr="thar_spectrum",
                      pixel_shift=0.0, velocity_ms=0.0,
                      min_overlap_angstrom=2.0, max_velocity_ms=20000.0,
                      oversample=5.0, min_contrast=3.0,
                      tilt=0.0, tilt_reference_m=0.0):
    """Measure whether adjacent orders agree, without using the atlas.

    Where two orders overlap they record the same lamp lines on different
    parts of the detector. Both are put on a common log-wavelength grid
    and cross-correlated, so the offset of the peak from zero is the
    disagreement in velocity, independent of the line list. Orders whose
    free spectral range is wider than the detector do not overlap and are
    skipped, as are pairs whose correlation peak stands less than
    min_contrast robust sigma above the median.

    Parameters
    ----------
    orders : list
        Traced orders carrying order_number and the named spectrum.
    solution : WavelengthSolution
        Surface used to build each order's wavelength axis.
    spectrum_attr : str, optional
        Attribute holding the spectrum to correlate. Default
        "thar_spectrum".
    pixel_shift : float, optional
        Shift subtracted from the pixel coordinate before the axis is
        evaluated, in pixels. Default 0.0.
    velocity_ms : float, optional
        Velocity in m/s divided out of each axis. Default 0.0.
    tilt : float, optional
        Change in pixel_shift per unit order number, in pixels per order.
        Default 0.0, meaning one rigid shift for every order.
    tilt_reference_m : float, optional
        Order number pixel_shift belongs to. Default 0.0. Ignored when
        tilt is 0.
    min_overlap_angstrom : float, optional
        Least wavelength overlap worth correlating, in Angstrom.
        Default 2.0.
    max_velocity_ms : float, optional
        Half width of the lag search, in m/s. Default 20000.0.
    oversample : float, optional
        Samples of the log-wavelength grid per detector pixel.
        Default 5.0.
    min_contrast : float, optional
        Least height of the correlation peak above the median, in robust
        sigma, for a pair to be reported. Default 3.0.

    Returns
    -------
    velocities : ndarray
        Disagreement of each usable pair, in m/s.
    pairs : list of tuple
        (order_number, order_number) for each entry in velocities.
    """
    velocities, pairs = [], []
    usable = [o for o in orders if o.order_number is not None
              and getattr(o, spectrum_attr, None) is not None]
    usable.sort(key=lambda o: -o.order_number)

    def axis_of(order, n):
        shift = pixel_shift
        if tilt:
            shift = pixel_shift + tilt * (order.order_number - tilt_reference_m)
        pixels = np.arange(n) - shift
        w = solution.wavelength(pixels, np.full(n, float(order.order_number)))
        return w / (1.0 + velocity_ms / C_LIGHT_MS)

    for a, b in zip(usable[:-1], usable[1:]):
        if abs(a.order_number - b.order_number) != 1:
            continue
        spec_a = getattr(a, spectrum_attr)
        spec_b = getattr(b, spectrum_attr)
        n = len(spec_a)
        wa = axis_of(a, n)
        wb = axis_of(b, n)
        lo = max(wa.min(), wb.min())
        hi = min(wa.max(), wb.max())
        if hi - lo < min_overlap_angstrom:
            continue

        # Log-wavelength grid, so one lag is one constant velocity,
        # sampled several times per detector pixel. Quantising at the
        # pixel scale would be too blunt to check anything.
        pixel_step = np.median(np.abs(np.diff(wa)))
        step = pixel_step / hi / oversample
        grid = np.arange(np.log(lo), np.log(hi), step)
        if len(grid) < 128:
            continue
        # np.interp needs its sample points increasing and does not check;
        # given a decreasing axis it returns the end value everywhere
        # rather than raising, so this gate would go on reporting
        # plausible numbers while testing nothing. measure_frame_shift
        # already guards this and overlap_agreement was missed.
        if wa[0] > wa[-1]:
            wa, spec_a = wa[::-1], np.asarray(spec_a)[::-1]
        if wb[0] > wb[-1]:
            wb, spec_b = wb[::-1], np.asarray(spec_b)[::-1]
        fa = np.interp(np.exp(grid), wa, spec_a)
        fb = np.interp(np.exp(grid), wb, spec_b)
        fa = fa - median_filter(fa, max(9, len(fa) // 8) | 1)
        fb = fb - median_filter(fb, max(9, len(fb) // 8) | 1)
        if fa.std() <= 0 or fb.std() <= 0:
            continue
        fa /= fa.std()
        fb /= fb.std()

        max_lag = int(min(len(grid) // 3, max_velocity_ms / C_LIGHT_MS / step))
        if max_lag < 5:
            continue
        lags = np.arange(-max_lag, max_lag + 1)
        cc = np.array([np.dot(fa[max(0, l):len(fa) + min(0, l)],
                              fb[max(0, -l):len(fb) + min(0, -l)]) /
                       max(len(fa) - abs(l), 1) for l in lags])
        if not np.isfinite(cc).all():
            continue
        k = int(np.argmax(cc))
        if not (0 < k < len(cc) - 1):
            continue                      # peak is outside the search range
        # a peak that does not stand out means the two orders share no
        # lines worth correlating, so no number is reported for them
        scatter = 1.4826 * np.median(np.abs(cc - np.median(cc)))
        if scatter <= 0 or (cc[k] - np.median(cc)) / scatter < min_contrast:
            continue
        denom = cc[k - 1] - 2 * cc[k] + cc[k + 1]
        sub = 0.5 * (cc[k - 1] - cc[k + 1]) / denom if denom != 0 else 0.0
        velocities.append((lags[k] + sub) * step * C_LIGHT_MS)
        pairs.append((a.order_number, b.order_number))

    return np.array(velocities), pairs


def residual_trends(matches, keep, residuals, n_pixels, n_bins=8):
    """Bin the fit residuals against pixel and against order number.

    A correct model leaves residuals with no structure. Drift with pixel
    means the dispersion shape is underfitted; drift with order number
    means the cross-order term is. Bins holding 5 lines or fewer are
    omitted.

    Parameters
    ----------
    matches : MatchSet
        All matched lines.
    keep : ndarray
        Boolean mask over matches selecting the lines to bin.
    residuals : ndarray
        Residual in m * lambda for every match, in Angstrom, shape
        (len(matches),).
    n_pixels : int
        Number of pixels along an order.
    n_bins : int, optional
        Number of bins along each axis. Default 8.

    Returns
    -------
    pixel_bins : list of tuple
        (label, n, mean, rms) per pixel bin, with mean and rms in
        milli-Angstrom.
    order_bins : list of tuple
        (label, n, mean, rms) per order-number bin, same units.
    """
    r = residuals[keep] / matches.m[keep] * 1000.0
    p = matches.pixel[keep]
    m = matches.m[keep]

    pixel_bins = []
    edges = np.linspace(0, n_pixels, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (p >= lo) & (p < hi)
        if s.sum() > 5:
            pixel_bins.append((f"{int(lo)}-{int(hi)}", int(s.sum()),
                               float(r[s].mean()), float(np.sqrt(np.mean(r[s] ** 2)))))

    order_bins = []
    m_edges = np.linspace(m.min(), m.max() + 1e-9, n_bins + 1)
    for lo, hi in zip(m_edges[:-1], m_edges[1:]):
        s = (m >= lo) & (m < hi)
        if s.sum() > 5:
            order_bins.append((f"m={int(lo)}-{int(hi)}", int(s.sum()),
                               float(r[s].mean()), float(np.sqrt(np.mean(r[s] ** 2)))))
    return pixel_bins, order_bins


def assess(orders, solution, matches, keep, residuals, n_pixels, lock_snr,
           m_degree, correction_degree, order_number_margin=None,
           max_rms_ma=15.0, max_cv_ma=20.0, max_overlap_ms=600.0,
           min_orders_with_lines=0.6, min_pixel_coverage=0.75,
           max_trend_ma=6.0, min_lock_snr=15.0, min_order_number_margin=2.0,
           verbose=True):
    """Run every quality check on a solution and return the report.

    The checks cover the atlas lock, the order numbering, the fitted
    residuals, cross-validation, order and pixel coverage, residual
    trends, and adjacent-order overlap. The thresholds are defaults, not
    laws; set them from what the instrument achieves.

    Parameters
    ----------
    orders : list
        Traced orders carrying order_number and thar_spectrum.
    solution : WavelengthSolution
        Surface to assess.
    matches : MatchSet
        Matches from the final fitting pass.
    keep : ndarray
        Boolean mask over matches of the surviving lines.
    residuals : ndarray
        Residual in m * lambda for every match, in Angstrom, shape
        (len(matches),).
    n_pixels : int
        Number of pixels along an order.
    lock_snr : float
        Correlation SNR reported by lock_seed.
    m_degree : int
        Polynomial degree in order number used for the fit.
    correction_degree : tuple of int or None
        Chebyshev correction degrees used for the fit.
    order_number_margin : float, optional
        Margin from check_order_number_offset. Default None, meaning the
        order-numbering check is skipped.
    max_rms_ma : float, optional
        Largest accepted fitted RMS, in milli-Angstrom. Default 15.0.
    max_cv_ma : float, optional
        Largest accepted cross-validated RMS, in milli-Angstrom.
        Default 20.0.
    max_overlap_ms : float, optional
        Largest accepted median overlap disagreement, in m/s.
        Default 600.0.
    min_orders_with_lines : float, optional
        Least fraction of numbered orders that must carry 4 or more
        matched lines. Default 0.6.
    min_pixel_coverage : float, optional
        Least fraction of the detector that matched lines must span in
        the median order. Default 0.75.
    max_trend_ma : float, optional
        Largest accepted binned mean residual, in milli-Angstrom.
        Default 6.0.
    min_lock_snr : float, optional
        Least accepted lock SNR. Default 15.0.
    min_order_number_margin : float, optional
        Least accepted order-numbering margin. Default 2.0.
    verbose : bool, optional
        Print the report and the residual tables. Default True.

    Returns
    -------
    report : QualityReport
        The checks, their outcomes, and the measured statistics.
    """
    report = QualityReport()
    lam = matches.m_lambda[keep] / matches.m[keep]
    resid_ang = residuals[keep] / matches.m[keep]
    rms = float(np.sqrt(np.mean(resid_ang ** 2)))
    velocity = float(np.sqrt(np.mean((resid_ang / lam * C_LIGHT_MS) ** 2)))

    report.stats["n_matched"] = int(keep.sum())
    report.stats["rms_angstrom"] = rms
    report.stats["rms_ms"] = velocity
    report.stats["focal_pixels"] = solution.focal_pixels

    report.add("atlas lock", lock_snr >= min_lock_snr,
               f"correlation SNR {lock_snr:.1f} (need >= {min_lock_snr:.0f})")

    if order_number_margin is not None:
        report.stats["order_number_margin"] = order_number_margin
        report.add("order numbers", order_number_margin >= min_order_number_margin,
                   f"the assigned numbering locks {order_number_margin:.1f}x better than "
                   f"m +/-1 or +/-2 (need >= {min_order_number_margin:.0f}x)")

    report.add("line residuals", rms * 1000 <= max_rms_ma,
               f"{keep.sum()} lines, rms {rms * 1000:.2f} mA = {velocity:.0f} m/s "
               f"(need <= {max_rms_ma:.0f} mA)")

    cv = cross_validate(matches, keep, n_pixels, m_degree, correction_degree)
    report.stats["cv_angstrom"] = cv
    report.add("cross-validation", np.isfinite(cv) and cv * 1000 <= max_cv_ma,
               f"held-out rms {cv * 1000:.2f} mA "
               f"({cv / max(rms, 1e-12):.2f}x the fitted rms; "
               f"a large ratio means over-fitting)")

    # how much of each order is actually constrained by matched lines
    coverages = []
    per_order = {}
    for order in orders:
        if order.order_number is None:
            continue
        s = keep & (matches.m == order.order_number)
        if s.sum() >= 4:
            span = matches.pixel[s].max() - matches.pixel[s].min()
            coverages.append(span / n_pixels)
            per_order[order.order_number] = (
                int(s.sum()),
                float(np.sqrt(np.mean((residuals[s] / order.order_number) ** 2)) * 1000),
                span / n_pixels)
    report.stats["per_order"] = per_order

    n_with = len(per_order)
    n_total = sum(1 for o in orders if o.order_number is not None)
    report.add("order coverage", n_with >= min_orders_with_lines * n_total,
               f"{n_with}/{n_total} orders carry >= 4 matched lines "
               f"(need >= {min_orders_with_lines:.0%})")

    median_cov = float(np.median(coverages)) if coverages else 0.0
    report.stats["median_pixel_coverage"] = median_cov
    report.add("pixel coverage", median_cov >= min_pixel_coverage,
               f"matched lines span {median_cov:.0%} of the detector in the median order "
               f"(need >= {min_pixel_coverage:.0%}; less means the ends are extrapolated, "
               f"and the ends are where orders overlap)")

    pixel_bins, order_bins = residual_trends(matches, keep, residuals, n_pixels)
    worst_pixel = max((abs(b[2]) for b in pixel_bins), default=0.0)
    worst_order = max((abs(b[2]) for b in order_bins), default=0.0)
    report.stats["pixel_bins"] = pixel_bins
    report.stats["order_bins"] = order_bins
    report.add("residual trends", max(worst_pixel, worst_order) <= max_trend_ma,
               f"largest binned mean residual {max(worst_pixel, worst_order):.2f} mA "
               f"vs pixel/order (need <= {max_trend_ma:.0f} mA; structure here means "
               f"the model is too stiff)")

    velocities, pairs = overlap_agreement(orders, solution)
    if len(velocities):
        med = float(np.median(np.abs(velocities)))
        worst = float(np.max(np.abs(velocities)))
        report.stats["overlap_ms"] = velocities
        report.stats["overlap_pairs"] = pairs
        report.add("order overlap", med <= max_overlap_ms,
                   f"{len(velocities)} overlapping pairs agree to {med:.0f} m/s median, "
                   f"{worst:.0f} m/s worst (need <= {max_overlap_ms:.0f} m/s)")
    else:
        report.add("order overlap", False,
                   "no adjacent orders overlap in wavelength, cannot check")

    if verbose:
        report.show()
        print("Residuals by pixel:")
        for label, n, mean, rms_b in pixel_bins:
            print(f"    {label:>12s}  n={n:4d}  mean={mean:+7.2f} mA  rms={rms_b:6.2f} mA")
        print("Residuals by order number:")
        for label, n, mean, rms_b in order_bins:
            print(f"    {label:>12s}  n={n:4d}  mean={mean:+7.2f} mA  rms={rms_b:6.2f} mA")
        if len(velocities):
            print(f"Adjacent-order overlap (independent of the atlas): "
                  f"median {np.median(np.abs(velocities)):.0f} m/s over {len(velocities)} pairs")
            worst_idx = np.argsort(-np.abs(velocities))[:5]
            for k in worst_idx:
                print(f"    m={pairs[k][0]}/{pairs[k][1]}: {velocities[k]:+8.0f} m/s")

        thin = [(m, v) for m, v in sorted(per_order.items()) if v[0] < 8 or v[2] < 0.5]
        worst = sorted(per_order.items(), key=lambda kv: -kv[1][1])[:6]
        print("Weakest orders (few lines, or lines covering little of the order):")
        if thin:
            for m, (n, rms_o, cov) in thin:
                print(f"    m={m:4d}  {n:3d} lines  rms {rms_o:6.2f} mA  covering {cov:4.0%} "
                      f"of the order, {solution.order_axis(m)[0]:.0f}-"
                      f"{solution.order_axis(m)[-1]:.0f} A")
        else:
            print("    none, every order has 8+ lines spanning at least half of it")
        print("Highest per-order residuals:")
        for m, (n, rms_o, cov) in worst:
            print(f"    m={m:4d}  {n:3d} lines  rms {rms_o:6.2f} mA  covering {cov:4.0%}")
        missing = [o.order_number for o in orders
                   if o.order_number is not None and o.order_number not in per_order]
        if missing:
            print(f"Orders with too few matched lines to check ({len(missing)}): "
                  f"{sorted(missing)}")
            print("    these still get a wavelength axis from the global surface, which is "
                  "the point of fitting one, but nothing in their own data confirms it")
    return report


def measure_frame_shift(anchors, solution, verbose=True):
    """Measure the pixel shift between a frame and the arc frame.

    Stellar lines of known wavelength are compared against the solution.
    They take no part in the fit, so their agreement on a single shift is
    an independent check as well as the correction itself. The result
    does not say which of two things the offset is, and that decides how
    it may be applied: flexure moves every order by the same number of
    pixels, and evaluating the solution at pixel - shift undoes it
    exactly, while a Doppler shift moves every order by the same fraction
    of its wavelength. Correcting one as though it were the other leaves
    adjacent orders disagreeing, worst in the red.
    diagnose_frame_offset() tells the two apart.

    Parameters
    ----------
    anchors : list of tuple
        (order_number, measured_pixel, rest_wavelength) per line, with
        the pixel in pixels and the wavelength in Angstrom. Anchors
        outside their order's wavelength range are skipped with a
        message.
    solution : WavelengthSolution
        Surface giving the predicted pixel of each rest wavelength.
    verbose : bool, optional
        Print the per-anchor table and any warning. Default True.

    Returns
    -------
    shift : float
        Median offset in pixels, or NaN if no anchor was usable.
    scatter : float
        Largest deviation of an anchor from the median, in pixels; 0.0
        for a single anchor, NaN if no anchor was usable.
    rows : list of tuple
        (order_number, wavelength, measured_pixel, predicted_pixel,
        shift_pixels, velocity_ms) per usable anchor.
    """
    rows = []
    for m, pixel, wave in anchors:
        axis = solution.order_axis(m)
        if not (axis.min() < wave < axis.max()):
            print(f"  {wave:.2f} A is outside order m={m} ({axis.min():.1f}-"
                  f"{axis.max():.1f} A), so this anchor is skipped")
            continue
        predicted = float(np.interp(wave, axis, np.arange(len(axis)))
                          if axis[0] < axis[-1] else
                          np.interp(wave, axis[::-1], np.arange(len(axis))[::-1]))
        disp = float(np.abs(solution.dispersion(predicted, m)))
        shift = pixel - predicted
        rows.append((m, wave, pixel, predicted, shift, shift * disp / wave * C_LIGHT_MS))

    if not rows:
        return np.nan, np.nan, rows

    shifts = np.array([r[4] for r in rows])
    shift = float(np.median(shifts))
    scatter = float(np.max(np.abs(shifts - shift))) if len(shifts) > 1 else 0.0

    if verbose:
        print("Stellar-line check (lines that took no part in the fit):")
        for m, wave, pixel, predicted, s, v in rows:
            print(f"  m={int(m):4d}  {wave:9.2f} A  measured px {pixel:8.2f}  "
                  f"solution px {predicted:8.2f}  ->  {s:+7.2f} px ({v / 1000:+7.1f} km/s)")
        velocity = float(np.median([r[5] for r in rows]))
        print(f"  median offset {shift:+.2f} px = {velocity / 1000:+.1f} km/s; "
              f"anchors agree to within {scatter:.2f} px")
        if scatter > 3.0:
            print("  WARNING: the anchors do NOT agree on one shift. That is not "
                  "flexure, so suspect a misidentified line or a wrong order number.")
        elif abs(shift) > 5.0:
            print("  Do not apply this as a pixel shift until diagnose_frame_offset() "
                  "says it is flexure. See that function.")
    return shift, scatter, rows


def diagnose_frame_offset(orders, solution, shift_pixels, spectrum_attr="science_spectrum",
                          verbose=True):
    """Decide whether a frame's offset is flexure or a Doppler shift.

    Adjacent orders see the same wavelengths where they overlap, so the
    right correction leaves them agreeing there and the wrong one pulls
    them apart. Under flexure the orders agree once the pixel shift is
    applied and disagree without it; under a Doppler shift they agree
    with no correction, and the pixel shift breaks that agreement. A
    Doppler offset must not be removed with a pixel shift, and in general
    should not be removed from the wavelength axis at all.

    Parameters
    ----------
    orders : list
        Traced orders carrying order_number and the named spectrum.
    solution : WavelengthSolution
        Surface used to build each order's wavelength axis.
    shift_pixels : float
        Offset to test, in pixels, as measured by measure_frame_shift.
    spectrum_attr : str, optional
        Attribute holding the spectrum to correlate. Default
        "science_spectrum".
    verbose : bool, optional
        Print both measurements and the verdict. Default True.

    Returns
    -------
    result : dict
        Keys "verdict", one of "flexure", "doppler" or "unknown", and
        "uncorrected_ms" and "shifted_ms", the median overlap
        disagreement in m/s without and with the pixel shift. Both
        velocities are NaN when too few orders overlap.
    """
    uncorrected, _ = overlap_agreement(orders, solution, spectrum_attr=spectrum_attr)
    shifted, _ = overlap_agreement(orders, solution, spectrum_attr=spectrum_attr,
                                   pixel_shift=shift_pixels)
    if len(uncorrected) == 0 or len(shifted) == 0:
        if verbose:
            print("diagnose_frame_offset: not enough overlapping orders to tell")
        return {"verdict": "unknown", "uncorrected_ms": np.nan, "shifted_ms": np.nan}

    a = float(np.median(np.abs(uncorrected)))
    b = float(np.median(np.abs(shifted)))
    verdict = "doppler" if a < b else "flexure"

    if verbose:
        print(f"diagnose_frame_offset ({spectrum_attr}, offset {shift_pixels:+.1f} px):")
        print(f"  adjacent orders agree to {a:7.0f} m/s with NO correction")
        print(f"  adjacent orders agree to {b:7.0f} m/s with the pixel shift applied")
        if verdict == "doppler":
            print(f"  -> Doppler-like, by a factor of {b / max(a, 1e-9):.0f}. The light is "
                  f"shifted in wavelength, not moved across the detector, so applying "
                  f"this as a pixel shift would break the order overlap rather than "
                  f"fix anything. Leave the axis alone; the offset is a velocity to be "
                  f"measured, not calibrated away.")
        else:
            print(f"  -> flexure, by a factor of {a / max(b, 1e-9):.0f}. The spectrum has "
                  f"moved on the detector, and the pixel shift is the right correction.")
    return {"verdict": verdict, "uncorrected_ms": a, "shifted_ms": b}


# ======================================================================
# applying the solution to the orders
# ======================================================================

def attach_solution(orders, solution, pixel_shift=0.0, velocity_ms=0.0, quiet=False,
                    tilt=0.0, tilt_reference_m=None):
    """Give every numbered order a callable pixel -> wavelength.

    Sets wavelength_poly in place on each order that has an order number.
    Both corrections default to zero, which is the arc frame: observed
    wavelengths exactly as measured, and what is saved as the master.
    Move away from it only for a reason diagnose_frame_offset supports.

    Flexure does not always move every order by the same amount. Where a
    tilt is supplied the shift becomes

        shift(m) = pixel_shift + tilt * (m - tilt_reference_m)

    which is what measure_arc_shift already fits across the orders; a
    tilt of zero reproduces the single rigid shift exactly.

    Parameters
    ----------
    orders : list
        Traced orders. Each numbered order gains a wavelength_poly
        attribute holding an OrderWavelength.
    solution : WavelengthSolution
        Surface to slice per order.
    pixel_shift : float, optional
        Shift applied to the pixel coordinate, in pixels, at
        tilt_reference_m. Default 0.0.
    velocity_ms : float, optional
        Velocity in m/s divided out of the axis. Default 0.0.
    quiet : bool, optional
        Suppress the summary line. Default False.
    tilt : float, optional
        Change in the shift per unit order number, in pixels per order.
        Default 0.0, meaning one rigid shift for every order.
    tilt_reference_m : float, optional
        Order number the shift was measured at. Default None, meaning the
        mean order number of the numbered orders. Ignored when tilt is 0.

    Returns
    -------
    None
        The orders are modified in place.
    """
    numbered = [o for o in orders if o.order_number is not None]
    if tilt and tilt_reference_m is None and numbered:
        tilt_reference_m = float(np.mean([o.order_number for o in numbered]))

    n = 0
    for order in numbered:
        shift = pixel_shift
        if tilt:
            shift = pixel_shift + tilt * (order.order_number - tilt_reference_m)
        order.wavelength_poly = OrderWavelength(solution, order.order_number,
                                                pixel_shift=shift,
                                                velocity_ms=velocity_ms)
        n += 1
    if not quiet:
        extra = ""
        if pixel_shift:
            extra += f" (shifted {pixel_shift:+.2f} px)"
        if tilt and numbered:
            span = tilt * (max(o.order_number for o in numbered)
                           - min(o.order_number for o in numbered))
            extra += f" (tilted {span:+.2f} px end to end)"
        if velocity_ms:
            extra += f" (rest frame, {velocity_ms / 1000:+.1f} km/s removed)"
        print(f"attach_solution: wavelength axis attached to {n} orders{extra}")


def store_matches(orders, matches, keep):
    """Record on each order the lines it was calibrated with.

    Sets thar_pixels and thar_wavelengths in place on every numbered
    order, for plots and for the saved master solution.

    Parameters
    ----------
    orders : list
        Traced orders carrying order_number.
    matches : MatchSet
        Matches from the final fitting pass.
    keep : ndarray
        Boolean mask over matches of the surviving lines.

    Returns
    -------
    None
        The orders are modified in place.
    """
    for order in orders:
        if order.order_number is None:
            continue
        s = keep & (matches.m == order.order_number)
        order.thar_pixels = matches.pixel[s].tolist()
        order.thar_wavelengths = (matches.m_lambda[s] / matches.m[s]).tolist()


# ======================================================================
# save / reuse
# ======================================================================

class OrderIdentifier:
    """Turns a position across the detector into a physical order number.

    Order spacing varies smoothly across the detector, so position to
    order number is a smooth monotonic curve. Fitting it, rather than
    tabulating it, tolerates the small cross-dispersion drift between
    nights and extends past the orders the master itself traced.

    Parameters
    ----------
    trace_x : ndarray
        Cross-dispersion position of each traced order, in pixels.
    order_number : ndarray
        Order number of each traced order.
    degree : int, optional
        Polynomial degree, reduced to len(trace_x) - 2 where there are
        too few orders for it. Default 4.
    """

    def __init__(self, trace_x, order_number, degree=4):
        self.trace_x = np.asarray(trace_x, float)
        self.order_number = np.asarray(order_number, float)
        order = np.argsort(self.trace_x)
        self.trace_x = self.trace_x[order]
        self.order_number = self.order_number[order]
        degree = min(degree, max(1, len(self.trace_x) - 2))
        self.coefficients = np.polyfit(self.trace_x, self.order_number, degree)
        residual = self.order_number - np.polyval(self.coefficients, self.trace_x)
        self.max_residual = float(np.max(np.abs(residual)))

    def __call__(self, x):
        """Order number at a cross-dispersion position, not rounded.

        Parameters
        ----------
        x : ndarray or float
            Cross-dispersion position, in pixels.

        Returns
        -------
        m : ndarray or float
            Fractional order number at x.
        """
        return np.polyval(self.coefficients, np.asarray(x, float))

    def spacing_at(self, x):
        """Local spacing between adjacent orders, in pixels.

        Parameters
        ----------
        x : ndarray or float
            Cross-dispersion position, in pixels.

        Returns
        -------
        spacing : ndarray or float
            Pixels between adjacent orders at x, NaN where the fitted slope
            is zero.
        """
        slope = np.polyval(np.polyder(self.coefficients), np.asarray(x, float))
        return np.abs(1.0 / np.where(np.abs(slope) < 1e-12, np.nan, slope))


def save_solution(path, solution, orders, report=None, white=None, atlas_path=None):
    """Write a master solution to a pickle file.

    Orders are identified by where they sit across the detector rather
    than by index, because a faint order the tracer misses shifts every
    later index. An order's position is set by the optics, so on a later
    night a trace found at that position is that order. Stored: the
    m * lambda surface; per order its number, spatial position, matched
    lines and reference arc spectrum; and the spatial map, holding the
    white-light profile, the row it came from, and an OrderIdentifier
    that extrapolates to orders the master never traced.

    Parameters
    ----------
    path : str
        Destination file path.
    solution : WavelengthSolution
        Surface to store.
    orders : list
        Traced orders; those with both an order number and a trace centre
        are saved.
    report : QualityReport, optional
        Quality report to store alongside. Default None, meaning no
        quality entry is written.
    white : ndarray, optional
        White-light image, shape (rows, columns); its middle row is
        stored as the registration profile. Default None, meaning no
        profile is stored.
    atlas_path : str, optional
        Path of the line list used, recorded for reuse. Default None.

    Returns
    -------
    None
    """
    numbered = [o for o in orders if o.order_number is not None
                and o.trace_center_pixel is not None]
    x = np.array([o.trace_center_pixel for o in numbered], float)
    m = np.array([o.order_number for o in numbered], float)
    identifier = OrderIdentifier(x, m)

    spatial = {
        "trace_x": x,
        "order_number": m,
        "identifier": identifier,
        "profile": None,
        "profile_row": None,
    }
    if white is not None:
        row = white.shape[0] // 2
        spatial["profile"] = np.asarray(white[row, :], float)
        spatial["profile_row"] = int(row)

    payload = {
        "solution": solution,
        "spatial": spatial,
        "atlas_path": atlas_path,
        "orders": [
            {
                "order_number": o.order_number,
                "trace_center_pixel": o.trace_center_pixel,
                "thar_pixels": o.thar_pixels,
                "thar_wavelengths": o.thar_wavelengths,
                "reference_thar_spectrum": o.thar_spectrum,
            }
            for o in numbered
        ],
        "quality": None if report is None else {"checks": report.checks,
                                                "stats": report.stats},
        # How the frames behind this master were prepared. The saved
        # reference arc spectra are what every later night is
        # cross-correlated against, so a master built with the flat field
        # on has to be used with it on: a pixel response is a fixed
        # multiplicative pattern, and correcting one side of that
        # correlation and not the other leaves a mismatch in every shift.
        # load_master checks these against config and says so.
        "processing": {
            "flat_field": bool(getattr(config, "FLAT_FIELD", False)),
            "flat_field_arcs": bool(getattr(config, "FLAT_FIELD_ARCS", True)),
            "apply_bias": bool(getattr(config, "APPLY_BIAS", False)),
            "apply_dark": bool(getattr(config, "APPLY_DARK", False)),
        },
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"save_solution: wrote {len(payload['orders'])} orders to {path}")
    print(f"  spatial map: orders identified by position across the detector "
          f"(x = {x.min():.0f} to {x.max():.0f} px, m = {int(m.max())} to {int(m.min())})"
          + ("" if spatial["profile"] is None
             else f", white-light profile from row {spatial['profile_row']} saved for registration"))


def load_solution(path):
    """Read a master solution written by save_solution.

    The pickle holds classes defined in this module, so this module must be
    importable when it is called.

    Parameters
    ----------
    path : str
        Path to the .pkl file.

    Returns
    -------
    saved : dict
        Keys solution, spatial, atlas_path, orders and quality, as written
        by save_solution.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


# ======================================================================
# interactive helpers
# ======================================================================

def refine_line_at_guess(spectrum, guess, window=15, kind="absorption"):
    """Snap an approximate pixel to the sub-pixel centroid of a line.

    A Gaussian is fitted in a window around the guess. If the fit fails
    or lands outside the window, the local extremum is returned and a
    message is printed.

    Parameters
    ----------
    spectrum : ndarray
        Extracted spectrum of one order, shape (n_pixels,).
    guess : float
        Approximate line position, in pixels.
    window : int, optional
        Half width of the fitting window, in pixels. Default 15.
    kind : str, optional
        "absorption" to seek a minimum, anything else a maximum.
        Default "absorption".

    Returns
    -------
    pixel : float
        Refined line centre, in pixels.
    """
    s = np.asarray(spectrum, float)
    guess = int(round(guess))
    lo = max(0, guess - window)
    hi = min(len(s), guess + window)
    x = np.arange(lo, hi)
    y = s[lo:hi]
    baseline = np.median(y)
    extreme = y.min() if kind == "absorption" else y.max()
    try:
        popt, _ = curve_fit(_gaussian, x, y,
                            p0=(extreme - baseline, guess, max(2.0, window / 4), baseline))
        if lo < popt[1] < hi:
            return float(popt[1])
    except Exception:
        pass
    print(f"  Gaussian refine failed near pixel {guess}; using the local extremum")
    return float(x[np.argmin(y)] if kind == "absorption" else x[np.argmax(y)])


def click_line(spectrum, title, window=15, kind="absorption"):
    """Show a spectrum, take one click, return the refined line centre.

    A click is used rather than an automatic search because in a
    line-rich spectrum only the operator knows which deep line is meant,
    and the seed rests on that. The click need only land inside the
    fitting window.

    Parameters
    ----------
    spectrum : ndarray
        Extracted spectrum of one order, shape (n_pixels,).
    title : str
        Plot title, also used in the printed messages.
    window : int, optional
        Half width of the fitting window, in pixels. Default 15.
    kind : str, optional
        "absorption" to seek a minimum, anything else a maximum.
        Default "absorption".

    Returns
    -------
    pixel : float or None
        Refined line centre in pixels, or None if the window was closed
        without a click.
    """
    spectrum = np.asarray(spectrum, float)
    fig = plt.figure(figsize=(13, 4))
    plt.plot(spectrum, lw=0.7)
    plt.title(f"{title}  (click on the line; close the window to skip)")
    plt.xlabel("pixel")
    plt.tight_layout()
    points = plt.ginput(1, timeout=0)
    plt.close(fig)
    if not points:
        print(f"  no click for {title}")
        return None
    pixel = refine_line_at_guess(spectrum, points[0][0], window=window, kind=kind)
    print(f"  {title}: clicked {points[0][0]:.0f} -> centroid {pixel:.2f}")
    return pixel


def plot_calibrated_orders(orders, spectrum_attr="science_spectrum", title=None,
                           mark_lines=True):
    """Plot every calibrated order on a common wavelength axis.

    Overlapping orders should lie on top of each other, which is the
    visual form of the overlap check. Orders without a wavelength axis,
    without the named spectrum, or with a non-positive peak are skipped.

    Parameters
    ----------
    orders : list
        Traced orders carrying wavelength_poly and the named spectrum.
    spectrum_attr : str, optional
        Attribute holding the spectrum to plot. Default
        "science_spectrum".
    title : str, optional
        Figure title. Default None, meaning a title naming the number of
        orders plotted.
    mark_lines : bool, optional
        Mark the ThAr lines each order was calibrated with. Default True.

    Returns
    -------
    None
    """
    plt.figure(figsize=(15, 6))
    n = 0
    for order in orders:
        if order.wavelength_poly is None:
            continue
        spectrum = getattr(order, spectrum_attr, None)
        if spectrum is None:
            continue
        pix = np.arange(len(spectrum))
        wave = order.wavelength_poly(pix)
        peak = np.nanmax(spectrum)
        if not np.isfinite(peak) or peak <= 0:
            continue
        plt.plot(wave, spectrum / peak, lw=0.6)
        n += 1
        if mark_lines and order.thar_pixels:
            plt.plot(order.wavelength_poly(np.array(order.thar_pixels)),
                     np.full(len(order.thar_pixels), 1.05), "kx", ms=3)
    plt.xlabel("wavelength (Angstrom)")
    plt.ylabel("normalised flux")
    plt.title(title or f"{n} calibrated orders (x = ThAr lines used)")
    plt.tight_layout()