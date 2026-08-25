"""Locate echelle orders on a flat, fit their traces, and extract spectra.

Images are 2D of shape (ny, nx). Rows, the y axis, are the dispersion
direction; columns, the x axis, are the cross dispersion direction.
Wavelengths are handled in wavelength_solution.py.

Main entry points: trace_orders, trace_single_order, choose_trace_window,
find_spurious_peaks, and the Order class.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

import config
import frames


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

        # Flat field products, filled in by flat_field.py. blaze is the
        # smooth part of the white light spectrum, carrying the lamp's own
        # colour and the grating's blaze together; pixel_response is what
        # is left of it pixel to pixel, and is the only part divided out.
        self.blaze = None
        self.pixel_response = None       # measured through the science aperture
        self.pixel_response_arc = None   # and through the narrower arc aperture

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

    def _aperture_grid(self, ny, nx, n_sigma):
        """Column indices and validity mask of the aperture at every row.

        The aperture bounds are the ones aperture() gives, evaluated for
        all rows at once. Rows are padded out to the widest aperture and
        the padding is masked off, which lets the extractions run as array
        operations rather than a Python loop over 4096 rows.

        Parameters
        ----------
        ny : int
            Number of rows in the image.
        nx : int
            Number of columns in the image.
        n_sigma : float
            Half width of the aperture in units of the profile sigma.

        Returns
        -------
        columns : ndarray of int
            Column index of each aperture slot, shape (ny, width), clipped
            into the image so it is always safe to index with.
        valid : ndarray of bool
            True where that slot is really inside both the aperture and
            the image, shape (ny, width).
        centers : ndarray
            Trace center per row, in pixels, shape (ny,).
        sigmas : ndarray
            Profile sigma per row, in pixels, shape (ny,).
        """
        y = np.arange(ny)
        centers = np.asarray(self.center(y), float)
        sigmas = np.asarray(self.sigma(y), float)

        # The loop this replaced went through int(), which raises on a
        # non-finite bound. Vectorised, np.floor(nan).astype(int) quietly
        # becomes the most negative integer there is, and the aperture
        # then lands nowhere and sums zero. A trace whose polynomial has
        # gone non-finite is a broken trace, and it should say so rather
        # than write a column of zeros into a spectrum.
        edges = np.concatenate([centers - n_sigma * sigmas,
                                centers + n_sigma * sigmas])
        if not np.isfinite(edges).all():
            raise ValueError(
                "this order's trace or width polynomial is not finite over the "
                "detector, so its extraction aperture is undefined. The trace fit "
                "failed; check it with one_off/check_tracing.py rather than "
                "extracting along it.")

        lo = np.floor(centers - n_sigma * sigmas).astype(int)
        hi = np.ceil(centers + n_sigma * sigmas).astype(int)      # exclusive

        # Only ever as wide as the detector. A runaway width polynomial
        # would otherwise ask for a (n_rows, huge) array before anything
        # got the chance to clip it.
        width = int(np.clip((hi - lo).max(), 1, nx))

        columns = lo[:, None] + np.arange(width)[None, :]
        valid = (columns < hi[:, None]) & (columns >= 0) & (columns < nx)
        return np.clip(columns, 0, nx - 1), valid, centers, sigmas

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
        columns, valid, _, _ = self._aperture_grid(ny, nx, n_sigma)
        rows = np.arange(ny)[:, None]
        out = np.where(valid, image[rows, columns], 0.0).sum(axis=1)
        return np.asarray(out, dtype=float)

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
        columns, valid, centers, sigmas = self._aperture_grid(ny, nx, n_sigma)
        rows = np.arange(ny)[:, None]

        # A row of zero width has an empty aperture, so `valid` already
        # rules it out; substituting here only keeps the division from
        # warning about a row whose weights are discarded anyway.
        safe_sigma = np.where(sigmas > 0, sigmas, 1.0)[:, None]
        weights = np.exp(-(columns - centers[:, None]) ** 2 / (2.0 * safe_sigma ** 2))
        weights = np.where(valid, weights, 0.0)
        total = weights.sum(axis=1)

        usable = (valid.sum(axis=1) >= 2) & (total > 0)
        safe = np.where(usable, total, 1.0)
        out = (np.where(valid, image[rows, columns], 0.0) * weights).sum(axis=1) / safe
        return np.where(usable, out, np.nan)

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


def _fit_row(image, row, guess, window):
    """Fit a Gaussian to one row of an image near a guessed column.

    Parameters
    ----------
    image : ndarray
        2D image of shape (ny, nx).
    row : int
        Row index to fit.
    guess : float
        Column the fit window is centred on, in pixels.
    window : float
        Half width of the fit window, in pixels.

    Returns
    -------
    popt : tuple or None
        (amplitude, center, sigma, offset) from the fit, or None if the
        window holds fewer than 5 columns or the fit did not converge.
    """
    nx = image.shape[1]
    xmin = max(0, int(np.floor(guess - window)))
    xmax = min(nx, int(np.ceil(guess + window + 1)))
    if xmax - xmin < 5:
        return None
    x = np.arange(xmin, xmax)
    profile = image[row, xmin:xmax]
    try:
        p0 = (profile.max() - np.median(profile), guess, 2.0, np.median(profile))
        popt, _ = curve_fit(gaussian, x, profile, p0=p0)
        return popt
    except Exception:
        return None


def trace_single_order(image, start_x, window=8, step=20, predictive=True,
                       max_jump=4.0, sigma_limits=(0.6, 6.0),
                       min_amplitude_sigma=3.0):
    """Trace one order by fitting a Gaussian to rows across the image.

    Fitting starts at the middle row with start_x as the first guess, then
    walks outwards in both directions every step rows. With predictive set,
    each row is fitted around the previous center extrapolated by the local
    slope, measured from the last two accepted rows, so the tracer follows
    a steep order as long as its slope changes slowly; a fit is kept only
    if it lands within max_jump of that prediction, its width lies inside
    sigma_limits and its amplitude stands above the scatter of the fit
    residual. A rejected row leaves NaN and the walk continues from the
    prediction, so one bad row does not drag the rest of the trace with it.
    With predictive cleared, each row is fitted around the previous center
    with no extrapolation and every converged fit is kept, which is the
    older and weaker behaviour. Rows that are not sampled or whose fit is
    rejected stay NaN. All three arrays are entirely NaN if the middle row
    fit fails.

    Parameters
    ----------
    image : ndarray
        2D image of shape (ny, nx), rows along dispersion.
    start_x : float
        Initial guess for the trace position at the middle row, in pixels.
    window : float, optional
        Half width of the column range fitted around each guess, in
        pixels. Default 8. Rows offering fewer than 5 columns are skipped.
    step : int, optional
        Row spacing between fits, in pixels. Default 20.
    predictive : bool, optional
        Extrapolate the local slope and validate each fit. Default True.
    max_jump : float, optional
        Largest accepted distance between a fitted center and the
        predicted one, in pixels. Default 4.0. Ignored unless predictive.
    sigma_limits : tuple of float, optional
        Smallest and largest accepted fitted profile width, in pixels.
        Default (0.6, 6.0). Ignored unless predictive.
    min_amplitude_sigma : float, optional
        Amplitude required, as a multiple of the robust scatter of the fit
        residual. Default 3.0. Ignored unless predictive.

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

    def accept(popt, prediction, row):
        """True if this fit can be the same order continuing."""
        if popt is None:
            return False
        if not predictive:
            return True
        amplitude, center, sigma, _ = popt
        if not np.isfinite([amplitude, center, sigma]).all():
            return False
        if abs(center - prediction) > max_jump:
            return False
        if not sigma_limits[0] <= abs(sigma) <= sigma_limits[1]:
            return False
        # significance against the scatter of the fit residual, which is a
        # real noise estimate; the scatter of the raw pixels in the window
        # is not, since the order itself dominates it
        xmin = max(0, int(np.floor(prediction - window)))
        xmax = min(nx, int(np.ceil(prediction + window + 1)))
        x = np.arange(xmin, xmax)
        scatter = 1.4826 * np.median(np.abs(image[row, xmin:xmax] - gaussian(x, *popt)))
        return amplitude > min_amplitude_sigma * max(scatter, 1e-9)

    popt = _fit_row(image, mid_y, start_x, window)
    if not accept(popt, start_x, mid_y):
        return centers, sigmas, amplitudes
    centers[mid_y], sigmas[mid_y], amplitudes[mid_y] = popt[1], abs(popt[2]), popt[0]

    for direction in (-step, step):
        last_row, last_x = mid_y, centers[mid_y]
        slope = 0.0
        rows = range(mid_y + direction, -1 if direction < 0 else ny, direction)
        for row in rows:
            prediction = last_x + slope * (row - last_row) if predictive else last_x
            popt = _fit_row(image, row, prediction, window)
            if not accept(popt, prediction, row):
                continue
            center = popt[1]
            if predictive and row != last_row:
                measured = (center - last_x) / (row - last_row)
                # ease the slope in so one noisy row cannot swing the walk
                slope = 0.5 * slope + 0.5 * measured
            centers[row], sigmas[row], amplitudes[row] = center, abs(popt[2]), popt[0]
            last_row, last_x = row, center

    return centers, sigmas, amplitudes


def choose_trace_window(peaks, window=8.0, verbose=True):
    """Shrink the fit window if the orders sit closer together than it.

    A window wider than half the order separation reaches into the
    neighbouring order, and the fit will eventually prefer that neighbour.

    Parameters
    ----------
    peaks : ndarray
        Order positions at the middle row, in pixels.
    window : float, optional
        Requested half width of the fit window, in pixels. Default 8.0.
    verbose : bool, optional
        Print a line when the window is reduced. Default True.

    Returns
    -------
    window : float
        The requested window, or half the smallest order separation minus
        one pixel where that is smaller, with a floor of 3 pixels.
    """
    if len(peaks) < 2:
        return window
    closest = float(np.min(np.diff(np.sort(peaks))))
    safe = max(3.0, closest / 2.0 - 1.0)
    if safe < window:
        if verbose:
            print(f"Orders come within {closest:.0f} px, so a +/-{window:g} px fit "
                  f"window overlaps the neighbour; using +/-{safe:.1f} px")
        return safe
    return window


def trace_orders(white_loc, diagnose=False, n_expected=None, auto_exclude=True,
                 exclude_rel_threshold=0.3, extra_exclude_indices=None,
                 prominence_fraction=0.005, min_separation=15,
                 window=8.0, step=20, auto_window=True, trace_degree=3,
                 pattern=None):
    """Find and trace every order visible in a coadded white light flat.

    All FITS files in white_loc are median coadded, peaks are located in
    the middle row of the coadd, and each surviving peak seeds one trace.
    Progress, exclusions and skipped orders are printed. With diagnose set,
    a matplotlib figure of the middle row profile is created but not shown.

    Parameters
    ----------
    white_loc : str
        Directory searched for white light flat frames, matching pattern.
        Frames are read through frames.read_image, so config.TRANSPOSE
        applies.
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
    window : float, optional
        Half width of the column range fitted around each guess, in
        pixels. Default 8.0.
    step : int, optional
        Row spacing between fits, in pixels. Default 20.
    auto_window : bool, optional
        Reduce window where the orders sit closer together than twice it,
        via choose_trace_window. Default True.
    trace_degree : int, optional
        Degree of the polynomials fitted to the measured trace centers and
        widths. Default 3. Raise it for strongly curved orders, having
        checked the residual with one_off/check_tracing.py.
    pattern : str, optional
        Filename glob selecting the flats. Default None, meaning
        config.FRAME_PATTERN.

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
        If white_loc contains no matching frames.
    """
    white_files = frames.list_frames(white_loc, pattern)
    if not white_files:
        raise FileNotFoundError(
            f"no frames matching {pattern or config.FRAME_PATTERN} in {white_loc}")
    # float32 for the stack. Ten 4096 x 4096 frames is 1.3 GB in double and
    # half that in single, and a median of ADU counts does not need the
    # other seven digits.
    stack = np.empty((len(white_files),) + frames.read_image(white_files[0]).shape,
                     dtype=np.float32)
    for i, path in enumerate(white_files):
        stack[i] = frames.read_image(path)
    coadded = np.median(stack, axis=0).astype(float)
    del stack

    saturated = np.mean(coadded >= config.FLAT_SATURATION)
    if saturated > 0.0005:
        print(f"WARNING: {100 * saturated:.2f}% of the coadded flat is at or above "
              f"{config.FLAT_SATURATION:g} ADU. Saturated flats flatten the order "
              f"profile, which widens the traces and corrupts any flat field taken "
              f"from them. Shorten the white light exposures.")

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

    if auto_window:
        window = choose_trace_window(peaks, window)

    orders = []
    for peak in peaks:
        centers, sigmas, _ = trace_single_order(coadded, start_x=peak,
                                                window=window, step=step)
        good = np.isfinite(centers)
        if np.sum(good) < trace_degree + 2:
            print(f"Skipping order at x={peak}: only {np.sum(good)} traced points")
            continue
        rows = np.where(good)[0]
        order = Order(
            center_poly=np.poly1d(np.polyfit(rows, centers[good], trace_degree)),
            sigma_poly=np.poly1d(np.polyfit(rows, sigmas[good], trace_degree)))
        order.trace_center_pixel = float(peak)
        orders.append(order)

    print(f"Traced {len(orders)} orders.")
    return orders, coadded