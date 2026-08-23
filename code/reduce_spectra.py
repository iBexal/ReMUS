"""Apply a saved master wavelength solution to a new night of data.

The master describes the instrument, so nothing here rebuilds it. Tonight's
traces are identified against the master by their position on the detector,
the arc nearest each science frame is registered against the master's own
arc to give a shift along the dispersion, the shifted solution is checked
against the atlas, and the science frames are extracted and written on the
resulting wavelength axes.

Main entry points: reduce_science, load_master, remove_cosmic_rays.
"""

import os
from datetime import timedelta

import numpy as np
from scipy.ndimage import median_filter

import config
import frames
import wavelength_solution as ws
from wavelength_solution import (C_LIGHT_MS, OrderWavelength, QualityReport,
                                 attach_solution, detect_all_orders, match_lines,
                                 overlap_agreement)
from order_tracing import trace_orders


def align_trace_positions(new_x, saved_x, max_shift=120.0, step=0.25, sigma=2.0):
    """Find the cross-dispersion shift between two sets of trace positions.

    Matching is done on the positions rather than on the white-light
    profile. Order spacing varies across the detector, so an alignment that
    is one order out fits at one end and not at the other, and the contrast
    reports how strongly the best shift beats such a rival.

    Parameters
    ----------
    new_x : ndarray
        Cross-dispersion positions of tonight's traces, in pixels.
    saved_x : ndarray
        Cross-dispersion positions of the master's traces, in pixels.
    max_shift : float, optional
        Largest shift searched, in pixels. Default 120.0.
    step : float, optional
        Spacing of the search grid, in pixels. Default 0.25.
    sigma : float, optional
        Width of the position-matching kernel, in pixels. Default 2.0.

    Returns
    -------
    shift : float
        Amount in pixels to subtract from a new position to land on the
        master's. 0.0 if either set holds fewer than three positions.
    contrast : float
        Score of the best shift divided by that of the best rival at least
        half an order away. 0.0 if no shift could be measured.
    """
    new_x = np.asarray(new_x, float)
    saved_x = np.asarray(saved_x, float)
    if len(new_x) < 3 or len(saved_x) < 3:
        return 0.0, 0.0
    trials = np.arange(-max_shift, max_shift + step, step)
    score = np.empty(len(trials))
    for i, shift in enumerate(trials):
        d = np.abs(saved_x[None, :] - (new_x - shift)[:, None]).min(axis=1)
        score[i] = np.sum(np.exp(-0.5 * (d / sigma) ** 2))
    k = int(np.argmax(score))
    if 0 < k < len(score) - 1:
        denom = score[k - 1] - 2 * score[k] + score[k + 1]
        sub = 0.5 * (score[k - 1] - score[k + 1]) / denom * step if denom != 0 else 0.0
    else:
        sub = 0.0
    shift = float(trials[k] + sub)
    # contrast against the best alignment that is at least half an order away
    far = np.abs(trials - shift) > 0.5 * np.median(np.diff(np.sort(saved_x)))
    rival = score[far].max() if far.any() else 0.0
    return shift, float(score[k] / max(rival, 1e-9))


def measure_spatial_shift(new_profile, saved_profile, max_shift=80):
    """Cross-dispersion shift between two white-light cross-sections.

    Measured once from the whole profile, so a sideways drift of the
    instrument is registered as a whole rather than order by order.

    Parameters
    ----------
    new_profile : ndarray
        Tonight's white-light cross-section, one value per pixel across
        the dispersion.
    saved_profile : ndarray
        The master's white-light cross-section, same convention.
    max_shift : int, optional
        Largest lag searched, in pixels. Default 80.

    Returns
    -------
    shift : float
        Amount in pixels to subtract from a new position to land on the
        master's, matching the sign of align_trace_positions. 0.0 if
        either profile has no variation.
    contrast : float
        Height of the correlation peak above the median, in units of the
        robust scatter of the correlation. 0.0 if no peak was measured.
    """
    a = np.asarray(saved_profile, float)
    b = np.asarray(new_profile, float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    a = a - median_filter(a, 201)
    b = b - median_filter(b, 201)
    if a.std() <= 0 or b.std() <= 0:
        return 0.0, 0.0
    a, b = a / a.std(), b / b.std()
    lags = np.arange(-int(max_shift), int(max_shift) + 1)
    cc = np.array([np.dot(a[max(0, l):n + min(0, l)], b[max(0, -l):n + min(0, -l)])
                   / max(n - abs(l), 1) for l in lags])
    k = int(np.argmax(cc))
    if 0 < k < len(cc) - 1:
        denom = cc[k - 1] - 2 * cc[k] + cc[k + 1]
        sub = 0.5 * (cc[k - 1] - cc[k + 1]) / denom if denom != 0 else 0.0
    else:
        sub = 0.0
    scatter = 1.4826 * np.median(np.abs(cc - np.median(cc)))
    contrast = (cc[k] - np.median(cc)) / scatter if scatter > 0 else 0.0
    # negated so the result means the same as align_trace_positions:
    # the amount to subtract from a new position to land on the master's
    return -float(lags[k] + sub), float(contrast)


def assign_order_numbers_from_saved(orders, saved, white=None, max_spatial_shift=80,
                                    tolerance_fraction=0.35, verbose=True):
    """Number tonight's traces from their position on the detector.

    A single cross-dispersion shift is measured first, then each trace
    takes the order number of the saved position it lands on, provided it
    lands within tolerance_fraction of the local order spacing. The
    tolerance is a fraction because the spacing varies across the
    detector. Traces beyond the range the master traced, but within two
    order spacings of it, are numbered from the fitted
    position-to-order-number curve instead; traces further out are left
    unnumbered.

    Parameters
    ----------
    orders : list of Order
        Tonight's traces. Modified in place: each order_number is set to
        an int or to None.
    saved : dict
        Master solution as returned by load_master, including its
        "spatial" map.
    white : ndarray, optional
        Coadded white-light frame, shape (n_rows, n_columns). Default
        None, meaning the shift is not cross-checked against the saved
        profile.
    max_spatial_shift : int, optional
        Largest cross-dispersion shift searched, in pixels. Default 80.
    tolerance_fraction : float, optional
        Fraction of the local order spacing within which a trace counts as
        matching a saved position. Default 0.35.
    verbose : bool, optional
        Print the shift, the counts and any warnings. Default True.

    Returns
    -------
    n_numbered : int
        Number of orders left holding an order number, matched and
        extrapolated together.
    shift : float
        Cross-dispersion shift used, in pixels.

    Raises
    ------
    ValueError
        If the master carries no spatial map.
    """
    spatial = saved.get("spatial")
    if spatial is None:
        raise ValueError(
            "this master has no spatial map, so it was written by an older version. "
            "Rebuild it (BUILD_NEW_MASTER = True) so orders can be identified by "
            "position instead of by their index in the trace list.")

    identifier = spatial["identifier"]
    saved_x = spatial["trace_x"]
    saved_m = spatial["order_number"]

    new_x = np.array([o.trace_center_pixel for o in orders
                      if o.trace_center_pixel is not None], float)
    shift, contrast = align_trace_positions(new_x, saved_x, max_shift=max_spatial_shift)
    if verbose:
        print(f"assign_order_numbers_from_saved: cross-dispersion shift since the master "
              f"is {shift:+.2f} px, preferred over any one-order-out alignment by "
              f"{contrast:.1f}x")
        if contrast < 1.15:
            print("  WARNING: that margin is thin. If the alignment is out by one order, "
                  "every wavelength will be wrong by a free spectral range while looking "
                  "entirely self-consistent. Check the white-light frames.")

    if white is not None and spatial.get("profile") is not None:
        row = min(spatial.get("profile_row", white.shape[0] // 2), white.shape[0] - 1)
        profile_shift, _ = measure_spatial_shift(white[row, :], spatial["profile"],
                                                 max_shift=max_spatial_shift)
        if verbose:
            agree = abs(profile_shift - shift)
            print(f"  the white-light profile independently gives {profile_shift:+.2f} px "
                  f"({'agrees' if agree < 2.0 else 'DISAGREES'}, {agree:.2f} px apart)")

    # refine on the trace positions themselves
    for _ in range(2):
        residuals = []
        for order in orders:
            if order.trace_center_pixel is None:
                continue
            x = order.trace_center_pixel - shift
            k = int(np.argmin(np.abs(saved_x - x)))
            if abs(saved_x[k] - x) < 0.4 * identifier.spacing_at(x):
                residuals.append(x - saved_x[k])
        if len(residuals) >= 5:
            shift += float(np.median(residuals))

    n_matched, n_extrapolated = 0, 0
    for order in orders:
        if order.trace_center_pixel is None:
            order.order_number = None
            continue
        x = order.trace_center_pixel - shift
        spacing = identifier.spacing_at(x)
        k = int(np.argmin(np.abs(saved_x - x)))
        if abs(saved_x[k] - x) <= tolerance_fraction * spacing:
            order.order_number = int(saved_m[k])
            n_matched += 1
        elif saved_x.min() - 2 * spacing < x < saved_x.max() + 2 * spacing:
            order.order_number = int(round(identifier(x)))
            n_extrapolated += 1
            if verbose:
                print(f"  trace at x={order.trace_center_pixel:.0f} is beyond the orders the "
                      f"master traced; numbered m={order.order_number} by extending the "
                      f"position-to-order curve")
        else:
            order.order_number = None
            if verbose:
                print(f"  trace at x={order.trace_center_pixel:.0f} is too far outside the "
                      f"master's range to number, so it is left out")

    numbered = [o for o in orders if o.order_number is not None]
    duplicates = len(numbered) - len(set(o.order_number for o in numbered))
    steps = np.diff([o.order_number for o in numbered])
    irregular = int(np.sum(np.abs(steps) != 1))

    if verbose:
        print(f"  identified {n_matched} orders by position"
              + (f", {n_extrapolated} by extending the curve" if n_extrapolated else "")
              + f"; {len(orders) - len(numbered)} left unnumbered")
        if numbered:
            print(f"  orders run m={max(o.order_number for o in numbered)} to "
                  f"m={min(o.order_number for o in numbered)} "
                  f"(the master had m={int(saved_m.max())} to {int(saved_m.min())})")
        if duplicates:
            print(f"  WARNING: {duplicates} order number(s) assigned twice, so the traces"
                  f"do not line up with the master's. Check the white-light frames.")
        if irregular:
            print(f"  note: {irregular} place(s) where consecutive traces are not "
                  f"consecutive orders. That is expected if the tracer missed a faint "
                  f"order in the middle, and harmless, since the numbering does not depend"
                  f"on counting.")
    return len(numbered), shift


def _cross_correlate_shift(reference, new, max_shift):
    a = np.asarray(reference, float)
    b = np.asarray(new, float)
    n = min(len(a), len(b))
    if n < 100:
        return None, 0.0
    a = np.nan_to_num(a[:n] - median_filter(a[:n], 201))
    b = np.nan_to_num(b[:n] - median_filter(b[:n], 201))
    if a.std() <= 0 or b.std() <= 0:
        return None, 0.0
    a, b = a / a.std(), b / b.std()
    lags = np.arange(-int(max_shift), int(max_shift) + 1)
    cc = np.array([np.dot(a[max(0, l):n + min(0, l)], b[max(0, -l):n + min(0, -l)])
                   / max(n - abs(l), 1) for l in lags])
    k = int(np.argmax(cc))
    if not (0 < k < len(cc) - 1):
        return None, 0.0
    denom = cc[k - 1] - 2 * cc[k] + cc[k + 1]
    sub = 0.5 * (cc[k - 1] - cc[k + 1]) / denom if denom != 0 else 0.0
    scatter = 1.4826 * np.median(np.abs(cc - np.median(cc)))
    contrast = (cc[k] - np.median(cc)) / scatter if scatter > 0 else 0.0
    # negated so a positive result means the spectrum has moved to higher
    # pixel numbers, which is what OrderWavelength's pixel_shift undoes
    return -float(lags[k] + sub), float(contrast)


def measure_arc_shift(orders, saved, max_shift_pixels=60.0, min_contrast=5.0,
                      verbose=True):
    """Measure an arc's shift along the dispersion since the master.

    Each order is cross-correlated against its saved reference arc, giving
    one shift per order, and the result is their robust median. A scatter
    of a few tenths of a pixel is ordinary flexure; a scatter of pixels, or
    a trend with order number, means the dispersion itself has changed and
    no single shift can express it.

    Parameters
    ----------
    orders : list of Order
        Tonight's traces, numbered, with thar_spectrum extracted.
    saved : dict
        Master solution as returned by load_master.
    max_shift_pixels : float, optional
        Largest shift searched, in pixels. Default 60.0.
    min_contrast : float, optional
        Least correlation peak contrast for an order to be used. Default
        5.0.
    verbose : bool, optional
        Print the shift, scatter, drift and any warning. Default True.

    Returns
    -------
    shift : float
        Median shift along the dispersion, in pixels.
    scatter : float
        Robust scatter of the per-order shifts, in pixels.
    per_order : dict
        Order number (int) to that order's own shift in pixels (float).
    tilt : float
        Slope of shift against order number, in pixels per order. 0.0 if
        eight or fewer orders survive clipping.

    Raises
    ------
    RuntimeError
        If no order could be registered against the master.
    """
    by_number = {e["order_number"]: e for e in saved["orders"]}
    shifts, numbers = [], []
    for order in orders:
        m = order.order_number
        if m is None or m not in by_number or order.thar_spectrum is None:
            continue
        ref = by_number[m]["reference_thar_spectrum"]
        if ref is None:
            continue
        shift, contrast = _cross_correlate_shift(ref, order.thar_spectrum,
                                                 max_shift_pixels)
        if shift is None or contrast < min_contrast:
            continue
        shifts.append(shift)
        numbers.append(m)

    if not shifts:
        raise RuntimeError(
            "no order could be registered against the master. Either the order "
            "numbers are wrong, or this arc has moved further than max_shift_pixels.")

    shifts = np.array(shifts)
    numbers = np.array(numbers, float)
    shift = float(np.median(shifts))
    scatter = float(1.4826 * np.median(np.abs(shifts - shift)))

    keep = np.abs(shifts - shift) < max(4 * scatter, 0.5)
    tilt = 0.0
    if keep.sum() > 8:
        tilt = float(np.polyfit(numbers[keep] - numbers[keep].mean(), shifts[keep], 1)[0])

    if verbose:
        print(f"measure_arc_shift: {shift:+.2f} px along the dispersion, from "
              f"{len(shifts)} orders (scatter {scatter:.2f} px)")
        print(f"  drift across the order range: {tilt * (numbers.max() - numbers.min()):+.2f} px "
              f"end to end")
        if scatter > 1.0:
            print("  WARNING: the orders disagree by more than a pixel, so this is not a "
                  "rigid shift. Something has changed the dispersion, not just moved the "
                  "spectrum. Rebuild the master rather than trusting a shift.")
    return shift, scatter, dict(zip(numbers.astype(int), shifts)), tilt


def verify_applied_solution(orders, solution, reference, pixel_shift, n_pixels,
                            max_rms_ma=15.0, max_overlap_ms=600.0,
                            detect_kwargs=None, verbose=True):
    """Check a reused solution against the arc it was just applied to.

    Registering one arc against another shows only that the two look
    alike; a shift measured against a reference that was itself wrong
    reproduces the error. The shifted solution is therefore checked the
    way it was built: the arc's detected lines are matched to the atlas
    and their residuals measured, and adjacent orders are separately asked
    whether they still agree where they overlap.

    Parameters
    ----------
    orders : list of Order
        Tonight's traces, numbered, with thar_spectrum extracted.
    solution : WavelengthSolution
        The master's m*lambda surface, unshifted.
    reference : ReferenceLines
        Atlas lines usable at this instrument's resolution.
    pixel_shift : float
        Shift along the dispersion to apply to the solution, in pixels.
    n_pixels : int
        Length of one order, in pixels.
    max_rms_ma : float, optional
        Largest passing residual against the atlas, in milliAngstrom.
        Default 15.0.
    max_overlap_ms : float, optional
        Largest passing median disagreement between overlapping orders, in
        m/s. Default 600.0.
    detect_kwargs : dict, optional
        Extra keyword arguments for detect_all_orders. Default None,
        meaning that function's own defaults.
    verbose : bool, optional
        Print the report. Default True.

    Returns
    -------
    report : QualityReport
        Holds the "atlas check" and "order overlap" checks, and the stats
        "n_matched" (int), "rms_angstrom" (float, Angstrom) and
        "overlap_ms" (ndarray, m/s). If fewer than 50 lines match the
        atlas, the atlas check fails, the overlap check is not run, and
        the stats keep their empty defaults.
    """
    report = QualityReport()
    report.stats.update({"n_matched": 0, "rms_angstrom": np.nan, "overlap_ms": np.array([])})
    detections = detect_all_orders(orders, **(detect_kwargs or {}))
    shifted = [OrderWavelength(solution, o.order_number, pixel_shift=pixel_shift)
               for o in orders if o.order_number is not None]

    class _Shifted:
        """The master surface evaluated through a pixel shift.

        Parameters
        ----------
        sol : WavelengthSolution
            The master's m*lambda surface.
        shift : float
            Shift along the dispersion, in pixels.
        """
        def __init__(self, sol, shift):
            self.sol, self.shift = sol, shift
            self.n_pixels = sol.n_pixels

        def wavelength(self, pixel, m):
            """Wavelength in Angstrom, with the shift applied."""
            return self.sol.wavelength(np.asarray(pixel, float) - self.shift, m)

        def dispersion(self, pixel, m):
            """Dispersion in Angstrom per pixel, with the shift applied."""
            return self.sol.dispersion(np.asarray(pixel, float) - self.shift, m)

        def order_axis(self, m, n=None):
            """Full wavelength axis of one order, in Angstrom."""
            n = self.n_pixels if n is None else n
            return self.wavelength(np.arange(n), np.full(n, float(m)))

    model = _Shifted(solution, pixel_shift)
    matches = match_lines(model, detections, [o.order_number for o in orders],
                          reference, n_pixels, 2.0)

    if len(matches) < 50:
        report.add("atlas check", False,
                   f"only {len(matches)} lines matched, so the shifted solution does not"
                   f"land on the atlas at all")
        if verbose:
            report.show()
        return report

    residual = matches.m_lambda - model.wavelength(matches.pixel, matches.m) * matches.m
    scatter = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    keep = np.abs(residual) < 4 * max(scatter, 1e-12)
    rms = float(np.sqrt(np.mean((residual[keep] / matches.m[keep]) ** 2)))
    lam = matches.m_lambda[keep] / matches.m[keep]
    velocity = float(np.sqrt(np.mean(((residual[keep] / matches.m[keep]) / lam
                                      * C_LIGHT_MS) ** 2)))
    report.stats["n_matched"] = int(keep.sum())
    report.stats["rms_angstrom"] = rms
    report.add("atlas check", rms * 1000 <= max_rms_ma,
               f"{keep.sum()} lines of this arc land on the atlas at {rms * 1000:.2f} mA "
               f"= {velocity:.0f} m/s (need <= {max_rms_ma:.0f} mA)")

    velocities, pairs = overlap_agreement(orders, solution, pixel_shift=pixel_shift)
    if len(velocities):
        med = float(np.median(np.abs(velocities)))
        report.stats["overlap_ms"] = velocities
        report.add("order overlap", med <= max_overlap_ms,
                   f"{len(velocities)} overlapping pairs agree to {med:.0f} m/s median "
                   f"(need <= {max_overlap_ms:.0f} m/s)")
    else:
        report.add("order overlap", False, "no overlapping orders to check")

    if verbose:
        report.show()
    return report


# ======================================================================
# choosing an arc, and the whole reduction
# ======================================================================

def load_master(path=None):
    """Load the master solution and print a summary of it.

    The pickle holds classes defined in wavelength_solution.py, so that
    module must be importable when this runs. A master carrying no spatial
    map is refused rather than half used.

    Parameters
    ----------
    path : str, optional
        Path to the master pickle. Default None, meaning
        config.MASTER_PATH.

    Returns
    -------
    saved : dict
        Keys "solution", "spatial", "orders", "atlas_path" and "quality".

    Raises
    ------
    FileNotFoundError
        If no file exists at the given path.
    ValueError
        If the master carries no spatial map.
    """
    path = path or config.MASTER_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no master solution at {path}. Build one first with "
            f"make_master_thar.py.")
    saved = ws.load_solution(path)
    if saved.get("spatial") is None:
        raise ValueError(
            f"the master at {path} has no spatial map, so it predates order"
            f"identification by position. Rebuild it with make_master_thar.py.")
    solution = saved["solution"]
    quality = saved.get("quality") or {}
    stats = quality.get("stats", {})
    print(f"Master solution: {path}")
    print(f"  {len(saved['orders'])} orders, m={int(saved['spatial']['order_number'].max())}"
          f" to {int(saved['spatial']['order_number'].min())}, "
          f"{solution.order_axis(int(saved['spatial']['order_number'].max()))[0]:.0f}"
          f"-{solution.order_axis(int(saved['spatial']['order_number'].min()))[-1]:.0f} A")
    if "rms_angstrom" in stats:
        print(f"  built to {stats['rms_angstrom'] * 1000:.2f} mA on "
              f"{stats.get('n_matched', '?')} lines")
    return saved


def reference_lines_for_master(saved, n_pixels):
    """Select the atlas lines usable at this instrument's resolution.

    The wavelength coverage and dispersion come from the master's own
    surface rather than from a seed.

    Parameters
    ----------
    saved : dict
        Master solution as returned by load_master.
    n_pixels : int
        Length of one order, in pixels.

    Returns
    -------
    reference : ReferenceLines
        Atlas lines strong and isolated enough to calibrate against, with
        wavelengths in Angstrom.
    """
    solution = saved["solution"]
    m = float(np.median(saved["spatial"]["order_number"]))
    centre = float(solution.m_lambda(n_pixels // 2, m))
    half = float(abs(solution.m_lambda(n_pixels - 1, m) - solution.m_lambda(0, m)) / 2.0)
    return ws.reference_lines_for(config.ATLAS_PATH, centre, half, n_pixels,
                                  config.EXPECTED_LINE_SIGMA_PIXELS,
                                  amplitude_min=config.ATLAS_AMPLITUDE_MIN,
                                  dominance=config.ATLAS_DOMINANCE)


def extract_arc(orders, arc_path):
    """Extract one arc frame along this night's traces.

    Parameters
    ----------
    orders : list of Order
        Tonight's traces. Modified in place: each thar_spectrum is set to
        the extracted flux, shape (n_pixels,).
    arc_path : str
        Path to the arc FITS frame.

    Returns
    -------
    None
    """
    image = frames.read_image(arc_path)
    for order in orders:
        order.thar_spectrum = order.extract_thar(image,
                                                 n_sigma=config.ARC_EXTRACT_NSIGMA)


def shift_for_arc(orders, saved, arc_path, reference=None, n_pixels=None,
                  verify=True):
    """Register one arc against the master and return its shift.

    The arc is extracted onto the traces first, so each order's
    thar_spectrum is replaced.

    Parameters
    ----------
    orders : list of Order
        Tonight's traces, numbered. Modified in place.
    saved : dict
        Master solution as returned by load_master.
    arc_path : str
        Path to the arc FITS frame.
    reference : ReferenceLines, optional
        Atlas lines for the verification step. Default None, meaning no
        verification is done.
    n_pixels : int, optional
        Length of one order, in pixels. Default None, meaning the master
        solution's own n_pixels.
    verify : bool, optional
        Check the shifted solution against the atlas. Default True.

    Returns
    -------
    shift : float
        Shift along the dispersion, in pixels.
    scatter : float
        Robust scatter of the per-order shifts, in pixels.
    report : QualityReport or None
        Result of verify_applied_solution, or None if no verification was
        done.
    """
    print(f"\nRegistering {os.path.basename(arc_path)}:")
    extract_arc(orders, arc_path)
    shift, scatter, _, tilt = measure_arc_shift(orders, saved)

    report = None
    if verify and reference is not None:
        report = verify_applied_solution(orders, saved["solution"], reference, shift,
                                         n_pixels or saved["solution"].n_pixels,
                                         **config.APPLY_QUALITY)
        if not report.passed:
            print("This arc does not reproduce the master. A shift cannot express a "
                  "change in the dispersion itself. If this persists, build a new"
                  "master from this night with make_master_thar.py.")
    return shift, scatter, report


# ======================================================================
# cosmic rays
# ======================================================================

def remove_cosmic_rays(spectrum, max_width=2, threshold=8.0, neighbourhood=7):
    """Replace narrow positive spikes with the median of their neighbours.

    Three tests, in order: the reference level is the median of nearby
    pixels with the candidate and everything within max_width of it
    excluded, so a spike cannot prop up its own baseline; the excess over
    that level must exceed threshold times the local noise; and flagged
    pixels are grouped into runs, with every run longer than max_width put
    back, so a real emission line survives untouched. Only positive spikes
    are replaced; pixels reading low are left alone.

    Parameters
    ----------
    spectrum : ndarray
        One order's extracted flux, shape (n_pixels,).
    max_width : int, optional
        Longest run of flagged pixels that is replaced, in pixels. Runs
        longer than this are put back. Default 2.
    threshold : float, optional
        Excess over the reference level required to flag a pixel, in units
        of the local noise. Default 8.0.
    neighbourhood : int, optional
        Half-width of the window the reference level and local noise are
        taken from, in pixels. Default 7.

    Returns
    -------
    cleaned : ndarray
        Copy of the input with flagged pixels set to the reference level,
        shape (n_pixels,). The input array is not modified.
    mask : ndarray of bool
        True where a pixel was replaced, shape (n_pixels,). All False, and
        cleaned an unchanged copy, if the spectrum is shorter than
        2 * neighbourhood + 3 pixels or is not everywhere finite.
    """
    s = np.asarray(spectrum, float)
    n = len(s)
    mask = np.zeros(n, bool)
    if n < 2 * neighbourhood + 3 or not np.all(np.isfinite(s)):
        return s.copy(), mask

    # Neighbours: everything within `neighbourhood` except the candidate and
    # the pixels close enough to belong to the same spike.
    offsets = np.array([d for d in range(-neighbourhood, neighbourhood + 1)
                        if abs(d) > max_width])
    index = np.clip(np.arange(n)[:, None] + offsets[None, :], 0, n - 1)
    neighbours = s[index]

    reference = np.median(neighbours, axis=1)
    local = 1.4826 * np.median(np.abs(neighbours - reference[:, None]), axis=1)

    # Pixel-to-pixel noise for the whole order. Successive differences are
    # blind to the smooth spectrum and see only the noise, and there are
    # far more of them than there are neighbours.
    order_noise = 1.4826 * np.median(np.abs(np.diff(s))) / np.sqrt(2.0)
    sigma = np.maximum(local, max(order_noise, 1e-9))

    candidate = (s - reference) > threshold * sigma
    if candidate.any():
        edges = np.flatnonzero(
            np.diff(np.concatenate(([0], candidate.view(np.int8), [0]))))
        for start, stop in zip(edges[::2], edges[1::2]):
            if stop - start <= max_width:
                mask[start:stop] = True

    cleaned = s.copy()
    cleaned[mask] = reference[mask]
    return cleaned, mask


def clean_orders(orders, attr="science_spectrum", verbose=True, **kwargs):
    """Run the spike removal over every order of one frame.

    Parameters
    ----------
    orders : list of Order
        Tonight's traces. Modified in place: the named spectrum is
        replaced by the cleaned one and cosmic_rays_removed is set to the
        number of pixels replaced in that order, which save_reduced writes
        into the output file.
    attr : str, optional
        Name of the attribute holding the spectrum to clean. Default
        "science_spectrum".
    verbose : bool, optional
        Print the totals and the worst order. Default True.
    **kwargs
        Passed through to remove_cosmic_rays.

    Returns
    -------
    total : int
        Number of pixels replaced across all orders.
    """
    total, worst = 0, (0, None)
    for order in orders:
        spectrum = getattr(order, attr, None)
        if spectrum is None:
            continue
        cleaned, mask = remove_cosmic_rays(spectrum, **kwargs)
        setattr(order, attr, cleaned)
        order.cosmic_rays_removed = int(mask.sum())
        total += order.cosmic_rays_removed
        if order.cosmic_rays_removed > worst[0]:
            worst = (order.cosmic_rays_removed, order.order_number)
    if verbose and total:
        n_orders = sum(1 for o in orders if getattr(o, "cosmic_rays_removed", 0))
        print(f"    narrow spikes: {total} pixel(s) replaced across {n_orders} orders "
              f"(worst m={worst[1]} with {worst[0]})")
        if worst[0] > 20:
            print(f"      m={worst[1]} is well above the rest. In a faint order that "
                  f"usually means detector defects rather than cosmic rays. Compare"
                  f"cosmic_rays_removed across several frames to tell them apart.")
    elif verbose:
        print("    narrow spikes: none found")
    return total


def _interpolated_shift(entries, when):
    """Shift at one time, from one or two (time, shift) measurements.

    Parameters
    ----------
    entries : list of tuple
        (datetime, shift in pixels) pairs in time order. Only the first
        two are used.
    when : datetime
        Time the shift is wanted for, normally mid-exposure.

    Returns
    -------
    shift : float
        Shift along the dispersion, in pixels: the single measured value
        for one entry, linearly interpolated for two, and their mean if
        the two share a timestamp.
    how : str
        One-line description of how the shift was obtained.
    """
    if len(entries) == 1:
        return entries[0][1], f"held at the value measured {entries[0][0]:%H:%M}"
    (t1, s1), (t2, s2) = entries[0], entries[1]
    span = (t2 - t1).total_seconds()
    if span <= 0:
        return 0.5 * (s1 + s2), "averaged (the two arcs share a timestamp)"
    frac = (when - t1).total_seconds() / span
    shift = s1 + frac * (s2 - s1)
    return shift, (f"interpolated {frac:.0%} of the way from {t1:%H:%M} ({s1:+.2f} px) "
                   f"to {t2:%H:%M} ({s2:+.2f} px)")


def save_reduced(path, orders, science_path, arc_names, shift):
    """Write the wavelength-calibrated orders of one science frame.

    Orders without an order number, a wavelength axis or a science
    spectrum are left out. The wavelength (Angstrom) and flux arrays are
    (n_orders, n_pixels), ordered blue to red by order number, so
    wavelength[i] and flux[i] belong together.

    Parameters
    ----------
    path : str
        Destination path for the compressed .npz file.
    orders : list of Order
        Tonight's traces, extracted and carrying a wavelength axis.
    science_path : str
        Path of the science frame; its base name is stored as
        "source_frame".
    arc_names : list of str
        Base names of the arcs used, stored as "arc_frames".
    shift : float
        Shift along the dispersion that was applied, in pixels, stored as
        "pixel_shift".

    Returns
    -------
    None
    """
    usable = [o for o in orders
              if o.order_number is not None and o.wavelength_poly is not None
              and o.science_spectrum is not None]
    usable.sort(key=lambda o: -o.order_number)
    pixels = np.arange(len(usable[0].science_spectrum))
    np.savez_compressed(
        path,
        wavelength=np.array([o.wavelength_poly(pixels) for o in usable]),
        flux=np.array([o.science_spectrum for o in usable]),
        order_number=np.array([o.order_number for o in usable]),
        trace_x=np.array([o.trace_center_pixel for o in usable]),
        pixel=pixels,
        cosmic_rays_removed=np.array([getattr(o, "cosmic_rays_removed", 0)
                                      for o in usable]),
        source_frame=os.path.basename(science_path),
        arc_frames=np.array(arc_names),
        pixel_shift=shift)
    print(f"    -> {os.path.basename(path)} ({len(usable)} orders)")


def reduce_science(white_loc, arc_loc, science_files, out_dir=None,
                   master_path=None, arc_pattern="*ThAr*.fits", verify_arcs=True,
                   clean_cosmic_rays=None):
    """Reduce a night: identify the orders, register the arcs, write files.

    Each science frame takes the arc or arcs nearest it in time. Where it
    is bracketed, the shift is interpolated to mid-exposure; where it is
    not, the nearest arc's shift is used as measured. Each arc is
    registered once however many frames use it, and every arc is verified,
    not just the first. Science spectra have narrow positive spikes
    replaced before they are written; the arcs are left alone.

    Parameters
    ----------
    white_loc : str
        Directory of white-light flats to trace this night's orders from.
    arc_loc : str
        Directory of arc frames.
    science_files : str or list of str
        One science frame path, or a list of them.
    out_dir : str, optional
        Directory for the output files, created if absent. Default None,
        meaning "unsorted" under config.REDUCED_ROOT.
    master_path : str, optional
        Path to the master pickle. Default None, meaning
        config.MASTER_PATH.
    arc_pattern : str, optional
        Glob pattern selecting arc frames. Default "*ThAr*.fits".
    verify_arcs : bool, optional
        Check each arc's shifted solution against the atlas. Default True.
    clean_cosmic_rays : bool, optional
        Replace narrow spikes in the science spectra. Default None,
        meaning config.CLEAN_COSMIC_RAYS.

    Returns
    -------
    written : list of tuple
        One (science path, output path, shift in pixels) per frame
        written.

    Raises
    ------
    FileNotFoundError
        If no arc matches arc_pattern in arc_loc.
    RuntimeError
        If no trace could be matched to the master by position.
    """
    if isinstance(science_files, str):
        science_files = [science_files]
    clean = (config.CLEAN_COSMIC_RAYS if clean_cosmic_rays is None
             else clean_cosmic_rays)
    saved = load_master(master_path)
    solution = saved["solution"]

    print("\nTracing this night's orders ...")
    orders, white = trace_orders(white_loc)
    n_pixels = white.shape[0]

    print()
    n_identified, spatial_shift = assign_order_numbers_from_saved(orders, saved,
                                                                  white=white)
    if n_identified == 0:
        raise RuntimeError("no trace could be matched to the master by position")

    print()
    arcs = frames.list_arcs(arc_loc, pattern=arc_pattern)
    if not arcs:
        raise FileNotFoundError(f"no arcs matching {arc_pattern} in {arc_loc}")
    reference = reference_lines_for_master(saved, n_pixels)

    print("\nMatching arcs to science frames by time:")
    plans = []
    for path in science_files:
        chosen, when = frames.describe_arc_choice(
            path, arcs, interpolate=config.INTERPOLATE_BETWEEN_ARCS,
            max_gap_minutes=config.MAX_ARC_GAP_MINUTES)
        plans.append((path, chosen, when))

    # Register each arc once, however many frames use it. Every arc is
    # checked against the atlas, not just the first: arcs from one night
    # can be hours apart, and an instrument that moved between them moved
    # between the science frames too.
    measured = {}
    failed = []
    for _, chosen, _ in plans:
        for arc_path, arc_time in chosen:
            if arc_path in measured:
                continue
            shift, scatter, report = shift_for_arc(
                orders, saved, arc_path, reference=reference, n_pixels=n_pixels,
                verify=verify_arcs)
            measured[arc_path] = (arc_time, shift)
            if report is not None and not report.passed:
                failed.append(os.path.basename(arc_path))

    if failed:
        print(f"\n{len(failed)} of {len(measured)} arcs did not verify against the "
              f"atlas: {', '.join(failed)}")
        print("Spectra are still written, but treat their wavelengths as unverified "
              "and see whether the master needs rebuilding for this instrument state.")

    out_dir = out_dir or os.path.join(config.REDUCED_ROOT, "unsorted")
    os.makedirs(out_dir, exist_ok=True)

    print("\nExtracting and writing science frames:")
    written = []
    for path, chosen, when in plans:
        entries = sorted((measured[a][0], measured[a][1]) for a, _ in chosen)
        shift, how = _interpolated_shift(entries, when or entries[0][0])
        print(f"  {os.path.basename(path)}: shift {shift:+.2f} px, {how}")

        image = frames.read_image(path)
        for order in orders:
            if order.order_number is None:
                continue
            order.science_spectrum = order.extract_weighted(
                image, n_sigma=config.SCIENCE_EXTRACT_NSIGMA)
        if clean:
            clean_orders(orders, max_width=config.COSMIC_RAY_MAX_WIDTH,
                         threshold=config.COSMIC_RAY_SIGMA)
        attach_solution(orders, solution, pixel_shift=shift, quiet=True)

        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(out_dir, stem + "_wave.npz")
        save_reduced(out_path, orders, path, [os.path.basename(a) for a, _ in chosen],
                     shift)
        written.append((path, out_path, shift))

    print(f"\nDone: {len(written)} frame(s) written to {out_dir}")
    return written
