"""
apply_master_thar.py

Reusing a master wavelength solution on any night's data.

The master is a description of the instrument, not of the night it came
from, so nothing here rebuilds it. What happens instead is:

  1. tonight's traces are identified against the master by WHERE they sit on
     the detector, never by counting down the trace list -- an order the
     tracer misses, at either end or in the middle, costs that order and
     nothing else;
  2. the relevant arc is chosen by time and registered against the master's
     own arc, giving a shift along the dispersion;
  3. that shift is applied, and the result is checked against the atlas
     rather than assumed to have worked;
  4. the science frames are extracted and written out on the resulting
     wavelength axes.

Step 3 matters more than it looks. Registering one arc against another only
shows the two look alike; a shift measured against a reference that was
itself wrong reproduces the error exactly. Only the line list can say the
wavelengths are right.
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
    """Find the cross-dispersion shift that lines a new set of trace
    positions up with the master's.

    Done on the positions rather than on the white-light profile, because
    the profile is close to a comb and a comb can lock a whole order out of
    step -- which would misnumber every order by one, silently and
    consistently. The positions cannot do that: order spacing runs from
    about 76 px at the blue end to 30 px at the red, so an alignment that
    is one order out fits at one end and is wildly wrong at the other. Only
    the true shift lines all of them up at once, and it wins by a margin
    that is worth reporting.

    Returns (shift, contrast).
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

    Everything moves together if the instrument has drifted sideways, so
    this is measured once from the whole profile rather than order by
    order, where each order would only be able to say which of its
    neighbours it was nearest to.
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
    """Number tonight's traces from where they sit on the detector.

    This is the step that makes a saved solution survive a different trace
    list. Order numbers are never re-derived by counting from the start of
    the list, so an order missed at the blue end -- or a new one picked up
    there -- shifts nothing: each trace is identified by its own position,
    independently of how many others were found.

    Two stages. A single cross-dispersion shift is measured first, from the
    whole white-light profile if one was saved, so a night where the whole
    spectrum has drifted sideways is registered as a whole rather than
    order by order. Then each trace takes the order number of the saved
    position it lands on, provided it lands within a fraction of the local
    order spacing -- a fraction, not a fixed number of pixels, because the
    spacing runs from about 76 px between the bluest orders to 30 px
    between the reddest, and one tolerance cannot mean the same thing at
    both ends. Traces outside the range the master covered are numbered
    from the fitted position-to-order-number curve instead, and flagged.

    Returns (n_matched, spatial_shift).
    """
    spatial = saved.get("spatial")
    if spatial is None:
        raise ValueError(
            "this master has no spatial map -- it was written by an older version. "
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
                      f"master's range to number -- leaving it out")

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
            print(f"  WARNING: {duplicates} order number(s) assigned twice -- the traces "
                  f"do not line up with the master's. Check the white-light frames.")
        if irregular:
            print(f"  note: {irregular} place(s) where consecutive traces are not "
                  f"consecutive orders. That is expected if the tracer missed a faint "
                  f"order in the middle, and harmless -- the numbering does not depend "
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
    """Measure how far this arc has moved along the dispersion since the
    master was built, by registering each order against its saved reference.

    One shift per order, then a robust median. The scatter between orders is
    the part worth reading: a rigid shift is a statement that the whole
    spectrum translated, and the orders disagreeing with each other is that
    statement failing. A few tenths of a pixel is ordinary flexure. Pixels
    of disagreement, or a trend with order number, means the dispersion
    itself has changed and no single shift can express it.

    Returns (shift, scatter, per-order dict, tilt) where tilt is the slope
    of shift against order number, in pixels per order.
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
                  "spectrum -- rebuild the master rather than trusting a shift.")
    return shift, scatter, dict(zip(numbers.astype(int), shifts)), tilt


def verify_applied_solution(orders, solution, reference, pixel_shift, n_pixels,
                            max_rms_ma=15.0, max_overlap_ms=600.0,
                            detect_kwargs=None, verbose=True):
    """Check a reused solution against the arc it was just applied to.

    Registering one arc against another says only that they look alike. It
    cannot say the wavelengths are right, because a shift measured against
    a reference that was itself wrong reproduces the error exactly. So the
    applied solution is checked the same way it was built: match the new
    arc's lines to the atlas and look at the residuals, and separately ask
    whether adjacent orders still agree where they overlap.

    Returns a QualityReport.
    """
    report = QualityReport()
    report.stats.update({"n_matched": 0, "rms_angstrom": np.nan, "overlap_ms": np.array([])})
    detections = detect_all_orders(orders, **(detect_kwargs or {}))
    shifted = [OrderWavelength(solution, o.order_number, pixel_shift=pixel_shift)
               for o in orders if o.order_number is not None]

    class _Shifted:
        """The saved surface as seen through the measured shift."""
        def __init__(self, sol, shift):
            self.sol, self.shift = sol, shift
            self.n_pixels = sol.n_pixels

        def wavelength(self, pixel, m):
            return self.sol.wavelength(np.asarray(pixel, float) - self.shift, m)

        def dispersion(self, pixel, m):
            return self.sol.dispersion(np.asarray(pixel, float) - self.shift, m)

        def order_axis(self, m, n=None):
            n = self.n_pixels if n is None else n
            return self.wavelength(np.arange(n), np.full(n, float(m)))

    model = _Shifted(solution, pixel_shift)
    matches = match_lines(model, detections, [o.order_number for o in orders],
                          reference, n_pixels, 2.0)

    if len(matches) < 50:
        report.add("atlas check", False,
                   f"only {len(matches)} lines matched -- the shifted solution does not "
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
    """Load the master solution and say what it is.

    The pickle holds classes defined in wavelength_solution.py, so that
    module has to be importable when this runs -- which is why the pipeline
    is run from the code directory. A master written before the spatial map
    existed is refused rather than half-used.
    """
    path = path or config.MASTER_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no master solution at {path}. Build one first with "
            f"make_master_thar.py.")
    saved = ws.load_solution(path)
    if saved.get("spatial") is None:
        raise ValueError(
            f"the master at {path} has no spatial map -- it predates order "
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
    """The atlas lines usable at this instrument's resolution, taken from the
    master's own surface rather than from a seed."""
    solution = saved["solution"]
    m = float(np.median(saved["spatial"]["order_number"]))
    centre = float(solution.m_lambda(n_pixels // 2, m))
    half = float(abs(solution.m_lambda(n_pixels - 1, m) - solution.m_lambda(0, m)) / 2.0)
    return ws.reference_lines_for(config.ATLAS_PATH, centre, half, n_pixels,
                                  config.EXPECTED_LINE_SIGMA_PIXELS,
                                  amplitude_min=config.ATLAS_AMPLITUDE_MIN,
                                  dominance=config.ATLAS_DOMINANCE)


def extract_arc(orders, arc_path):
    """Put an arc's spectra onto this night's traces."""
    image = frames.read_image(arc_path)
    for order in orders:
        order.thar_spectrum = order.extract_thar(image,
                                                 n_sigma=config.ARC_EXTRACT_NSIGMA)


def shift_for_arc(orders, saved, arc_path, reference=None, n_pixels=None,
                  verify=True):
    """Register one arc against the master and return its shift.

    Returns (shift, scatter, report or None).
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
                  "change in the dispersion itself -- if this persists, build a new "
                  "master from this night with make_master_thar.py.")
    return shift, scatter, report


def _interpolated_shift(entries, when):
    """Shift at `when`, from one or two (time, shift) measurements."""
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

    Arrays are (n_orders, n_pixels), ordered blue to red by order number,
    so wavelength[i] and flux[i] belong together.
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
        source_frame=os.path.basename(science_path),
        arc_frames=np.array(arc_names),
        pixel_shift=shift)
    print(f"    -> {os.path.basename(path)} ({len(usable)} orders)")


def reduce_science(white_loc, arc_loc, science_files, out_dir=None,
                   master_path=None, arc_pattern="*ThAr*.fits", verify_first=True):
    """Reduce a night: identify the orders, register the arcs, write spectra.

    science_files : a list of paths, or a single path.

    Each science frame gets the arc or arcs nearest it in time. Where it is
    bracketed, the shift is interpolated to the middle of the exposure;
    where it is not, the nearest arc's shift is used as measured. Each arc
    is registered once however many frames use it.

    Returns a list of (science path, output path, shift).
    """
    if isinstance(science_files, str):
        science_files = [science_files]
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

    # register each arc once, however many frames want it
    measured = {}
    verified = False
    for _, chosen, _ in plans:
        for arc_path, arc_time in chosen:
            if arc_path in measured:
                continue
            shift, scatter, report = shift_for_arc(
                orders, saved, arc_path, reference=reference, n_pixels=n_pixels,
                verify=verify_first and not verified)
            measured[arc_path] = (arc_time, shift)
            if report is not None:
                verified = True

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
        attach_solution(orders, solution, pixel_shift=shift, quiet=True)

        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(out_dir, stem + "_wave.npz")
        save_reduced(out_path, orders, path, [os.path.basename(a) for a, _ in chosen],
                     shift)
        written.append((path, out_path, shift))

    print(f"\nDone: {len(written)} frame(s) written to {out_dir}")
    return written
