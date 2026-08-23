"""Build the master white and check the order tracing against it.

Standalone diagnostic, not part of the pipeline. Run it on a new
spectrograph before anything else, to see whether the tracer follows the
orders and whether the trace polynomial is flexible enough for them.

It coadds the white-light flats, finds the orders, traces each one keeping
the raw per-row measurements, and produces:

  1. the master white with every trace drawn on top, plus a zoom panel at
     full pixel scale;
  2. the measured trace slope against the limit the tracer can follow;
  3. the residual of the trace polynomial, at several degrees.

The two numbers that decide whether tracing works are printed as a verdict.
The first is the slope: trace_single_order walks outwards in steps of
TRACE_STEP rows and fits inside a window of plus or minus TRACE_WINDOW
pixels around the previous centre, so it loses the order once the centre
moves more than TRACE_WINDOW pixels in TRACE_STEP rows. The second is the
polynomial residual: order_tracing fits a cubic to the measured centres, so
an order whose shape a cubic cannot follow is traced but then described
badly.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob

import fitsio
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

import config
from order_tracing import (choose_trace_window, find_spurious_peaks,
                           trace_single_order)

# ----------------------------------------------------------------------
# What to look at
# ----------------------------------------------------------------------
NIGHT = os.path.join(config.SPECTRA_ROOT, "2025-03-12")
WHITE_LOC = os.path.join(NIGHT, "White")

OUT_DIR = os.path.join(config.PROJECT_ROOT, "reduced", "tracing_check")
SAVE_MASTER_WHITE = True          # write the coadd as a FITS file
SHOW = True                       # open the figures as well as saving them

# ----------------------------------------------------------------------
# Tracing settings, matching order_tracing.trace_orders
# ----------------------------------------------------------------------
# Change these here to test a setting before editing order_tracing.py.
TRACE_WINDOW = 8                  # half width of the fit window, pixels
TRACE_STEP = 20                   # rows between fits
PROMINENCE_FRACTION = 0.005       # peak prominence, as a fraction of the max
MIN_SEPARATION = 15               # minimum peak separation, pixels
AUTO_EXCLUDE = True               # drop peaks far fainter than their neighbours

# A trace whose cubic residual is worse than this did not follow one order.
LOST_RESIDUAL_PIXELS = 1.0

# ----------------------------------------------------------------------
# The predictive tracer
# ----------------------------------------------------------------------
# order_tracing.trace_single_order takes a predictive flag. Set it and each
# row is fitted around the previous centre extrapolated by the local slope,
# and a fit that cannot be the same order continuing is refused, so the
# limit becomes how fast the slope changes rather than how steep it is.
# Clear it to reproduce the older behaviour, where the guess was the
# previous centre and every converged fit was kept.
PREDICTIVE = True
COMPARE = True                    # trace both ways and print the difference
MAX_JUMP_PIXELS = 4.0             # reject a centre this far from the prediction
SIGMA_LIMITS = (0.6, 6.0)         # reject a fitted profile width outside this
MIN_AMPLITUDE_SIGMA = 3.0         # reject a peak this weak against row scatter
AUTO_WINDOW = True                # shrink the window if the orders are packed

# Degrees to try when fitting the measured centres. order_tracing uses 3.
POLY_DEGREES = (3, 4, 5)

# Where to put the zoom panel, as a fraction of the image. The default looks
# at the top left corner, where curvature is usually worst.
ZOOM_CENTRE = (0.12, 0.88)        # (x, y) fraction
ZOOM_SIZE = 400                   # pixels across


def build_master_white(white_loc):
    """Median coadd of every FITS frame in a directory.

    Parameters
    ----------
    white_loc : str
        Directory holding the white-light flats.

    Returns
    -------
    coadd : ndarray
        Median of the frames, shape (ny, nx).
    n_frames : int
        Number of frames combined.

    Raises
    ------
    FileNotFoundError
        If the directory holds no FITS files.
    """
    paths = sorted(glob.glob(os.path.join(white_loc, "*.fits")))
    if not paths:
        raise FileNotFoundError(f"no FITS files in {white_loc}")
    frames = []
    for path in paths:
        with fitsio.FITS(path) as f:
            frames.append(f[0].read())
    return np.median(frames, axis=0), len(paths)


def find_order_peaks(coadd, prominence_fraction=PROMINENCE_FRACTION,
                     min_separation=MIN_SEPARATION, auto_exclude=AUTO_EXCLUDE):
    """Order positions at the middle row, as trace_orders finds them.

    Parameters
    ----------
    coadd : ndarray
        Master white, shape (ny, nx).
    prominence_fraction : float, optional
        Peak prominence required, as a fraction of the profile maximum.
    min_separation : int, optional
        Minimum separation between peaks, in pixels.
    auto_exclude : bool, optional
        Drop peaks far fainter than their neighbours. Default True.

    Returns
    -------
    peaks : ndarray
        Kept peak positions along x, in pixels.
    excluded : ndarray
        Positions dropped by the faint-peak test, in pixels.
    profile : ndarray
        The middle row of the coadd, shape (nx,).
    """
    profile = coadd[coadd.shape[0] // 2, :]
    peaks, _ = find_peaks(profile, prominence=np.max(profile) * prominence_fraction,
                          distance=min_separation)
    excluded = np.array([], dtype=int)
    if auto_exclude:
        flagged = find_spurious_peaks(profile[peaks])
        if flagged:
            excluded = peaks[flagged]
            keep = np.ones(len(peaks), bool)
            keep[flagged] = False
            peaks = peaks[keep]
    return peaks, excluded, profile


def trace_all(coadd, peaks, window=TRACE_WINDOW, step=TRACE_STEP,
              predictive=PREDICTIVE):
    """Trace every order, keeping the raw per-row measurements.

    trace_orders throws these away and returns only the fitted polynomial.
    They are what the diagnostics need.

    Parameters
    ----------
    coadd : ndarray
        Master white, shape (ny, nx).
    peaks : ndarray
        Starting positions along x, in pixels.
    window : int, optional
        Half width of the fit window, in pixels.
    step : int, optional
        Rows between fits.
    predictive : bool, optional
        Use trace_single_order_predictive rather than the pipeline tracer.

    Returns
    -------
    centers : ndarray
        Measured trace position per row, shape (n_orders, ny), NaN where no
        fit was made or the fit failed.
    sigmas : ndarray
        Measured profile sigma per row, same shape, in pixels.
    """
    ny = coadd.shape[0]
    centers = np.full((len(peaks), ny), np.nan)
    sigmas = np.full((len(peaks), ny), np.nan)
    for i, peak in enumerate(peaks):
        c, s, _ = trace_single_order(coadd, start_x=peak, window=window,
                                     step=step, predictive=predictive,
                                     max_jump=MAX_JUMP_PIXELS,
                                     sigma_limits=SIGMA_LIMITS,
                                     min_amplitude_sigma=MIN_AMPLITUDE_SIGMA)
        centers[i] = c
        sigmas[i] = s
    return centers, sigmas


def diagnose(centers, step=TRACE_STEP, window=TRACE_WINDOW, degrees=POLY_DEGREES):
    """Measure how hard the orders are to trace and to fit.

    Parameters
    ----------
    centers : ndarray
        Measured trace positions, shape (n_orders, ny), NaN where missing.
    step : int, optional
        Rows between fits, used to convert slope into pixels per step.
    window : int, optional
        Half width of the fit window, in pixels.
    degrees : tuple of int, optional
        Polynomial degrees to fit to the measured centres.

    Returns
    -------
    stats : list of dict
        One entry per order, with keys n_fitted, n_attempted, drift, slope,
        peak_slope, px_per_step, residual and lost. slope is the 95th
        percentile of |dx/dy| and is the number the verdict uses;
        peak_slope is the single steepest row-to-row step, which a trace
        that jumps to its neighbour drives to a meaningless value. residual
        maps each degree to the RMS of the fit in pixels. lost is True for a
        trace the tracer did not follow, judged by a peak slope beyond what
        the window allows or a cubic residual above LOST_RESIDUAL_PIXELS.
    """
    n_orders, ny = centers.shape
    rows = np.arange(ny)
    limit = window / step
    stats = []
    for i in range(n_orders):
        good = np.isfinite(centers[i])
        y, x = rows[good], centers[i][good]
        entry = {
            "n_fitted": int(good.sum()),
            "n_attempted": int(np.ceil(ny / step)),
            "drift": float(x.max() - x.min()) if good.sum() > 1 else np.nan,
            "slope": np.nan,
            "peak_slope": np.nan,
            "px_per_step": np.nan,
            "residual": {},
            "curvature": np.nan,
            "prediction_error": np.nan,
            "lost": True,
        }
        if good.sum() > 3:
            slope = np.abs(np.diff(x) / np.diff(y))
            entry["slope"] = float(np.nanpercentile(slope, 95))
            entry["peak_slope"] = float(np.nanmax(slope))
            entry["px_per_step"] = entry["slope"] * step
            for degree in degrees:
                if good.sum() > degree + 2:
                    fit = np.poly1d(np.polyfit(y, x, degree))
                    entry["residual"][degree] = float(np.sqrt(np.mean((x - fit(y)) ** 2)))
            if good.sum() > 6:
                cubic_fit = np.polyfit(y, x, 3)
                curvature = np.abs(np.polyval(np.polyder(cubic_fit, 2), y))
                entry["curvature"] = float(curvature.max())
                # how far the straight-line prediction misses over one step
                entry["prediction_error"] = 0.5 * entry["curvature"] * step ** 2
            cubic = entry["residual"].get(3, np.nan)
            entry["lost"] = bool(entry["peak_slope"] > limit
                                 or (np.isfinite(cubic)
                                     and cubic > LOST_RESIDUAL_PIXELS))
        stats.append(entry)
    return stats


def report(stats, peaks, window=TRACE_WINDOW, step=TRACE_STEP, degrees=POLY_DEGREES,
           max_jump=MAX_JUMP_PIXELS):
    """Print the per-order table and the two verdicts.

    Parameters
    ----------
    stats : list of dict
        Output of diagnose.
    peaks : ndarray
        Order positions at the middle row, in pixels.
    window : int, optional
        Half width of the fit window, in pixels.
    step : int, optional
        Rows between fits.
    degrees : tuple of int, optional
        Degrees present in the residual entries.
    max_jump : float, optional
        Largest prediction miss the predictive tracer accepts, in pixels.
    """
    limit = window / step
    print(f"\n{len(peaks)} orders found at x = {peaks.min()} to {peaks.max()}")
    if len(peaks) > 1:
        gaps = np.diff(peaks)
        print(f"order separation {gaps.min()} to {gaps.max()} px")

    header = f"{'order':>5s} {'x@mid':>6s} {'rows':>9s} {'drift':>7s} {'px/row':>7s} {'px/step':>8s}"
    for degree in degrees:
        header += f" {'res d' + str(degree):>7s}"
    header += "  flag"
    print("\n" + header)
    for i, (entry, peak) in enumerate(zip(stats, peaks)):
        line = (f"{i:5d} {peak:6d} {entry['n_fitted']:4d}/{entry['n_attempted']:<4d} "
                f"{entry['drift']:7.1f} {entry['slope']:7.3f} "
                f"{entry['px_per_step']:8.2f}")
        for degree in degrees:
            value = entry["residual"].get(degree)
            line += f" {value:7.3f}" if value is not None else f" {'-':>7s}"
        line += "  LOST" if entry["lost"] else ""
        print(line)

    lost = [i for i, e in enumerate(stats) if e["lost"]]
    kept = [i for i, e in enumerate(stats) if not e["lost"]]
    if not kept:
        print("\nEvery trace was lost. Nothing below can be measured.")
        return
    slopes = np.array([stats[i]["slope"] for i in kept], dtype=float)
    worst = np.nanmax(slopes)
    fitted = np.array([stats[i]["n_fitted"] for i in kept], dtype=float)
    attempted = np.array([stats[i]["n_attempted"] for i in kept], dtype=float)
    completeness = np.nanmedian(fitted / attempted)

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  {len(kept)} of {len(stats)} traces followed a single order.")
    if lost:
        print(f"  [WARN] {len(lost)} trace(s) were lost: "
              + ", ".join(str(i) for i in lost))
        print("         A lost trace has jumped onto its neighbour or run off the")
        print("         detector. Check those on the image panel before trusting")
        print("         them. They are excluded from the numbers below.")
    print()
    print(f"  fit window +/-{window} px every {step} rows, so the tracer can follow")
    print(f"  a slope of up to {limit:.3f} px/row before the order leaves the window.")
    print(f"  Steepest slope measured: {worst:.3f} px/row, "
          f"{100 * worst / limit:.0f}% of the limit.")
    if worst > limit:
        print("  [FAIL] Orders are steeper than the tracer can follow. Reduce "
              "TRACE_STEP or raise TRACE_WINDOW, or both.")
        print(f"         step {max(1, int(window / worst))} would be inside the limit "
              f"at the current window.")
    elif worst > 0.6 * limit:
        print("  [WARN] Little headroom. A slightly steeper order, or one poorly "
              "seeded at the middle row, will be lost.")
        print(f"         Halving TRACE_STEP to {step // 2} doubles the headroom.")
    else:
        print("  [PASS] Comfortable headroom.")

    if PREDICTIVE:
        errors = np.array([stats[i]["prediction_error"] for i in kept], dtype=float)
        worst_error = np.nanmax(errors)
        print()
        print("  The predictive tracer does not care about slope, only about how")
        print("  fast the slope changes, since it extrapolates between rows. Over")
        print(f"  one {step}-row step the straight-line prediction misses by at most")
        print(f"  {worst_error:.3f} px here, against the {max_jump:g} px it will accept.")
        if worst_error > max_jump:
            print("  [FAIL] Curvature outruns the prediction. Reduce TRACE_STEP, "
                  "which helps as its square, or raise MAX_JUMP_PIXELS.")
        elif worst_error > 0.6 * max_jump:
            print("  [WARN] Little headroom on curvature. Halving TRACE_STEP cuts "
                  "the miss by four.")
        else:
            print("  [PASS] Comfortable headroom on curvature.")

    print(f"\n  Rows successfully fitted: {100 * completeness:.0f}% of those attempted "
          f"in the median order.")
    if completeness < 0.8:
        print("  [WARN] Many rows failed. A failed fit does not update the guess, "
              "so failures compound along a curved order.")
    else:
        print("  [PASS] Most rows fitted.")

    print()
    for degree in degrees:
        values = np.array([stats[i]["residual"].get(degree, np.nan) for i in kept],
                          dtype=float)
        if np.all(np.isnan(values)):
            continue
        tag = "  <- used by order_tracing" if degree == 3 else ""
        print(f"  degree {degree}: trace polynomial residual "
              f"median {np.nanmedian(values):.3f} px, worst {np.nanmax(values):.3f} px"
              f"{tag}")
    best_third = np.array([stats[i]["residual"].get(3, np.nan) for i in kept],
                          dtype=float)
    if np.nanmedian(best_third) > 0.3:
        print("  [FAIL] A cubic does not describe these orders. Raise the degree in "
              "order_tracing.trace_orders, where center_poly is fitted.")
    elif np.nanmedian(best_third) > 0.1:
        print("  [WARN] The cubic is marginal. Compare the higher degrees above.")
    else:
        print("  [PASS] A cubic describes the traces to well under a tenth of a pixel.")
    print("=" * 72)


def plot_traces(coadd, centers, peaks, excluded, stats, out_path=None,
                degrees=POLY_DEGREES, zoom_centre=ZOOM_CENTRE, zoom_size=ZOOM_SIZE,
                window=TRACE_WINDOW, step=TRACE_STEP):
    """Draw the master white with the traces on it, plus the diagnostics.

    Parameters
    ----------
    coadd : ndarray
        Master white, shape (ny, nx).
    centers : ndarray
        Measured trace positions, shape (n_orders, ny).
    peaks : ndarray
        Order positions at the middle row, in pixels.
    excluded : ndarray
        Peak positions dropped as spurious, in pixels.
    stats : list of dict
        Output of diagnose.
    out_path : str, optional
        Where to save the figure. Default None, meaning it is not saved.
    degrees : tuple of int, optional
        Degrees present in the residual entries.
    zoom_centre : tuple of float, optional
        Centre of the zoom panel, as (x, y) fractions of the image.
    zoom_size : int, optional
        Width of the zoom panel, in pixels.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure, so the caller can show or further edit it.
    """
    ny, nx = coadd.shape
    rows = np.arange(ny)
    lo, hi = np.percentile(coadd, [5, 99.5])
    ramp = plt.get_cmap("viridis")
    ink, soft, faint = "#0b0b0b", "#52514e", "#8a8880"
    accent, surface = "#eb6834", "#fcfcfb"

    fig = plt.figure(figsize=(16, 9.0), facecolor=surface)
    gs = fig.add_gridspec(2, 3, width_ratios=[2.1, 1, 1], height_ratios=[1, 1],
                          hspace=0.28, wspace=0.24, left=0.055, right=0.985,
                          top=0.90, bottom=0.07)

    def style(ax, xl=None, yl=None, title=None):
        ax.set_facecolor(surface)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d8d6cf")
        ax.tick_params(colors=faint, labelsize=8)
        if xl:
            ax.set_xlabel(xl, fontsize=8.5, color=soft)
        if yl:
            ax.set_ylabel(yl, fontsize=8.5, color=soft)
        if title:
            ax.set_title(title, fontsize=10, color=ink, loc="left", pad=7)

    # the whole detector
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(coadd, origin="lower", cmap="gray", vmin=lo, vmax=hi,
              aspect="equal", interpolation="nearest")
    n_lost = sum(1 for e in stats if e["lost"])
    for i in range(len(centers)):
        good = np.isfinite(centers[i])
        if good.sum() > 1:
            if stats[i]["lost"]:
                ax.plot(centers[i][good], rows[good], lw=1.0, color=accent,
                        ls="--", alpha=0.9)
            else:
                ax.plot(centers[i][good], rows[good], lw=0.7,
                        color=ramp(i / max(len(centers) - 1, 1)))
    for x in excluded:
        ax.axvline(x, color=accent, lw=0.8, ls=":", alpha=0.8)
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    style(ax, "x, cross dispersion (px)", "y, dispersion (px)",
          f"Master white, {len(centers)} traces"
          + (f", {len(excluded)} peak{'s' if len(excluded) != 1 else ''} dropped"
             if len(excluded) else "")
          + (f", {n_lost} lost" if n_lost else ""))

    # zoom, at full pixel scale
    ax = fig.add_subplot(gs[0, 1])
    cx, cy = int(zoom_centre[0] * nx), int(zoom_centre[1] * ny)
    x0, x1 = max(0, cx - zoom_size // 2), min(nx, cx + zoom_size // 2)
    y0, y1 = max(0, cy - zoom_size // 2), min(ny, cy + zoom_size // 2)
    cut = coadd[y0:y1, x0:x1]
    zlo, zhi = np.percentile(cut, [5, 99.5])
    ax.imshow(cut, origin="lower", cmap="gray", vmin=zlo, vmax=zhi,
              extent=[x0, x1, y0, y1], aspect="equal", interpolation="nearest")
    for i in range(len(centers)):
        good = np.isfinite(centers[i]) & (rows >= y0) & (rows < y1)
        if good.sum() > 1:
            ax.plot(centers[i][good], rows[good], lw=1.1, color=accent, alpha=0.9)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    style(ax, "x (px)", "y (px)", "Zoom, full pixel scale")

    # slope against the limit
    ax = fig.add_subplot(gs[0, 2])
    limit = window / step
    peak_slope = 0.0
    for i in range(len(centers)):
        good = np.isfinite(centers[i])
        if good.sum() < 6 or stats[i]["lost"]:
            continue
        y, x = rows[good], centers[i][good]
        # the derivative of the fitted shape, since row-to-row differences
        # of a centroid are dominated by measurement noise
        slope = np.abs(np.polyval(np.polyder(np.polyfit(y, x, 3)), y))
        peak_slope = max(peak_slope, float(slope.max()))
        ax.plot(y, slope, lw=0.8, color=ramp(i / max(len(centers) - 1, 1)),
                alpha=0.85)
    ax.axhline(limit, color=accent, lw=1.6)
    ax.annotate(f"tracer limit, {limit:.2f} px/row", (0.98, limit),
                xycoords=("axes fraction", "data"), xytext=(0, 5),
                textcoords="offset points", ha="right", fontsize=8.5, color=accent)
    ax.set_ylim(0, max(limit * 1.15, peak_slope * 1.15))
    style(ax, "y, dispersion (px)", "|dx/dy| (px per row)",
          "Trace slope against what the tracer can follow")

    # polynomial residual
    ax = fig.add_subplot(gs[1, 1:])
    width = 0.8 / len(degrees)
    index = np.arange(len(stats))
    keep_mask = np.array([not e["lost"] for e in stats])
    for k, degree in enumerate(degrees):
        values = np.array([e["residual"].get(degree, np.nan) for e in stats])
        ax.bar(index[keep_mask] + k * width, values[keep_mask], width=width,
               label=f"degree {degree}" + (" (in use)" if degree == 3 else ""),
               color=ramp(0.15 + 0.7 * k / max(len(degrees) - 1, 1)))
    for i in index[~keep_mask]:
        ax.axvspan(i - 0.1, i + 0.9, color=accent, alpha=0.12, lw=0)
    ax.axhline(0.1, color=accent, lw=1.4)
    ax.annotate("0.1 px", (0.995, 0.1), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", ha="right",
                fontsize=8.5, color=accent)
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8, labelcolor=soft, ncol=len(degrees))
    style(ax, "order (trace index)", "RMS of the polynomial fit (px)",
          "How well a polynomial describes each trace")

    fig.text(0.055, 0.955, "Order tracing check", fontsize=16, color=ink,
             weight="semibold")
    fig.text(0.055, 0.922,
             f"{len(centers)} orders, "
             f"{'predictive' if PREDICTIVE else 'pipeline'} tracer, fit window "
             f"+/-{window:g} px every {step} rows. "
             f"Dotted orange lines are peaks dropped as spurious."
             + (" Dashed orange traces were lost." if n_lost else ""),
             fontsize=9, color=soft)

    if out_path:
        fig.savefig(out_path, dpi=125, facecolor=surface)
        print(f"\nwrote {out_path}")
    return fig


os.makedirs(OUT_DIR, exist_ok=True)

print(f"White flats: {WHITE_LOC}")
coadd, n_frames = build_master_white(WHITE_LOC)
print(f"master white: {n_frames} frames, shape {coadd.shape}, "
      f"median {np.median(coadd):.0f} counts")

if SAVE_MASTER_WHITE:
    master_path = os.path.join(OUT_DIR, "master_white.fits")
    if os.path.exists(master_path):
        os.remove(master_path)
    with fitsio.FITS(master_path, "rw") as f:
        f.write(coadd.astype(np.float32))
    print(f"wrote {master_path}")

peaks, excluded, profile = find_order_peaks(coadd)
print(f"\n{len(peaks)} orders found, {len(excluded)} peaks dropped as spurious")

window = choose_trace_window(peaks, TRACE_WINDOW) if AUTO_WINDOW else TRACE_WINDOW

if COMPARE:
    print(f"\ntracing {len(peaks)} orders with the pipeline tracer ...")
    plain, _ = trace_all(coadd, peaks, window=window, predictive=False)
    plain_stats = diagnose(plain, window=window)
    n_plain = sum(1 for e in plain_stats if e["lost"])

    print(f"tracing {len(peaks)} orders with the predictive tracer ...")
    centers, sigmas = trace_all(coadd, peaks, window=window, predictive=True)
    stats = diagnose(centers, window=window)
    n_new = sum(1 for e in stats if e["lost"])

    print("\n" + "=" * 72)
    print("TRACER COMPARISON")
    print("=" * 72)
    med_plain = np.nanmedian([e["residual"].get(3, np.nan) for e in plain_stats])
    med_new = np.nanmedian([e["residual"].get(3, np.nan) for e in stats])
    print(f"  pipeline    {len(peaks) - n_plain:3d}/{len(peaks)} traces kept, "
          f"median cubic residual {med_plain:.3f} px")
    print(f"  predictive  {len(peaks) - n_new:3d}/{len(peaks)} traces kept, "
          f"median cubic residual {med_new:.3f} px")
    if n_new < n_plain:
        print("  The predictive tracer keeps more orders, so leave")
        print("  trace_single_order's predictive flag set, which is its default.")
    print("=" * 72)
    if not PREDICTIVE:
        centers, sigmas, stats = plain, None, plain_stats
else:
    print(f"\ntracing {len(peaks)} orders ...")
    centers, sigmas = trace_all(coadd, peaks, window=window)
    stats = diagnose(centers, window=window)

report(stats, peaks, window=window)

figure = plot_traces(coadd, centers, peaks, excluded, stats, window=window,
                     out_path=os.path.join(OUT_DIR, "tracing_check.png"))
if SHOW:
    plt.show()
