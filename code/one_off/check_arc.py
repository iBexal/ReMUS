"""Check why the arc is or is not giving up its ThAr lines.

Standalone diagnostic, not part of the pipeline. Run it on a new
spectrograph when detect_all_orders returns too few lines, or when solve
loses lines as the tolerance tightens instead of gaining them.

detect_arc_lines rejects a peak whose fitted width falls outside
width_tolerance times config.EXPECTED_LINE_SIGMA_PIXELS. That setting
describes the instrumental profile of one spectrograph. Carried onto
another, whose lines are narrower or wider, it throws away every real line
while leaving the noise threshold untouched, so the failure looks like a
signal problem when it is a width problem.

This script detects lines with the width cut switched off, measures the
profile width the arc actually has, and reports how many peaks each cut
removes. It prints the value config.EXPECTED_LINE_SIGMA_PIXELS should
hold, and produces:

  1. one order's arc spectrum with kept and rejected peaks marked;
  2. the distribution of measured line widths, against the window the
     current setting accepts;
  3. what each rejection cut costs, as a waterfall;
  4. lines per order, at the current setting and at the measured one.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

import config
import frames
import wavelength_solution as ws
from order_tracing import trace_orders

# ----------------------------------------------------------------------
# What to look at
# ----------------------------------------------------------------------
NIGHT = os.path.join(config.SPECTRA_ROOT, "2025-03-12")
WHITE_LOC = os.path.join(NIGHT, "White")
ARC_LOC = os.path.join(NIGHT, "Flat")
ARC_FILE = None                   # None takes the earliest arc in ARC_LOC

OUT_DIR = os.path.join(config.PROJECT_ROOT, "reduced", "arc_check")
SHOW = True

# ----------------------------------------------------------------------
# Detection settings, matching wavelength_solution.detect_arc_lines
# ----------------------------------------------------------------------
DETECTION_SIGMA = 7.0             # peak prominence, in robust noise units
SATURATION = 45000.0
MIN_SEPARATION = 8                # pixels
HALF_WINDOW = 9                   # pixels
WIDTH_TOLERANCE = (0.5, 2.0)      # multiples of the expected sigma
MAX_FIT_RESIDUAL = 0.15           # fraction of the line amplitude
CONTINUUM_WINDOW = 201            # pixels

# The search runs with this far wider width range, so the real profile can
# be measured rather than assumed.
SEARCH_SIGMA_LIMITS = (0.4, 20.0)

ORDER_TO_PLOT = None              # None picks the order with the most lines


def detect_with_reasons(spectrum, expected_sigma_pixels,
                        detection_sigma=DETECTION_SIGMA, saturation=SATURATION,
                        min_separation=MIN_SEPARATION, half_window=HALF_WINDOW,
                        max_fit_residual=MAX_FIT_RESIDUAL,
                        continuum_window=CONTINUUM_WINDOW,
                        search_sigma_limits=SEARCH_SIGMA_LIMITS):
    """Detect arc lines without a width cut, recording every rejection.

    Mirrors wavelength_solution.detect_arc_lines except that the width cut
    is not applied, so the measured widths of real lines can be seen. The
    caller applies the cut afterwards and sees what it costs.

    Parameters
    ----------
    spectrum : ndarray
        Extracted arc spectrum of one order, shape (n_pixels,).
    expected_sigma_pixels : float
        Starting value for the fitted sigma, in pixels.
    detection_sigma : float, optional
        Peak prominence threshold, in units of the robust noise.
    saturation : float, optional
        Counts at or above which a peak is discarded.
    min_separation : int, optional
        Least separation between detected peaks, in pixels.
    half_window : int, optional
        Half width of the fitting window, in pixels.
    max_fit_residual : float, optional
        Largest allowed RMS fit residual, as a fraction of the amplitude.
    continuum_window : int, optional
        Median filter width used for the continuum, in pixels.
    search_sigma_limits : tuple of float, optional
        Widest range of fitted sigma kept at all, in pixels. Only there to
        drop runaway fits, not to select lines.

    Returns
    -------
    lines : ndarray
        Shape (n, 4), columns (pixel, sigma, amplitude, signal to noise),
        for peaks passing everything except the width cut.
    reasons : dict
        Count of peaks removed by each cut, keyed 'saturated', 'edge',
        'fit failed', 'bad shape' and 'fit residual', plus 'found' and
        'kept'.
    noise : float
        Robust noise of the continuum-subtracted spectrum, in counts.
    """
    reasons = dict.fromkeys(
        ["found", "saturated", "edge", "fit failed", "bad shape",
         "fit residual", "kept"], 0)
    s = np.asarray(spectrum, float)
    if s.size == 0 or not np.all(np.isfinite(s)):
        return np.zeros((0, 4)), reasons, np.nan

    continuum = median_filter(s, continuum_window)
    resid = s - continuum
    mad = np.median(np.abs(resid - np.median(resid)))
    noise = 1.4826 * mad if mad > 0 else np.std(resid)
    if noise <= 0:
        return np.zeros((0, 4)), reasons, np.nan

    peaks, _ = find_peaks(resid, prominence=detection_sigma * noise,
                          distance=min_separation)
    reasons["found"] = len(peaks)

    out = []
    for p in peaks:
        if s[p] >= saturation:
            reasons["saturated"] += 1
            continue
        lo, hi = max(0, p - half_window), min(len(s), p + half_window + 1)
        if hi - lo < 7:
            reasons["edge"] += 1
            continue
        x = np.arange(lo, hi)
        y = resid[lo:hi]
        try:
            popt, _ = curve_fit(ws._gaussian, x, y,
                                p0=(y.max(), float(p), expected_sigma_pixels, 0.0))
        except Exception:
            reasons["fit failed"] += 1
            continue
        amp, mu, sigma, _ = popt
        sigma = abs(sigma)
        if (amp <= 0 or not (lo + 1 < mu < hi - 1)
                or not search_sigma_limits[0] < sigma < search_sigma_limits[1]):
            reasons["bad shape"] += 1
            continue
        model = ws._gaussian(x, *popt)
        if np.sqrt(np.mean((y - model) ** 2)) / amp > max_fit_residual:
            reasons["fit residual"] += 1
            continue
        out.append((mu, sigma, amp, amp / noise))
        reasons["kept"] += 1

    return (np.array(out) if out else np.zeros((0, 4))), reasons, noise


def measure_profile(all_lines, snr_min=20.0):
    """Measure the instrumental profile width from the detected lines.

    Only lines well above the noise are used, since a weak line's fitted
    width is poorly determined and biases the estimate upwards.

    Parameters
    ----------
    all_lines : ndarray
        Stacked detections, shape (n, 4), columns (pixel, sigma, amplitude,
        signal to noise).
    snr_min : float, optional
        Least signal to noise a line must have to count. Default 20.0.

    Returns
    -------
    sigma : float
        Median fitted sigma of the strong lines, in pixels.
    spread : tuple of float
        16th and 84th percentiles of the same, in pixels.
    n_used : int
        Number of lines the estimate rests on.
    """
    if len(all_lines) == 0:
        return np.nan, (np.nan, np.nan), 0
    strong = all_lines[all_lines[:, 3] >= snr_min]
    if len(strong) < 20:
        strong = all_lines
    sigma = float(np.median(strong[:, 1]))
    spread = tuple(float(v) for v in np.percentile(strong[:, 1], [16, 84]))
    return sigma, spread, len(strong)


def count_with_width_cut(per_order, expected_sigma, tolerance=WIDTH_TOLERANCE):
    """Line count per order that survives a given width cut.

    Parameters
    ----------
    per_order : list of ndarray
        Detections per order, each shape (n, 4).
    expected_sigma : float
        Assumed instrumental sigma, in pixels.
    tolerance : tuple of float, optional
        Lower and upper multiples of expected_sigma that a line's fitted
        sigma must lie between.

    Returns
    -------
    counts : ndarray
        Number of surviving lines per order.
    """
    lo, hi = tolerance[0] * expected_sigma, tolerance[1] * expected_sigma
    return np.array([int(np.sum((d[:, 1] > lo) & (d[:, 1] < hi))) if len(d) else 0
                     for d in per_order])


def report(per_order, reasons, measured, spread, n_used, noise_levels):
    """Print the rejection budget and the width verdict.

    Parameters
    ----------
    per_order : list of ndarray
        Detections per order, each shape (n, 4), width cut not applied.
    reasons : dict
        Summed rejection counts from detect_with_reasons.
    measured : float
        Median fitted line sigma, in pixels.
    spread : tuple of float
        16th and 84th percentiles of the fitted sigma, in pixels.
    n_used : int
        Number of lines the width estimate rests on.
    noise_levels : ndarray
        Robust noise per order, in counts.

    Returns
    -------
    suggested : float
        The value config.EXPECTED_LINE_SIGMA_PIXELS should hold, rounded
        to one decimal.
    """
    current = config.EXPECTED_LINE_SIGMA_PIXELS
    suggested = round(measured, 1) if np.isfinite(measured) else current

    now = count_with_width_cut(per_order, current)
    then = count_with_width_cut(per_order, suggested)
    found = sum(len(d) for d in per_order)

    print("\n" + "=" * 72)
    print("WHERE THE PEAKS WENT")
    print("=" * 72)
    order = ["found", "saturated", "edge", "fit failed", "bad shape",
             "fit residual", "kept"]
    total = max(reasons["found"], 1)
    for key in order:
        bar = "#" * int(round(40 * reasons[key] / total))
        print(f"  {key:14s} {reasons[key]:6d}  {100 * reasons[key] / total:5.1f}%  {bar}")
    print("  (the width cut is not applied above, it is applied below)")

    print("\n" + "=" * 72)
    print("THE INSTRUMENTAL PROFILE")
    print("=" * 72)
    if not np.isfinite(measured):
        print("  No lines survived even without a width cut. The problem is")
        print("  upstream: check the extraction, or lower DETECTION_SIGMA.")
        print("=" * 72)
        return suggested

    print(f"  measured line sigma  {measured:.2f} px "
          f"(16th-84th percentile {spread[0]:.2f} to {spread[1]:.2f}, "
          f"{n_used} strong lines)")
    print(f"  that is             {2.355 * measured:.2f} px FWHM")
    print(f"  config setting      EXPECTED_LINE_SIGMA_PIXELS = {current}")
    print(f"  which accepts       {WIDTH_TOLERANCE[0] * current:.2f} to "
          f"{WIDTH_TOLERANCE[1] * current:.2f} px")

    inside = WIDTH_TOLERANCE[0] * current < measured < WIDTH_TOLERANCE[1] * current
    print()
    if not inside:
        print(f"  [FAIL] The real profile is outside the window the current setting")
        print(f"         accepts, so the width cut is throwing away real lines.")
        print(f"         Set EXPECTED_LINE_SIGMA_PIXELS = {suggested} in config.py.")
    elif abs(measured - current) / current > 0.25:
        print(f"  [WARN] The real profile is {measured / current:.2f}x the setting. It")
        print(f"         still falls inside the window, but the atlas selection uses")
        print(f"         this value too, so set it to {suggested}.")
    else:
        print(f"  [PASS] The setting matches the instrument.")

    print(f"\n  lines kept at {current:<4} : {now.sum():5d} total, "
          f"median {int(np.median(now))} per order")
    print(f"  lines kept at {suggested:<4} : {then.sum():5d} total, "
          f"median {int(np.median(then))} per order")
    print(f"  peaks found before the width cut: {found}")

    empty_now = int(np.sum(now < 4))
    empty_then = int(np.sum(then < 4))
    print(f"\n  orders with fewer than 4 lines: {empty_now} now, "
          f"{empty_then} at the measured width, out of {len(per_order)}")

    print("\n" + "=" * 72)
    print("IS IT NOISE?")
    print("=" * 72)
    finite = noise_levels[np.isfinite(noise_levels)]
    if len(finite):
        print(f"  robust noise per order: median {np.median(finite):.0f} counts, "
              f"range {finite.min():.0f} to {finite.max():.0f}")
    snr = np.concatenate([d[:, 3] for d in per_order if len(d)]) \
        if any(len(d) for d in per_order) else np.array([])
    if len(snr):
        print(f"  detected line SNR: median {np.median(snr):.0f}, "
              f"{int(np.sum(snr > 50))} above 50, {int(np.sum(snr > 20))} above 20")
        if np.median(snr) > 20 and found > 500:
            print("  [PASS] There is plenty of signal. A shortage of lines after this")
            print("         point is a cut throwing them away, not a noise problem.")
        else:
            print("  [WARN] The lines really are weak. Check the arc exposure and the")
            print("         extraction width ARC_EXTRACT_NSIGMA before touching cuts.")
    print("=" * 72)
    return suggested


def plot_arc(spectra, per_order, reasons, measured, suggested, which,
             out_path=None):
    """Draw the arc diagnostics.

    Parameters
    ----------
    spectra : list of ndarray
        Extracted arc spectrum per order.
    per_order : list of ndarray
        Detections per order, each shape (n, 4), width cut not applied.
    reasons : dict
        Summed rejection counts from detect_with_reasons.
    measured : float
        Median fitted line sigma, in pixels.
    suggested : float
        Value proposed for config.EXPECTED_LINE_SIGMA_PIXELS.
    which : int
        Index of the order drawn in the spectrum panel.
    out_path : str, optional
        Where to save the figure. Default None, meaning it is not saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure, so the caller can show or further edit it.
    """
    ink, soft, faint = "#0b0b0b", "#52514e", "#8a8880"
    accent, good, surface = "#eb6834", "#2f7d6d", "#fcfcfb"
    current = config.EXPECTED_LINE_SIGMA_PIXELS

    fig = plt.figure(figsize=(15.5, 9.0), facecolor=surface)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 1], hspace=0.45,
                          wspace=0.2, left=0.06, right=0.98, top=0.86,
                          bottom=0.07)

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

    # one order's arc, with what the width cut keeps and drops
    ax = fig.add_subplot(gs[0, :])
    s = spectra[which]
    ax.plot(s, lw=0.6, color=ink)
    d = per_order[which]
    if len(d):
        lo, hi = WIDTH_TOLERANCE[0] * current, WIDTH_TOLERANCE[1] * current
        keep = (d[:, 1] > lo) & (d[:, 1] < hi)
        at = np.clip(d[:, 0].astype(int), 0, len(s) - 1)
        height = s[at] + 0.04 * (np.nanmax(s) - np.nanmin(s))
        ax.plot(d[keep, 0], height[keep], "v", ms=5, color=good, ls="none",
                label=f"kept by the current width cut ({int(keep.sum())})")
        ax.plot(d[~keep, 0], height[~keep], "v", ms=5, color=accent, ls="none",
                label=f"dropped by it ({int((~keep).sum())})")
        ax.legend(frameon=False, fontsize=8, labelcolor=soft, ncol=2, loc="upper right")
    style(ax, "pixel", "counts",
          f"Arc spectrum, order index {which}")

    # measured widths against the accepted window
    ax = fig.add_subplot(gs[1, 0])
    sig = np.concatenate([d[:, 1] for d in per_order if len(d)]) \
        if any(len(d) for d in per_order) else np.array([])
    if len(sig):
        ax.hist(sig, bins=60, color="#3b4a6b", edgecolor="none")
    ax.axvspan(WIDTH_TOLERANCE[0] * current, WIDTH_TOLERANCE[1] * current,
               color=good, alpha=0.13, lw=0)
    ax.axvline(current, color=good, lw=1.6)
    ax.annotate(f"current, {current}", (current, 0.97), xycoords=("data", "axes fraction"),
                xytext=(4, 0), textcoords="offset points", fontsize=8.5,
                color=good, va="top")
    if np.isfinite(measured):
        ax.axvline(measured, color=accent, lw=1.6)
        ax.annotate(f"measured, {measured:.2f}", (measured, 0.86),
                    xycoords=("data", "axes fraction"), xytext=(4, 0),
                    textcoords="offset points", fontsize=8.5, color=accent, va="top")
    style(ax, "fitted line sigma (px)", "lines",
          "Line width, against the window the setting accepts")

    # what each cut costs
    ax = fig.add_subplot(gs[1, 1])
    keys = ["saturated", "edge", "fit failed", "bad shape", "fit residual"]
    values = [reasons[k] for k in keys]
    lo, hi = WIDTH_TOLERANCE[0] * current, WIDTH_TOLERANCE[1] * current
    width_cut = int(np.sum([(d[:, 1] <= lo).sum() + (d[:, 1] >= hi).sum()
                            for d in per_order if len(d)]))
    keys, values = keys + ["width cut"], values + [width_cut]
    colours = ["#3b4a6b"] * 5 + [accent]
    ax.barh(range(len(keys)), values, color=colours)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys, fontsize=8.5, color=soft)
    ax.invert_yaxis()
    for i, v in enumerate(values):
        ax.annotate(f"{v}", (v, i), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=soft)
    style(ax, f"peaks removed, of {reasons['found']} found", None,
          "What each cut costs")

    # lines per order, now and at the measured width
    ax = fig.add_subplot(gs[2, :])
    now = count_with_width_cut(per_order, current)
    then = count_with_width_cut(per_order, suggested)
    index = np.arange(len(per_order))
    ax.bar(index - 0.21, now, width=0.42, color=good,
           label=f"EXPECTED_LINE_SIGMA_PIXELS = {current} (current)")
    ax.bar(index + 0.21, then, width=0.42, color=accent,
           label=f"EXPECTED_LINE_SIGMA_PIXELS = {suggested} (measured)")
    ax.axhline(4, color=faint, lw=1.0, ls="--")
    ax.annotate("4 lines, the minimum an order needs", (0.005, 4),
                xycoords=("axes fraction", "data"), xytext=(0, 5),
                textcoords="offset points", ha="left", fontsize=8, color=faint)
    ax.legend(frameon=False, fontsize=8, labelcolor=soft, ncol=2)
    style(ax, "order (trace index)", "ThAr lines kept", "Lines per order")

    fig.text(0.06, 0.960, "Arc line detection check", fontsize=16, color=ink,
             weight="semibold")
    fig.text(0.06, 0.928,
             f"{sum(len(d) for d in per_order)} peaks found over {len(per_order)} "
             f"orders, before the width cut. Orange marks what the current setting "
             f"discards.", fontsize=9, color=soft)

    if out_path:
        fig.savefig(out_path, dpi=125, facecolor=surface)
        print(f"\nwrote {out_path}")
    return fig


os.makedirs(OUT_DIR, exist_ok=True)

arc_file = ARC_FILE or frames.list_arcs(ARC_LOC)[0][0]
print(f"White flats : {WHITE_LOC}")
print(f"Arc frame   : {os.path.basename(arc_file)}\n")

orders, white = trace_orders(WHITE_LOC)
arc_image = frames.read_image(arc_file)
print("\nExtracting the arc along every trace ...")
spectra = [o.extract_thar(arc_image, n_sigma=config.ARC_EXTRACT_NSIGMA)
           for o in orders]

print("Detecting lines with the width cut switched off ...")
per_order, noise_levels = [], []
totals = dict.fromkeys(
    ["found", "saturated", "edge", "fit failed", "bad shape",
     "fit residual", "kept"], 0)
for s in spectra:
    lines, reasons, noise = detect_with_reasons(
        s if s is not None else np.array([]), config.EXPECTED_LINE_SIGMA_PIXELS)
    per_order.append(lines)
    noise_levels.append(noise)
    for key in totals:
        totals[key] += reasons[key]
noise_levels = np.array(noise_levels, float)

all_lines = np.vstack([d for d in per_order if len(d)]) if any(
    len(d) for d in per_order) else np.zeros((0, 4))
measured, spread, n_used = measure_profile(all_lines)
suggested = report(per_order, totals, measured, spread, n_used, noise_levels)

which = ORDER_TO_PLOT
if which is None:
    which = int(np.argmax([len(d) for d in per_order]))
figure = plot_arc(spectra, per_order, totals, measured, suggested, which,
                  out_path=os.path.join(OUT_DIR, "arc_check.png"))
if SHOW:
    plt.show()
