"""
ReMUS.py

Driver. Traces the orders on a white-light flat, extracts ThAr and science
spectra along them, builds (or reuses) a wavelength solution, and refuses
to save one that does not pass its quality checks.

Two modes, set by BUILD_NEW_MASTER:

  True   Build a master solution from scratch against the ThAr atlas.
         Needs one click on each Na D line to seed the dispersion. Do this
         once per instrument configuration.

  False  Reuse the saved master, correcting only the rigid pixel shift
         since it was built. This is the normal night-to-night path.
"""

import glob
import os

import fitsio
import matplotlib.pyplot as plt
import numpy as np

from order_tracing import trace_orders
import wavelength_solution as ws

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
SPECTRA_LOC = "/Users/ethan/Desktop/University/ReMES/spectra/2025-03-12/"
WHITE_LOC = os.path.join(SPECTRA_LOC, "White")
THAR_LOC = os.path.join(SPECTRA_LOC, "Flat")
SCIENCE_FILE = os.path.join(SPECTRA_LOC, "Light", "Arcturus",
                            "Arcturus_Light_300_secs_2025-03-13T00-56-21_005.fits")
SOLUTION_PATH = os.path.join(SPECTRA_LOC, "wavelength_solution.pkl")
ATLAS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "thar_linelist", "ThAr_lines.dat")

THAR_FILE_INDEX = 2      # which ThAr frame to use; any clean one will do

BUILD_NEW_MASTER = True

# ----------------------------------------------------------------------
# Instrument
# ----------------------------------------------------------------------
BLAZE_ANGLE_DEG = 64.35
GROOVE_DENSITY_MM = 31.6

# m_i = m0 + DIRECTION * trace_index. -1 means trace index 0 is the bluest
# (highest order number), which is what the anchors below imply and what
# the white-flat brightness envelope confirms.
DIRECTION = -1

# Orders whose identification is trusted: Hbeta and Halpha, 30 traces and
# 30 orders apart. Both anchors must imply the same m0 or assign_order_numbers
# refuses to continue.
ANCHORS = [(50, 6562.8),    # (trace index, rest wavelength) -- H-alpha
           (20, 4861.3)]    #                                -- H-beta

# The dispersion seed: click both Na D lines in this order's science
# spectrum. A doublet 5.97 A apart with two unmistakable components, in a
# star far too line-rich to identify anything automatically. Everything
# else -- every order, every pixel -- follows from this one measurement
# plus the order numbers, because m*lambda is a shared function of pixel.
NAD_TRACE = 40
NAD_LINES = [(5889.95, "Na D2"), (5895.92, "Na D1")]

# Fill these in from the pixels printed by a previous run to skip the
# clicking. Same instrument and same target means the same pixels; they
# still get re-centroided, so a stale value is fine as long as it is close.
NAD_PIXEL_GUESSES = None          # this data: [2559.0, 2868.9]

# Optional, and worth having: the pixel positions of Halpha and Hbeta in
# the SCIENCE spectra of their anchor traces. They take no part in the fit.
# They are used afterwards to check the finished solution against lines in
# three different orders, and to measure how far the science exposure has
# drifted from the arc (see ws.measure_frame_shift). Leave as None to skip.
HALPHA_PIXEL_GUESS = None         # this data: 2367.6, in trace ANCHORS[0]
HBETA_PIXEL_GUESS = None          # this data: 1222.0, in trace ANCHORS[1]

# The wavelength axis is in the arc frame -- observed wavelengths, as
# measured. Set this True to divide out the offset measured from the
# stellar anchors above, putting the science spectra in their own rest
# frame. That is a multiplicative rescaling, identical in every order, so
# it cannot disturb how the orders overlap. It is off by default because
# the offset is the target's velocity: something to measure from the data,
# not to calibrate out of it.
SCIENCE_REST_FRAME = False

# ----------------------------------------------------------------------
# Calibration settings
# ----------------------------------------------------------------------
EXPECTED_LINE_SIGMA_PIXELS = 3.3   # instrumental profile; ~7.8 px FWHM here
ATLAS_AMPLITUDE_MIN = 200.0        # reference lines must be strong ...
ATLAS_DOMINANCE = 5.0              # ... and dominate their own resolution element

# Quality gates. A solution that fails any of these is reported and NOT
# saved. Tighten them once you know what this instrument actually delivers.
QUALITY = dict(
    max_rms_ma=15.0,          # per-line residual against the atlas
    max_cv_ma=20.0,           # same, for lines held out of the fit
    max_overlap_ms=600.0,     # adjacent orders must agree this well
    min_orders_with_lines=0.6,
    min_pixel_coverage=0.75,  # matched lines must span most of each order
    max_trend_ma=6.0,         # no systematic drift with pixel or order
    min_lock_snr=15.0,
    min_order_number_margin=2.0,
)


def pick_thar_file(index=THAR_FILE_INDEX):
    files = sorted(glob.glob(os.path.join(THAR_LOC, "*ThAr*.fits")))
    if not files:
        raise FileNotFoundError(f"no ThAr frames in {THAR_LOC}")
    print(f"Using ThAr frame {index + 1}/{len(files)}: {os.path.basename(files[index])}")
    return files[index]


def get_nad_pixels(orders):
    """Clicked (or hardcoded) pixel positions of the two Na D lines."""
    order = orders[NAD_TRACE]
    if NAD_PIXEL_GUESSES is not None:
        pixels = [ws.refine_line_at_guess(order.science_spectrum, g, window=25)
                  for g in NAD_PIXEL_GUESSES]
        print(f"Na D re-centroided from hardcoded guesses {NAD_PIXEL_GUESSES} -> "
              f"{[round(p, 2) for p in pixels]}")
    else:
        pixels = []
        for wave, name in NAD_LINES:
            p = ws.click_line(order.science_spectrum,
                              f"{name} ({wave} A) -- trace {NAD_TRACE}", window=15)
            if p is None:
                raise RuntimeError(f"{name} was not clicked; the seed needs both lines")
            pixels.append(p)
        print(f"\n  HARDCODE HINT: NAD_PIXEL_GUESSES = "
              f"[{pixels[0]:.1f}, {pixels[1]:.1f}]\n")
    return pixels


def build_master(orders, n_pixels):
    K = ws.compute_grating_K(BLAZE_ANGLE_DEG, GROOVE_DENSITY_MM)
    print(f"\nGrating constant K = 2 d sin(blaze) = {K:.0f} A\n")

    print("Checking the trace list for a missing order:")
    ws.check_trace_spacing(orders)
    print()

    ws.assign_order_numbers(orders, K, ANCHORS, DIRECTION)

    # --- seed ---------------------------------------------------------
    print("\nSeeding the dispersion from the Na D doublet:")
    nad_pixels = get_nad_pixels(orders)
    seed = ws.seed_from_doublet(nad_pixels, [w for w, _ in NAD_LINES],
                                orders[NAD_TRACE].order_number, n_pixels, K=K)

    # --- atlas --------------------------------------------------------
    print()
    sel_wave, sel_amp, full_wave, full_amp = ws.load_atlas(ATLAS_PATH)

    # Resolution in Angstrom at a given wavelength, from the seed: the
    # reference-line selection needs to know what the instrument can
    # actually separate, and that scales with wavelength.
    fwhm_pixels = 2.355 * EXPECTED_LINE_SIGMA_PIXELS

    def resolution_angstrom(wave):
        return fwhm_pixels * wave * 2.0 * seed.B / ((n_pixels - 1) * seed.A)

    reference = ws.select_reference_lines(
        sel_wave, sel_amp, full_wave, full_amp, resolution_angstrom,
        amplitude_min=ATLAS_AMPLITUDE_MIN, dominance=ATLAS_DOMINANCE)

    # --- detect and lock ----------------------------------------------
    print()
    detections = ws.detect_all_orders(orders,
                                      expected_sigma_pixels=EXPECTED_LINE_SIGMA_PIXELS)
    order_numbers = [o.order_number for o in orders]

    print()
    locked, lock_snr = ws.lock_seed(detections, order_numbers, reference, seed, n_pixels,
                                    expected_sigma_pixels=EXPECTED_LINE_SIGMA_PIXELS)
    print()
    _, order_number_margin = ws.check_order_number_offset(
        detections, order_numbers, reference, locked, n_pixels,
        expected_sigma_pixels=EXPECTED_LINE_SIGMA_PIXELS)

    # --- solve --------------------------------------------------------
    print("\nSolving:")
    solution, matches, keep, residuals = ws.solve(orders, detections, reference,
                                                  locked, n_pixels)

    print()
    m_degree, correction_degree, _ = ws.choose_degrees(matches, keep, n_pixels)
    if (m_degree, correction_degree) != (2, (4, 2)):
        print("Refitting with the cross-validated complexity:")
        solution, matches, keep, residuals = ws.solve(
            orders, detections, reference, locked, n_pixels,
            m_degree=m_degree, correction_degree=correction_degree)

    # --- quality gate -------------------------------------------------
    ws.attach_solution(orders, solution)
    ws.store_matches(orders, matches, keep)
    report = ws.assess(orders, solution, matches, keep, residuals, n_pixels,
                       lock_snr, m_degree, correction_degree,
                       order_number_margin=order_number_margin, **QUALITY)

    # --- independent check on stellar lines ---------------------------
    stellar = [(orders[NAD_TRACE].order_number, p, w)
               for p, (w, _) in zip(nad_pixels, NAD_LINES)]
    for guess, (trace, wave) in zip((HALPHA_PIXEL_GUESS, HBETA_PIXEL_GUESS), ANCHORS):
        if guess is not None:
            pixel = ws.refine_line_at_guess(orders[trace].science_spectrum, guess,
                                            window=60)
            stellar.append((orders[trace].order_number, pixel, wave))
    print()
    shift, scatter, rows = ws.measure_frame_shift(stellar, solution)
    print()
    offset = ws.diagnose_frame_offset(orders, solution, shift)

    if report.passed:
        ws.save_solution(SOLUTION_PATH, solution, orders, report)
    else:
        print("NOT saving: the solution failed at least one check above. Fix the "
              "cause rather than loosening the threshold -- the checks are the only "
              "thing standing between a wrong wavelength axis and your science.")

    # The axis stays in the arc frame -- observed wavelengths, which is what
    # the science spectra want. A pixel shift is applied only if the overlap
    # test says the exposure really has moved on the detector; if the offset
    # is Doppler-like it is left in the data, where it belongs, because it
    # is the target's velocity and not a calibration error.
    if offset["verdict"] == "flexure" and abs(shift) > 1.0 and scatter < 3.0:
        ws.attach_solution(orders, solution, pixel_shift=shift)
    elif SCIENCE_REST_FRAME and offset["verdict"] == "doppler":
        velocity = float(np.median([r[5] for r in rows])) if rows else 0.0
        ws.attach_solution(orders, solution, velocity_ms=velocity)
    return solution, report


def reuse_master(orders):
    saved = ws.load_solution(SOLUTION_PATH)
    ws.assign_order_numbers_from_saved(orders, saved)
    ws.apply_saved_solution(orders, saved)


def main():
    orders, white = trace_orders(WHITE_LOC)
    n_pixels = white.shape[0]

    thar_file = pick_thar_file()
    with fitsio.FITS(thar_file, "r") as f:
        thar_image = f[0].read()
    with fitsio.FITS(SCIENCE_FILE, "r") as f:
        science_image = f[0].read()

    print("Extracting spectra along every trace ...")
    for order in orders:
        order.thar_spectrum = order.extract_thar(thar_image)
        order.science_spectrum = order.extract_weighted(science_image, n_sigma=3)

    if BUILD_NEW_MASTER:
        build_master(orders, n_pixels)
    else:
        reuse_master(orders)

    ws.plot_calibrated_orders(orders, spectrum_attr="science_spectrum",
                              title="Arcturus, wavelength-calibrated orders")
    plt.show()


if __name__ == "__main__":
    main()
