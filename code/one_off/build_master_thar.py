"""
build_master_thar.py

Building a master wavelength solution from one night's white-light flats,
one arc, and one science frame.

This is the expensive, interactive, once-per-instrument-configuration step.
Run it from make_master_thar.py. Everything after it -- every other night,
every science frame -- goes through reduce_spectra.py instead, which only
ever registers this result onto new data.

The solution it produces is not tied to the night it was built from. It is a
description of the instrument: one smooth surface giving m*lambda over
(pixel, order number), plus the map from position on the detector to order
number that lets any later night's traces be identified against it.
"""

import os
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

import config
import frames
import wavelength_solution as ws
from order_tracing import trace_orders


def click_nad_pixels(orders, guesses=None):
    """Pixel positions of the two Na D lines in the seed order.

    Clicking beats fitting here. In a spectrum as line-rich as Arcturus
    nothing automatic can know which deep line you meant, and the whole
    dispersion seed rests on getting that right. You only need to land
    within about 15 px; a Gaussian does the rest.

    Pass `guesses` (from a previous run's printed hint) to skip the clicking
    and re-centroid at those positions instead.
    """
    order = orders[config.NAD_TRACE]
    if guesses is not None:
        pixels = [ws.refine_line_at_guess(order.science_spectrum, g, window=25)
                  for g in guesses]
        print(f"Na D re-centroided from {guesses} -> {[round(p, 2) for p in pixels]}")
        return pixels

    pixels = []
    for wave, name in config.NAD_LINES:
        pixel = ws.click_line(order.science_spectrum,
                              f"{name} ({wave} A) -- trace {config.NAD_TRACE}",
                              window=15)
        if pixel is None:
            raise RuntimeError(f"{name} was not clicked; the seed needs both lines")
        pixels.append(pixel)
    print(f"\n  HARDCODE HINT: NAD_PIXEL_GUESSES = [{pixels[0]:.1f}, {pixels[1]:.1f}]\n")
    return pixels


def _write_summary(path, solution, orders, report, sources, shift_note):
    numbers = [o.order_number for o in orders if o.order_number is not None]
    lines = [
        "ReMUS master wavelength solution",
        "=" * 60,
        f"built            {datetime.now():%Y-%m-%d %H:%M}",
        f"white flats      {sources['white']}",
        f"arc frame        {sources['arc']}",
        f"science frame    {sources['science']}",
        f"orders traced    {max(numbers)} down to {min(numbers)}",
        f"orders fitted    {solution.m_max:.0f} down to {solution.m_min:.0f}"  # the rest still get an axis from the shared surface
        ,
        f"camera focal     {solution.focal_pixels:.0f} px",
        "",
        "quality",
        "-" * 60,
    ]
    for name, ok, message in report.checks:
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name:22s} {message}")
    lines += ["", shift_note, ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def build_master(white_loc, arc_file, science_file, nad_pixel_guesses=None,
                 halpha_pixel_guess=None, hbeta_pixel_guess=None,
                 master_path=None, save=True, plot=True):
    """Derive a master solution and, if it passes every check, save it.

    Returns (solution, report, orders).
    """
    master_path = master_path or config.MASTER_PATH

    print(f"White flats : {white_loc}")
    print(f"Arc frame   : {os.path.basename(arc_file)}")
    print(f"Science     : {os.path.basename(science_file)}\n")

    orders, white = trace_orders(white_loc)
    n_pixels = white.shape[0]

    arc_image = frames.read_image(arc_file)
    science_image = frames.read_image(science_file)
    print("Extracting spectra along every trace ...")
    for order in orders:
        order.thar_spectrum = order.extract_thar(arc_image,
                                                 n_sigma=config.ARC_EXTRACT_NSIGMA)
        order.science_spectrum = order.extract_weighted(
            science_image, n_sigma=config.SCIENCE_EXTRACT_NSIGMA)

    # --- order numbers --------------------------------------------------
    K = ws.compute_grating_K(config.BLAZE_ANGLE_DEG, config.GROOVE_DENSITY_MM)
    print(f"\nGrating constant K = 2 d sin(blaze) = {K:.0f} A\n")
    print("Checking the trace list for a missing order:")
    ws.check_trace_spacing(orders)
    print()
    ws.assign_order_numbers(orders, K, config.ANCHORS, config.DIRECTION)

    # --- dispersion seed ------------------------------------------------
    print("\nSeeding the dispersion from the Na D doublet:")
    nad_pixels = click_nad_pixels(orders, nad_pixel_guesses)
    seed = ws.seed_from_doublet(nad_pixels, [w for w, _ in config.NAD_LINES],
                                orders[config.NAD_TRACE].order_number, n_pixels, K=K)

    # --- atlas ----------------------------------------------------------
    print()
    reference = ws.reference_lines_for(
        config.ATLAS_PATH, seed.A, seed.B, n_pixels,
        config.EXPECTED_LINE_SIGMA_PIXELS,
        amplitude_min=config.ATLAS_AMPLITUDE_MIN, dominance=config.ATLAS_DOMINANCE)

    # --- lock -----------------------------------------------------------
    print()
    detections = ws.detect_all_orders(
        orders, expected_sigma_pixels=config.EXPECTED_LINE_SIGMA_PIXELS)
    order_numbers = [o.order_number for o in orders]

    print()
    locked, lock_snr = ws.lock_seed(
        detections, order_numbers, reference, seed, n_pixels,
        expected_sigma_pixels=config.EXPECTED_LINE_SIGMA_PIXELS)
    print()
    _, order_number_margin = ws.check_order_number_offset(
        detections, order_numbers, reference, locked, n_pixels,
        expected_sigma_pixels=config.EXPECTED_LINE_SIGMA_PIXELS)

    # --- solve ----------------------------------------------------------
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

    # --- judge ----------------------------------------------------------
    ws.attach_solution(orders, solution)
    ws.store_matches(orders, matches, keep)
    report = ws.assess(orders, solution, matches, keep, residuals, n_pixels,
                       lock_snr, m_degree, correction_degree,
                       order_number_margin=order_number_margin, **config.QUALITY)

    # --- an independent check on lines that took no part in the fit -----
    stellar = [(orders[config.NAD_TRACE].order_number, p, w)
               for p, (w, _) in zip(nad_pixels, config.NAD_LINES)]
    for guess, (trace, wave) in zip((halpha_pixel_guess, hbeta_pixel_guess),
                                    config.ANCHORS):
        if guess is not None:
            pixel = ws.refine_line_at_guess(orders[trace].science_spectrum, guess,
                                            window=60)
            stellar.append((orders[trace].order_number, pixel, wave))
    print()
    shift, scatter, _ = ws.measure_frame_shift(stellar, solution)
    print()
    offset = ws.diagnose_frame_offset(orders, solution, shift)
    shift_note = (f"science frame sits {shift:+.1f} px from the arc "
                  f"({offset['verdict']}); axis left in the arc frame")

    # --- save -----------------------------------------------------------
    if report.passed and save:
        os.makedirs(os.path.dirname(master_path), exist_ok=True)
        os.makedirs(config.MASTER_ARCHIVE_DIR, exist_ok=True)
        ws.save_solution(master_path, solution, orders, report, white=white,
                         atlas_path=config.ATLAS_PATH)
        sources = {"white": white_loc,
                   "arc": os.path.basename(arc_file),
                   "science": os.path.basename(science_file)}
        _write_summary(config.MASTER_SUMMARY_PATH, solution, orders, report,
                       sources, shift_note)
        label = os.path.basename(os.path.normpath(os.path.dirname(white_loc)))
        archive = os.path.join(config.MASTER_ARCHIVE_DIR,
                               f"master_{label}_{datetime.now():%Y%m%d-%H%M}.pkl")
        shutil.copy2(master_path, archive)
        print(f"  summary written to {config.MASTER_SUMMARY_PATH}")
        print(f"  archived as {os.path.basename(archive)}")
    elif not report.passed:
        print("NOT saving: the solution failed at least one check above. Fix the cause "
              "rather than loosening the threshold -- the checks are the only thing "
              "standing between a wrong wavelength axis and your science.")

    if plot:
        ws.plot_calibrated_orders(orders, spectrum_attr="science_spectrum",
                                  title="wavelength-calibrated orders, master build")
        plt.show()

    return solution, report, orders
