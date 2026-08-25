"""Build a master wavelength solution for one instrument configuration.

The build takes one night's white-light flats, one arc frame and one
science frame, and is interactive. It is run once per configuration, from
make_master_thar.py; later nights go through reduce_spectra.py, which
registers this result onto new data. The result describes the instrument
rather than the night: a smooth surface giving m*lambda over (pixel,
order number), plus the map from detector position to order number used
to identify later traces.

Main entry point: build_master.
"""

import os
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

import config
import flat_field
import frames
import wavelength_solution as ws
from order_tracing import trace_orders


def click_nad_pixels(orders, guesses=None):
    """Locate the two Na D lines in the seed order.

    The lines are clicked rather than found automatically because in a
    line-rich spectrum only the operator knows which deep line is meant,
    and the dispersion seed rests on that. A click need only land within
    about 15 pixels of the line; a Gaussian fit supplies the centre.

    Parameters
    ----------
    orders : list of Order
        Traced orders carrying science_spectrum. The seed order is
        orders[config.NAD_TRACE].
    guesses : sequence of float, optional
        Approximate pixel positions of the two lines, in the order of
        config.NAD_LINES, re-centroided instead of clicked. Default
        None, meaning the lines are clicked and the positions found are
        printed for reuse.

    Returns
    -------
    pixels : list of float
        Refined pixel positions of the two Na D lines, in the order of
        config.NAD_LINES.

    Raises
    ------
    RuntimeError
        If a line is not clicked; the seed needs both.
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
                              f"{name} ({wave} A), trace {config.NAD_TRACE}",
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
        # orders outside this range still get an axis from the surface
        f"orders fitted    {solution.m_max:.0f} down to {solution.m_min:.0f}"
        ,
        f"camera focal     {solution.focal_pixels:.0f} px",
        # 83 km/s hangs on this, and nothing about the numbers below
        # reveals it, so it is stated where the numbers are.
        f"wavelengths      {'AIR' if config.ATLAS_AIR else 'VACUUM'} "
        f"(ATLAS_AIR = {config.ATLAS_AIR})",
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
    """Derive a master wavelength solution and save it if it passes.

    The orders are traced from the flats, the arc and science spectra
    are extracted, order numbers are assigned, the dispersion is seeded
    from the Na D doublet, atlas lines are matched and a surface is
    fitted and assessed. Halpha, Hbeta and the Na D lines take no part
    in the fit, so they serve as an independent check on the science
    frame's shift from the arc. The solution is written, summarised and
    archived only if every quality check passed.

    Parameters
    ----------
    white_loc : str
        Directory of white-light flat FITS files.
    arc_file : str
        Path to the arc frame.
    science_file : str
        Path to the science frame used for the stellar checks.
    nad_pixel_guesses : sequence of float, optional
        Approximate pixel positions of the two Na D lines, which skip
        the clicking step. Default None, meaning the lines are clicked.
    halpha_pixel_guess : float, optional
        Approximate pixel position of Halpha in its anchor trace, used
        for the independent shift check. Default None, meaning Halpha is
        left out of that check.
    hbeta_pixel_guess : float, optional
        As halpha_pixel_guess, for Hbeta. Default None.
    master_path : str, optional
        Path to save the solution to. Default None, meaning
        config.MASTER_PATH.
    save : bool, optional
        Save, summarise and archive a solution that passes every check.
        Default True.
    plot : bool, optional
        Show the calibrated orders at the end. Default True.

    Returns
    -------
    solution : WavelengthSolution
        The fitted surface and the order number map.
    report : QualityReport
        The checks measured on the solution and whether all passed.
    orders : list of Order
        The traced orders, mutated in place: the extracted spectra,
        order numbers, wavelength axes and matched lines are set on
        them.
    """
    master_path = master_path or config.MASTER_PATH

    print(f"White flats : {white_loc}")
    print(f"Arc frame   : {os.path.basename(arc_file)}")
    print(f"Science     : {os.path.basename(science_file)}\n")

    orders, white = trace_orders(white_loc)
    n_pixels = white.shape[0]

    # The master's own reference arc has to be reduced the way the arcs
    # registered against it will be. A pixel response is a fixed
    # multiplicative pattern, so correcting one side and not the other
    # leaves a mismatch in every later cross-correlation.
    if config.FLAT_FIELD:
        print()
        flat_field.flat_field_orders(orders, white)

    arc_image = frames.read_image(arc_file)
    science_image = frames.read_image(science_file)
    print("Extracting spectra along every trace ...")
    for order in orders:
        order.thar_spectrum = order.extract_thar(arc_image,
                                                 n_sigma=config.ARC_EXTRACT_NSIGMA)
        order.science_spectrum = order.extract_weighted(
            science_image, n_sigma=config.SCIENCE_EXTRACT_NSIGMA)
    if config.FLAT_FIELD:
        if config.FLAT_FIELD_ARCS:
            flat_field.apply_pixel_response(orders, "thar_spectrum",
                                            response_attr="pixel_response_arc",
                                            verbose=False)
        flat_field.apply_pixel_response(orders, "science_spectrum", verbose=False)

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
        orders, expected_sigma_pixels=config.EXPECTED_LINE_SIGMA_PIXELS,
        saturation=config.ARC_SATURATION)
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
              "rather than loosening the threshold. The checks are the only thing "
              "standing between a wrong wavelength axis and your science.")

    if plot:
        ws.plot_calibrated_orders(orders, spectrum_attr="science_spectrum",
                                  title="wavelength-calibrated orders, master build")
        plt.show()

    return solution, report, orders
