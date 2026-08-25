"""Locate FITS frames, read them, and read the times they were taken.

Times let an arc be chosen for a science exposure by comparison of
timestamps rather than by file order. Every time is reported with the
source it came from, since the header and the filename are different
clocks.

read_image is the one place a frame enters the pipeline, so it is where
the detector calibration happens too: optional bias and dark subtraction,
controlled by config and off by default.

Main entry points: read_image, read_header, mid_exposure_time, list_frames,
list_arcs, nearest_arc, bracketing_arcs, describe_arc_choice,
master_frame, calibrate_image.
"""

import glob
import os
import re
from datetime import datetime, timedelta

import fitsio
import numpy as np

import config

# Filenames carry a time as well, for example
# ..._2025-03-12T22-57-07_008.fits, but it is not the same clock as the
# header: on this data DATE-OBS is UTC and the filename is local, eleven
# hours apart. The header is used where available and the filename only as a
# last resort, with the source reported so a mixture can be spotted.
_FILENAME_TIME = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})")


def _read_raw(path, transpose=None):
    """Read a frame with no calibration applied.

    Used for the calibration frames themselves, which must not be
    calibrated against each other, and by read_image once it has decided
    what to subtract.

    Parameters
    ----------
    path : str
        Path to the FITS file.
    transpose : bool, optional
        Transpose the frame as it is read. Default None, meaning
        config.TRANSPOSE.

    Returns
    -------
    image : ndarray
        Data of the primary HDU as float, 2D of shape (ny, nx), rows
        along dispersion.
    """
    if transpose is None:
        transpose = config.TRANSPOSE
    with fitsio.FITS(path, "r") as f:
        image = f[0].read()
    image = np.asarray(image, dtype=float)
    return np.ascontiguousarray(image.T) if transpose else image


def read_image(path, transpose=None, calibrate=None):
    """Read the primary image of a FITS file, in the pipeline's orientation.

    The pipeline needs rows to be the dispersion direction and columns the
    cross dispersion direction. Detectors that read out the other way round
    are transposed here, so every module downstream sees one orientation.
    This is the only place a frame enters the pipeline, so it is the only
    place the transpose, and the bias and dark subtraction, have to happen.

    Bias and dark subtraction are off unless config.APPLY_BIAS or
    config.APPLY_DARK is set. See calibrate_image for what they do.

    Parameters
    ----------
    path : str
        Path to the FITS file.
    transpose : bool, optional
        Transpose the frame as it is read. Default None, meaning
        config.TRANSPOSE.
    calibrate : bool, optional
        Subtract bias and dark. Default None, meaning follow
        config.APPLY_BIAS and config.APPLY_DARK. False forces the raw
        frame, which is what the calibration frames themselves need.

    Returns
    -------
    image : ndarray
        Data of the primary HDU as float, 2D of shape (ny, nx), rows
        along dispersion. Counts are in ADU above bias when bias
        subtraction is on, and raw ADU when it is not.
    """
    image = _read_raw(path, transpose=transpose)
    if calibrate is False:
        return image
    return calibrate_image(image, path, transpose=transpose, force=calibrate)


# ======================================================================
# bias and dark
# ======================================================================
#
# Why this is optional. A bias pedestal is a constant added to every
# pixel, and the wavelength solution is blind to constants: line
# centroids are fitted above a median-filtered continuum, which removes
# the pedestal along with everything else smooth. So leaving the bias in
# costs nothing as long as the only product is a wavelength axis.
#
# It stops being free the moment anything divides. Flat fielding divides
# one frame by another, and (S + B) / (F + B) is not S / F, so a flat
# field applied to un-debiased frames is wrong by an amount that varies
# with how bright the pixel is. Turn APPLY_BIAS on with FLAT_FIELD.
#
# Dark current on this detector is small, around 0.15 ADU/s at 0 C,
# so a 300 s exposure collects tens of ADU against a bias of about 1000.
# It matters for the faintest orders and for the hot pixels in the tail
# of that distribution, and not much else.

# Master frames are 4096 x 4096 here, so reading one per science frame
# would dominate the run time. Keyed by (resolved source, transpose).
_CALIBRATION_CACHE = {}
_CALIBRATION_REPORTED = set()


def _frame_exptime(path):
    """Exposure time of a frame in seconds, 0.0 if it cannot be read."""
    try:
        header = read_header(path)
        return float(header.get("EXPTIME") or header.get("DARKTIME") or 0.0)
    except Exception:
        return 0.0


def _frame_imagetype(path):
    """IMAGETYP of a frame, lower case, empty string if absent."""
    try:
        return str(read_header(path).get("IMAGETYP") or "").strip().lower()
    except Exception:
        return ""


def _combine(paths, method="median", transpose=None):
    """Combine calibration frames into one master.

    Parameters
    ----------
    paths : list of str
        Frames to combine.
    method : str, optional
        "median", or "mean" for a sigma-clipped mean. Default "median".
    transpose : bool, optional
        Passed to _read_raw. Default None, meaning config.TRANSPOSE.

    Returns
    -------
    master : ndarray
        Combined frame, 2D of shape (ny, nx).

    Raises
    ------
    ValueError
        If paths is empty, or the frames are not all the same shape.
    """
    if not paths:
        raise ValueError("no frames to combine")
    stack = []
    for p in paths:
        frame = _read_raw(p, transpose=transpose)
        if stack and frame.shape != stack[0].shape:
            raise ValueError(
                f"{os.path.basename(p)} is {frame.shape} but "
                f"{os.path.basename(paths[0])} is {stack[0].shape}. Calibration "
                f"frames must all be the same shape and binning.")
        stack.append(frame)
    stack = np.array(stack)
    if len(stack) == 1:
        return stack[0]
    if method == "mean":
        # sigma-clipped mean: better noise than a median once there are
        # enough frames, without letting a cosmic ray through
        med = np.median(stack, axis=0)
        scatter = 1.4826 * np.median(np.abs(stack - med), axis=0)
        keep = np.abs(stack - med) <= 5.0 * np.maximum(scatter, 1e-9)
        n = keep.sum(axis=0)
        total = np.where(keep, stack, 0.0).sum(axis=0)
        return np.where(n > 0, total / np.maximum(n, 1), med)
    return np.median(stack, axis=0)


def _resolve_calibration_paths(source, want=None):
    """Expand a calibration setting into a list of frame paths.

    Parameters
    ----------
    source : str or list of str
        A FITS file, a directory searched for config.FRAME_PATTERN, or an
        explicit list of paths.
    want : str, optional
        Keep only frames whose IMAGETYP contains this, where the frames
        carry one. Default None, meaning keep everything. Used so a
        directory holding both a master bias and master darks yields only
        the ones asked for.

    Returns
    -------
    paths : list of str
        Matching paths, sorted.

    Raises
    ------
    FileNotFoundError
        If nothing matches.
    """
    if isinstance(source, (list, tuple)):
        paths = [p for s in source for p in _resolve_calibration_paths(s, want)]
    elif os.path.isdir(source):
        paths = list_frames(source, recursive=True)
    elif os.path.exists(source):
        paths = [source]
    else:
        paths = sorted(glob.glob(source))
    if not paths:
        raise FileNotFoundError(f"no calibration frames at {source}")

    if want:
        typed = [(p, _frame_imagetype(p)) for p in paths]
        # Only filter where the frames actually say what they are, so a
        # directory of untyped frames still works.
        if any(t for _, t in typed):
            filtered = [p for p, t in typed if want in t]
            if filtered:
                paths = filtered
    return sorted(set(paths))


def master_frame(source, want=None, match_exptime=None, transpose=None,
                 method=None, verbose=True):
    """Build, or fetch from the cache, a master calibration frame.

    A single file is used as it is. A directory or a list is grouped by
    exposure time, the group nearest match_exptime is chosen, and its
    frames are combined. Grouping by exposure time is what lets one
    directory hold a 1 s dark and a 900 s dark without them being averaged
    together.

    Parameters
    ----------
    source : str or list of str
        A FITS file, a directory, or a list of paths.
    want : str, optional
        IMAGETYP substring to keep, "bias" or "dark". Default None.
    match_exptime : float, optional
        Exposure time to prefer, in seconds. Default None, meaning
        combine everything found.
    transpose : bool, optional
        Passed to _read_raw. Default None, meaning config.TRANSPOSE.
    method : str, optional
        Combination method. Default None, meaning
        config.CALIBRATION_COMBINE.
    verbose : bool, optional
        Print a line the first time each master is built. Default True.

    Returns
    -------
    master : ndarray
        The master frame, 2D of shape (ny, nx).
    exptime : float
        Exposure time the master represents, in seconds. 0.0 where none
        could be read.
    """
    method = method or getattr(config, "CALIBRATION_COMBINE", "median")
    if transpose is None:
        transpose = config.TRANSPOSE

    paths = _resolve_calibration_paths(source, want)
    exptimes = np.array([_frame_exptime(p) for p in paths])

    chosen = paths
    if match_exptime is not None and len(paths) > 1 and np.any(exptimes > 0):
        # nearest exposure time in the log, so 1 s against 900 s is a
        # bigger difference than 900 s against 1200 s
        safe = np.where(exptimes > 0, exptimes, np.nan)
        distance = np.abs(np.log(safe) - np.log(max(match_exptime, 1e-3)))
        distance = np.where(np.isfinite(distance), distance, np.inf)
        best = float(safe[np.argmin(distance)])
        chosen = [p for p, t in zip(paths, exptimes) if np.isclose(t, best)]
        exptimes = np.array([best] * len(chosen))

    # Keyed on the frames actually chosen rather than on the exposure time
    # asked for. Science frames are rarely all the same length, and keying
    # on the request would rebuild the same master for 299.5 s as for
    # 300 s, which is the whole cost this cache exists to avoid.
    key = (tuple(chosen), bool(transpose), method)
    if key in _CALIBRATION_CACHE:
        return _CALIBRATION_CACHE[key]

    master = _combine(chosen, method=method, transpose=transpose)
    exptime = float(np.median(exptimes[exptimes > 0])) if np.any(exptimes > 0) else 0.0

    # Handed out by reference to keep a 4096 x 4096 frame from being
    # copied per science frame, so it is made read-only: callers subtract
    # into their own array and nothing should be writing through this.
    master.setflags(write=False)

    if verbose and key not in _CALIBRATION_REPORTED:
        _CALIBRATION_REPORTED.add(key)
        label = want or "calibration"
        where = (os.path.basename(chosen[0]) if len(chosen) == 1
                 else f"{len(chosen)} frames from {os.path.basename(os.path.dirname(chosen[0]))}")
        print(f"  master {label}: {where}"
              + (f", {exptime:g} s" if exptime else "")
              + f", median {np.median(master):.1f} ADU")

    _CALIBRATION_CACHE[key] = (master, exptime)
    return master, exptime


def calibrate_image(image, path, transpose=None, force=None, verbose=True):
    """Subtract bias and dark from a frame that has just been read.

    The order is bias first, then dark. Where the master dark still
    carries its own bias, which is the usual case for a dark written
    straight off the camera, config.DARK_INCLUDES_BIAS makes the bias be
    removed from the dark before the dark current is scaled, so the
    pedestal is never subtracted twice.

    Scaling assumes dark current is linear in time, which it is to well
    within its own noise over the range of exposures used here. A scale
    far from 1 is reported, because a 1 s dark stretched to 900 s carries
    its read noise up with it.

    Parameters
    ----------
    image : ndarray
        Raw frame, 2D of shape (ny, nx).
    path : str
        Path the frame came from, used to read its exposure time.
    transpose : bool, optional
        Orientation the calibration frames should be read in. Default
        None, meaning config.TRANSPOSE.
    force : bool, optional
        True applies both corrections whatever config says, False applies
        neither. Default None, meaning follow config.APPLY_BIAS and
        config.APPLY_DARK.
    verbose : bool, optional
        Print what was applied, once per master. Default True.

    Returns
    -------
    image : ndarray
        The calibrated frame. The input array is not modified.

    Raises
    ------
    ValueError
        If a master frame is a different shape from the frame being
        calibrated, which normally means a different binning or a
        different TRANSPOSE.
    """
    do_bias = getattr(config, "APPLY_BIAS", False) if force is None else force
    do_dark = getattr(config, "APPLY_DARK", False) if force is None else force
    if not (do_bias or do_dark):
        return image

    bias_source = getattr(config, "MASTER_BIAS", None)
    dark_source = getattr(config, "MASTER_DARK", None)
    dark_carries_bias = getattr(config, "DARK_INCLUDES_BIAS", True)
    # The dark needs the bias too when it still carries its own pedestal,
    # so this is not just the APPLY_BIAS case.
    need_bias = do_bias or (do_dark and dark_carries_bias)

    if need_bias and not bias_source:
        raise ValueError(
            "config.MASTER_BIAS is empty. It is needed because "
            + ("config.APPLY_BIAS is set" if do_bias else
               "config.APPLY_DARK is set with DARK_INCLUDES_BIAS, so the bias has "
               "to be taken off the master dark before the dark current can be "
               "scaled. Set MASTER_BIAS, or clear DARK_INCLUDES_BIAS if the dark "
               "is already bias-subtracted") + ".")
    if do_dark and not dark_source:
        raise ValueError("config.APPLY_DARK is set but config.MASTER_DARK is empty")

    out = np.array(image, dtype=float, copy=True)

    def check(master, label):
        if master.shape != out.shape:
            raise ValueError(
                f"the master {label} is {master.shape} but "
                f"{os.path.basename(path)} is {out.shape}. Different binning, a "
                f"different region of the detector, or a TRANSPOSE that does not "
                f"match.")

    bias = None
    if need_bias:
        bias, _ = master_frame(bias_source, want="bias", transpose=transpose,
                               verbose=verbose)
        check(bias, "bias")

    if do_bias:
        out -= bias

    if do_dark:
        exptime = _frame_exptime(path)
        dark, dark_exptime = master_frame(dark_source, want="dark",
                                          match_exptime=exptime or None,
                                          transpose=transpose, verbose=verbose)
        check(dark, "dark")
        # what is left after the pedestal is the dark current itself
        current = dark - bias if dark_carries_bias else dark

        scale = 1.0
        if getattr(config, "SCALE_DARK_BY_EXPTIME", True):
            if exptime > 0 and dark_exptime > 0:
                scale = exptime / dark_exptime
            elif verbose:
                print(f"  WARNING: no exposure time for "
                      f"{os.path.basename(path)} or for the master dark, so the "
                      f"dark is subtracted unscaled.")
        tolerance = getattr(config, "DARK_EXPTIME_TOLERANCE", 4.0)
        if verbose and tolerance and not (1.0 / tolerance <= scale <= tolerance):
            key = ("scale", round(scale, 3), os.path.basename(path))
            if key not in _CALIBRATION_REPORTED:
                _CALIBRATION_REPORTED.add(key)
                print(f"  WARNING: the nearest master dark is {dark_exptime:g} s "
                      f"against a {exptime:g} s exposure, so it is being scaled by "
                      f"{scale:.1f}x. That scales its read noise too. Take darks "
                      f"nearer the science exposure time.")

        # Only ever the dark current, never the pedestal. Subtracting a
        # master dark whole would take the bias with it, and take it
        # scaled, which is wrong by (scale - 1) x bias. With APPLY_BIAS
        # off the pedestal is deliberately left in place; the two
        # corrections stay independent so either can be used alone.
        out -= scale * current

    return out


def read_header(path):
    """Read the primary header of a FITS file.

    Parameters
    ----------
    path : str
        Path to the FITS file.

    Returns
    -------
    header : fitsio.FITSHDR
        Header of the primary HDU, indexable by keyword.
    """
    with fitsio.FITS(path, "r") as f:
        return f[0].read_header()


def observation_time(path):
    """Return the start time of an exposure and where it came from.

    The DATE-OBS header is used when it is present and parsable, otherwise
    a timestamp in the filename. The two are different clocks, so the
    source is reported alongside the time.

    Parameters
    ----------
    path : str
        Path to the FITS file.

    Returns
    -------
    start : datetime or None
        Start of the exposure, None if neither the header nor the filename
        yielded a time.
    exptime : float
        Exposure time in seconds, 0.0 if EXPTIME is missing or unreadable.
    source : str
        "DATE-OBS", "DATE", "filename", or "none" if no time was found.
    """
    exptime = 0.0
    try:
        header = read_header(path)
        # HERCULES writes DATE rather than DATE-OBS
        raw = header.get("DATE-OBS") or header.get("DATE")
        try:
            exptime = float(header.get("EXPTIME") or 0.0)
        except (TypeError, ValueError):
            exptime = 0.0
        if raw:
            text = str(raw).strip().strip("'").replace("Z", "")
            for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
                try:
                    return (datetime.strptime(text, fmt), exptime,
                            "DATE-OBS" if header.get("DATE-OBS") else "DATE")
                except ValueError:
                    continue
    except Exception:
        pass

    match = _FILENAME_TIME.search(os.path.basename(path))
    if match:
        date, hh, mm, ss = match.groups()
        return (datetime.strptime(f"{date}T{hh}:{mm}:{ss}", "%Y-%m-%dT%H:%M:%S"),
                exptime, "filename")
    return None, exptime, "none"


def mid_exposure_time(path):
    """Return the mid point of an exposure and where its time came from.

    The mid point is the moment an arc should be matched against. Half an
    exposure is 150 s on a 300 s frame, which matters when arcs are minutes
    apart.

    Parameters
    ----------
    path : str
        Path to the FITS file.

    Returns
    -------
    middle : datetime or None
        Start time plus half the exposure time, None if no start time could
        be read.
    source : str
        "DATE-OBS", "DATE", "filename", or "none".
    """
    start, exptime, source = observation_time(path)
    if start is None:
        return None, source
    return start + timedelta(seconds=exptime / 2.0), source


def list_frames(directory, pattern=None, recursive=False):
    """List the frames in a directory, sorted by path.

    Parameters
    ----------
    directory : str
        Directory to search.
    pattern : str, optional
        Filename glob pattern. Default None, meaning config.FRAME_PATTERN.
    recursive : bool, optional
        When True, search subdirectories as well. Default False.

    Returns
    -------
    paths : list of str
        Matching file paths in sorted order, empty if nothing matches.
    """
    pattern = pattern or config.FRAME_PATTERN
    if recursive:
        found = glob.glob(os.path.join(directory, "**", pattern), recursive=True)
    else:
        found = glob.glob(os.path.join(directory, pattern))
    return sorted(found)


def list_arcs(directory, pattern=None, verbose=True):
    """List arc frames with their mid-exposure times, earliest first.

    Frames whose time cannot be read are dropped and reported, since an arc
    that cannot be placed in time cannot be chosen by time. A warning is
    printed when the times came from more than one source, because DATE-OBS
    and the filename are not comparable.

    Parameters
    ----------
    directory : str
        Directory to search.
    pattern : str, optional
        Filename glob pattern. Default None, meaning config.ARC_PATTERN.
    verbose : bool, optional
        When True, print the number of arcs found, the span they cover in
        minutes, and the time sources used. Default True.

    Returns
    -------
    arcs : list of tuple
        (path, mid-exposure datetime) pairs sorted by time, empty if no
        usable arc was found.
    """
    arcs = []
    sources = set()
    for path in list_frames(directory, pattern or config.ARC_PATTERN):
        when, source = mid_exposure_time(path)
        if when is None:
            print(f"  skipping {os.path.basename(path)}: no usable observation time")
            continue
        arcs.append((path, when))
        sources.add(source)
    arcs.sort(key=lambda item: item[1])

    if verbose and arcs:
        span = (arcs[-1][1] - arcs[0][1]).total_seconds() / 60.0
        print(f"list_arcs: {len(arcs)} arcs in {directory}")
        print(f"  {arcs[0][1]:%Y-%m-%d %H:%M} to {arcs[-1][1]:%H:%M} "
              f"({span:.0f} minutes), times from {'/'.join(sorted(sources))}")
    if len(sources) > 1:
        print("  WARNING: arc times came from more than one source. DATE-OBS and the "
              "filename are not the same clock, so these are not comparable.")
    return arcs


def nearest_arc(arcs, when):
    """Return the arc closest in time to a given moment.

    Parameters
    ----------
    arcs : list of tuple
        (path, datetime) pairs, as returned by list_arcs.
    when : datetime
        Moment to compare against, usually a science mid-exposure time.

    Returns
    -------
    path : str or None
        Path of the closest arc, None if arcs is empty.
    gap : float or None
        Absolute separation in time, in minutes, None if arcs is empty.
    """
    if not arcs:
        return None, None
    gaps = [abs((t - when).total_seconds()) / 60.0 for _, t in arcs]
    k = int(np.argmin(gaps))
    return arcs[k][0], gaps[k]


def bracketing_arcs(arcs, when):
    """Return the last arc at or before a moment and the first after it.

    Two bracketing arcs let the shift be interpolated to the exposure
    rather than assumed constant across the gap.

    Parameters
    ----------
    arcs : list of tuple
        (path, datetime) pairs, as returned by list_arcs.
    when : datetime
        Moment to bracket, usually a science mid-exposure time.

    Returns
    -------
    before : tuple or None
        (path, datetime) of the latest arc at or before when, None if there
        is no such arc.
    after : tuple or None
        (path, datetime) of the earliest arc after when, None if there is
        no such arc.
    """
    before = [a for a in arcs if a[1] <= when]
    after = [a for a in arcs if a[1] > when]
    return (before[-1] if before else None), (after[0] if after else None)


def describe_arc_choice(science_path, arcs, interpolate=True, max_gap_minutes=90.0):
    """Choose the arc or arcs a science frame should use and report them.

    The choice is printed, along with warnings when the frame's time source
    differs from the arcs', when the nearest arc is further away than
    max_gap_minutes, and when the exposure is not bracketed by arcs.

    Parameters
    ----------
    science_path : str
        Path to the science FITS frame.
    arcs : list of tuple
        (path, datetime) pairs, as returned by list_arcs.
    interpolate : bool, optional
        When True, return both bracketing arcs where they exist so the
        shift can be interpolated. Default True.
    max_gap_minutes : float, optional
        Separation beyond which a single nearest arc raises a printed
        warning, in minutes. Default 90.0.

    Returns
    -------
    chosen : list of tuple
        (path, datetime) pairs. Two entries mean the shift is interpolated
        between them, one entry means it is taken from that arc, and an
        empty list means no arc was available.
    when : datetime or None
        Mid-exposure time of the science frame, None if no time could be
        read, in which case the first arc is returned if there is one.
    """
    when, source = mid_exposure_time(science_path)
    name = os.path.basename(science_path)
    if when is None:
        print(f"  {name}: no observation time, falling back to the first arc")
        return ([arcs[0]] if arcs else []), None

    arc_sources = {observation_time(a)[2] for a, _ in arcs}
    if arc_sources and source not in arc_sources:
        print(f"  WARNING: this frame's time came from {source} while the arcs' came "
              f"from {'/'.join(sorted(arc_sources))}. Those are different clocks: "
              f"DATE-OBS is UTC here and the filename is local, so the gaps below "
              f"are not meaningful.")

    before, after = bracketing_arcs(arcs, when)
    if interpolate and before and after:
        gap_before = (when - before[1]).total_seconds() / 60.0
        gap_after = (after[1] - when).total_seconds() / 60.0
        print(f"  {name} (mid-exposure {when:%H:%M:%S}, from {source}) is bracketed: "
              f"{gap_before:.0f} min after {os.path.basename(before[0])}, "
              f"{gap_after:.0f} min before {os.path.basename(after[0])} "
              f"-- interpolating between them")
        return [before, after], when

    path, gap = nearest_arc(arcs, when)
    if path is None:
        return [], when
    arc_time = dict(arcs)[path]
    print(f"  {name} (mid-exposure {when:%H:%M:%S}, from {source}): nearest arc is "
          f"{os.path.basename(path)}, {gap:.0f} min "
          f"{'before' if arc_time <= when else 'after'}")
    if gap > max_gap_minutes:
        print(f"    WARNING: {gap:.0f} minutes is a long way from the exposure. Any "
              f"flexure in between is unmeasured, and with arcs on one side only it "
              f"cannot be separated from the target's velocity either.")
    if not (before and after):
        print("    note: this exposure is not bracketed by arcs. Taking one either side "
              "is what lets drift be measured independently of the target's motion.")
    return [(path, arc_time)], when