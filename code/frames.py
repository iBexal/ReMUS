"""Locate FITS frames, read them, and read the times they were taken.

Times let an arc be chosen for a science exposure by comparison of
timestamps rather than by file order. Every time is reported with the
source it came from, since the header and the filename are different
clocks.

Main entry points: read_image, read_header, mid_exposure_time, list_frames,
list_arcs, nearest_arc, bracketing_arcs, describe_arc_choice.
"""

import glob
import os
import re
from datetime import datetime, timedelta

import fitsio
import numpy as np

# Filenames carry a time as well, for example
# ..._2025-03-12T22-57-07_008.fits, but it is not the same clock as the
# header: on this data DATE-OBS is UTC and the filename is local, eleven
# hours apart. The header is used where available and the filename only as a
# last resort, with the source reported so a mixture can be spotted.
_FILENAME_TIME = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})")


def read_image(path):
    """Read the primary image of a FITS file.

    Parameters
    ----------
    path : str
        Path to the FITS file.

    Returns
    -------
    image : ndarray
        Data of the primary HDU, 2D of shape (ny, nx) for detector frames.
    """
    with fitsio.FITS(path, "r") as f:
        return f[0].read()


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
        "DATE-OBS", "filename", or "none" if no time was found.
    """
    exptime = 0.0
    try:
        header = read_header(path)
        raw = header.get("DATE-OBS")
        try:
            exptime = float(header.get("EXPTIME") or 0.0)
        except (TypeError, ValueError):
            exptime = 0.0
        if raw:
            text = str(raw).strip().strip("'").replace("Z", "")
            for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
                try:
                    return datetime.strptime(text, fmt), exptime, "DATE-OBS"
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
        "DATE-OBS", "filename", or "none".
    """
    start, exptime, source = observation_time(path)
    if start is None:
        return None, source
    return start + timedelta(seconds=exptime / 2.0), source


def list_frames(directory, pattern="*.fits", recursive=False):
    """List the frames in a directory, sorted by path.

    Parameters
    ----------
    directory : str
        Directory to search.
    pattern : str, optional
        Filename glob pattern. Default "*.fits".
    recursive : bool, optional
        When True, search subdirectories as well. Default False.

    Returns
    -------
    paths : list of str
        Matching file paths in sorted order, empty if nothing matches.
    """
    if recursive:
        found = glob.glob(os.path.join(directory, "**", pattern), recursive=True)
    else:
        found = glob.glob(os.path.join(directory, pattern))
    return sorted(found)


def list_arcs(directory, pattern="*ThAr*.fits", verbose=True):
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
        Filename glob pattern. Default "*ThAr*.fits".
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
    for path in list_frames(directory, pattern):
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
