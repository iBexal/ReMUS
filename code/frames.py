"""
frames.py

Finding FITS frames, reading them, and knowing when they were taken.

Nothing here knows about wavelengths. It exists so that choosing an arc for
a science exposure is a matter of comparing timestamps rather than counting
files in a directory and hoping the order is meaningful.
"""

import glob
import os
import re
from datetime import datetime, timedelta

import fitsio
import numpy as np

# Filenames carry a time too -- ..._2025-03-12T22-57-07_008.fits -- but it is
# not the same clock as the header. On this data DATE-OBS is UTC while the
# filename is local, eleven hours apart. Mixing the two silently would put an
# arc most of a day away from its science frame, so the header is used for
# everything and the filename only as a last resort, with the source
# reported so a mixture can be spotted.
_FILENAME_TIME = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})")


def read_image(path):
    """Read the primary image of a FITS file."""
    with fitsio.FITS(path, "r") as f:
        return f[0].read()


def read_header(path):
    with fitsio.FITS(path, "r") as f:
        return f[0].read_header()


def observation_time(path):
    """(start time, exposure seconds, where the time came from).

    Returns (None, ...) if no time could be found at all.
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
    """The middle of the exposure, which is the moment an arc should match.

    (start time, source). Half the exposure is 150 s on a 300 s frame, which
    is not nothing when arcs are minutes apart.
    """
    start, exptime, source = observation_time(path)
    if start is None:
        return None, source
    return start + timedelta(seconds=exptime / 2.0), source


def list_frames(directory, pattern="*.fits", recursive=False):
    """Every frame in a directory, sorted by name."""
    if recursive:
        found = glob.glob(os.path.join(directory, "**", pattern), recursive=True)
    else:
        found = glob.glob(os.path.join(directory, pattern))
    return sorted(found)


def list_arcs(directory, pattern="*ThAr*.fits", verbose=True):
    """Arc frames with their mid-exposure times, earliest first.

    Frames whose time cannot be read are dropped -- an arc that cannot be
    placed in time cannot be chosen by time, and silently treating it as if
    it were at some default moment is worse than not using it.
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
    """The arc closest in time, and the gap in minutes."""
    if not arcs:
        return None, None
    gaps = [abs((t - when).total_seconds()) / 60.0 for _, t in arcs]
    k = int(np.argmin(gaps))
    return arcs[k][0], gaps[k]


def bracketing_arcs(arcs, when):
    """The last arc before `when` and the first after it.

    Either may be None. Two of them let the shift be interpolated to the
    exposure rather than assumed constant across the gap, which is the whole
    reason for bracketing a science frame with arcs.
    """
    before = [a for a in arcs if a[1] <= when]
    after = [a for a in arcs if a[1] > when]
    return (before[-1] if before else None), (after[0] if after else None)


def describe_arc_choice(science_path, arcs, interpolate=True, max_gap_minutes=90.0):
    """Work out which arc or arcs a science frame should use, and say so.

    Returns (list of (path, time), science mid-time). One entry means the
    shift is taken from that arc; two means it is interpolated between them.
    """
    when, source = mid_exposure_time(science_path)
    name = os.path.basename(science_path)
    if when is None:
        print(f"  {name}: no observation time, falling back to the first arc")
        return ([arcs[0]] if arcs else []), None

    arc_sources = {observation_time(a)[2] for a, _ in arcs}
    if arc_sources and source not in arc_sources:
        print(f"  WARNING: this frame's time came from {source} while the arcs' came "
              f"from {'/'.join(sorted(arc_sources))}. Those are different clocks -- "
              f"DATE-OBS is UTC here and the filename is local -- so the gaps below "
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
