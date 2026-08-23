"""
one_off/make_master_thar.py

Run this to build a master wavelength solution. Once per instrument
configuration -- not once per night.

It needs one night's white-light flats to trace the orders, one arc, and one
science frame to click the Na D doublet in. The result goes to
config.MASTER_PATH, where every later reduction finds it, and is archived
alongside so a good master is never lost to the next rebuild.

Everything else -- every other night, every science frame -- goes through
ReMUS.py, which only registers this result onto new data.

Before this can run at all, config.py needs ANCHORS, NAD_TRACE and DIRECTION
for the instrument. For a spectrograph the pipeline has not seen, get those
from find_anchors.py in this folder first.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import frames
from one_off.build_master_thar import build_master

# ----------------------------------------------------------------------
# Which data to build from
# ----------------------------------------------------------------------
NIGHT = os.path.join(config.SPECTRA_ROOT, "2025-03-12")

WHITE_LOC = os.path.join(NIGHT, "White")
ARC_LOC = os.path.join(NIGHT, "Flat")
SCIENCE_FILE = os.path.join(NIGHT, "Light", "Arcturus",
                            "Arcturus_Light_300_secs_2025-03-13T00-56-21_005.fits")

# Which arc to build from. Any clean one will do -- this is the master's own
# reference, and every later arc gets registered against it. None takes the
# one nearest in time to SCIENCE_FILE.
ARC_FILE = None

# ----------------------------------------------------------------------
# The two clicks
# ----------------------------------------------------------------------
# Leave as None the first time and click the Na D doublet when the plots
# open -- D2 (5889.95 A) first, then D1 (5895.92 A). The run prints a hint
# to paste back here so you never click again on this data.
NAD_PIXEL_GUESSES = None            # this data: [2559.0, 2868.9]

# Optional. These take no part in the fit; they check the finished solution
# against lines in two further orders, and measure how far the science frame
# sits from the arc.
HALPHA_PIXEL_GUESS = None           # this data: 2367.6, in trace config.ANCHORS[0]
HBETA_PIXEL_GUESS = None            # this data: 1222.0, in trace config.ANCHORS[1]


arc_file = ARC_FILE
if arc_file is None:
    when, _ = frames.mid_exposure_time(SCIENCE_FILE)
    arc_file, gap = frames.nearest_arc(frames.list_arcs(ARC_LOC), when)
    print(f"Arc chosen by time: {os.path.basename(arc_file)}, {gap:.0f} min from "
          f"the science frame\n")

build_master(
    white_loc=WHITE_LOC,
    arc_file=arc_file,
    science_file=SCIENCE_FILE,
    nad_pixel_guesses=NAD_PIXEL_GUESSES,
    halpha_pixel_guess=HALPHA_PIXEL_GUESS,
    hbeta_pixel_guess=HBETA_PIXEL_GUESS,
)
