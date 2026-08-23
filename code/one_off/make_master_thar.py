"""Build a master wavelength solution. Run once per instrument configuration.

Inputs: one night's white-light flats to trace the orders, one arc frame,
and one science frame in which to click the Na D doublet. The result is
written to config.MASTER_PATH, where every later reduction finds it, with a
plain-text summary beside it and a dated copy in the archive folder.

config.py must already hold ANCHORS, NAD_TRACE and DIRECTION for the
instrument. For a spectrograph the pipeline has not seen, get those from
find_anchors.py in this folder first.

Every other night and every science frame goes through ReMUS.py, which only
registers this result onto new data.
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

# Which arc to build from. Any clean one will do. It becomes the master's
# reference, and every later arc is registered against it. None selects the
# arc nearest in time to SCIENCE_FILE.
ARC_FILE = None

# ----------------------------------------------------------------------
# The two clicks
# ----------------------------------------------------------------------
# Leave as None the first time and click the Na D doublet when the plots
# open: D2 (5889.95 A) first, then D1 (5895.92 A). The run prints the
# refined pixels to paste back here so the clicking is only needed once.
NAD_PIXEL_GUESSES = None            # this data: [2559.0, 2868.9]

# Optional, and take no part in the fit. They check the finished solution
# against lines in two further orders and measure how far the science frame
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
