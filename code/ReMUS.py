"""
ReMUS.py

Run this to reduce a night. It takes the master wavelength solution built by
make_master_thar.py, registers it onto this night's traces and arcs, and
writes wavelength-calibrated spectra.

No master is built here and nothing is fitted from scratch. Each science
frame is given the arc nearest it in time -- interpolated between two if it
is bracketed -- and the result is checked against the line list before
anything is written.
"""

import os

import config
import frames
from apply_master_thar import reduce_science

# ----------------------------------------------------------------------
# Which night, which target
# ----------------------------------------------------------------------
NIGHT = os.path.join(config.SPECTRA_ROOT, "2025-03-12")
TARGET = "Arcturus"

WHITE_LOC = os.path.join(NIGHT, "White")
ARC_LOC = os.path.join(NIGHT, "Flat")
SCIENCE_LOC = os.path.join(NIGHT, "Light", TARGET)

# Every frame in SCIENCE_LOC, or list them explicitly to reduce a subset.
SCIENCE_FILES = frames.list_frames(SCIENCE_LOC, "*.fits")

OUT_DIR = os.path.join(config.REDUCED_ROOT, os.path.basename(NIGHT), TARGET)


reduce_science(
    white_loc=WHITE_LOC,
    arc_loc=ARC_LOC,
    science_files=SCIENCE_FILES,
    out_dir=OUT_DIR,
)
