"""Reduce one night of science frames against the saved master solution.

Nothing is fitted here. The master built by one_off/make_master_thar.py is
registered onto this night's traces and arcs, each science frame is given
the arc nearest it in time (interpolated between two where it is bracketed),
the result is checked against the ThAr atlas, and calibrated spectra are
written to config.REDUCED_ROOT as one .npz per frame.

Set NIGHT and TARGET below and run it.
"""

import os

import config
import frames
from reduce_spectra import reduce_science

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
