"""Settings for the instrument and the pipeline.

Every module imports this, so a value defined here is defined once. Settings
that change from run to run, such as which night and which target, live in
the run files instead: ReMUS.py and one_off/make_master_thar.py.
"""

import os

# ----------------------------------------------------------------------
# Where everything lives
# ----------------------------------------------------------------------
# The root of the whole project. Everything below is derived from it, so
# moving or renaming the project only means changing this line.
PROJECT_ROOT = "/Users/ethan/Desktop/University/ReMUS"

SPECTRA_ROOT = os.path.join(PROJECT_ROOT, "spectra")
CODE_DIR = os.path.dirname(os.path.abspath(__file__))

# The master solution does NOT live with a night's data. It describes the
# instrument, not an observing run, and it has to be reachable whatever is
# being reduced, including a night whose folder did not exist when the
# master was built. So it sits in one fixed place at the top of the project.
CALIBRATION_DIR = os.path.join(PROJECT_ROOT, "calibration")
MASTER_PATH = os.path.join(CALIBRATION_DIR, "master_wavelength_solution.pkl")
MASTER_SUMMARY_PATH = os.path.join(CALIBRATION_DIR, "master_wavelength_solution.txt")
# Every master that passes is also copied here under the night it came from
# and the time it was built, so a good one is never lost to the next rebuild.
MASTER_ARCHIVE_DIR = os.path.join(CALIBRATION_DIR, "archive")

ATLAS_PATH = os.path.join(CODE_DIR, "thar_linelist", "ThAr_lines.dat")

# Reduced spectra are written here, under a folder per night and target,
# unless a run file overrides it.
REDUCED_ROOT = os.path.join(PROJECT_ROOT, "reduced")

# ----------------------------------------------------------------------
# The instrument
# ----------------------------------------------------------------------
BLAZE_ANGLE_DEG = 64.35
GROOVE_DENSITY_MM = 31.6

# m_i = m0 + DIRECTION * trace_index. -1 means trace index 0 is the bluest
# (highest order number), which the anchors below imply and the white-light
# brightness envelope confirms.
DIRECTION = -1

# The orders whose identification is trusted, as (trace index, wavelength).
# These are the only things fixing the absolute order numbers when a master
# is built. Both must imply the same m0 or the build stops.
ANCHORS = [(50, 6562.8),    # H-alpha
           (20, 4861.3)]    # H-beta

# The dispersion seed: two clicks on the Na D doublet in this trace's
# science spectrum. One doublet sets the dispersion for every order, because
# m*lambda is a shared function of pixel.
NAD_TRACE = 40
NAD_LINES = [(5889.95, "Na D2"), (5895.92, "Na D1")]

# Instrumental profile, in pixels of sigma (~7.8 px FWHM here). Drives the
# line-detection width cuts and how isolated an atlas line has to be.
EXPECTED_LINE_SIGMA_PIXELS = 3.3

# Reference lines must be strong, and must dominate their own resolution
# element. The atlas carries about four lines per Angstrom in the blue
# against a resolution element of a tenth of one.
ATLAS_AMPLITUDE_MIN = 200.0
ATLAS_DOMINANCE = 5.0

# ----------------------------------------------------------------------
# Choosing an arc
# ----------------------------------------------------------------------
# When reducing, the arc is chosen by time rather than by hand. Times come
# from the FITS DATE-OBS header, offset to mid-exposure.
#
# If arcs exist both before and after a science frame, the shift is measured
# from each and interpolated to the exposure's mid-time, which is what
# bracketing arcs are for. With arcs on one side only, the nearest is used
# and its shift applied as measured.
INTERPOLATE_BETWEEN_ARCS = True
MAX_ARC_GAP_MINUTES = 90.0     # warn beyond this; flexure has had time to act

# ----------------------------------------------------------------------
# Quality gates
# ----------------------------------------------------------------------
# A master that fails any of these is reported and NOT saved. Tighten them
# once you know what the instrument actually delivers.
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

# The gates applied when a saved master is registered onto a new arc.
APPLY_QUALITY = dict(
    max_rms_ma=15.0,
    max_overlap_ms=600.0,
)

# ----------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------
SCIENCE_EXTRACT_NSIGMA = 3.0
ARC_EXTRACT_NSIGMA = 2.5

# ----------------------------------------------------------------------
# Cosmic rays
# ----------------------------------------------------------------------

CLEAN_COSMIC_RAYS = True
COSMIC_RAY_MAX_WIDTH = 2      # pixels; anything wider is left alone
COSMIC_RAY_SIGMA = 8.0        # how far above the local scatter counts as a spike
