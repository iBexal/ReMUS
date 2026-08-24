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

# Detector orientation. The pipeline needs rows to be the dispersion
# direction and columns the cross dispersion direction. Set True where the
# raw frames come off the detector the other way round, and every frame is
# transposed as it is read. HERCULES needs True, ReMUS needs False. Check
# it with one_off/check_tracing.py: if the orders run across the image
# rather than up it, this is wrong.
TRANSPOSE = False

# Filename patterns. Frames are not always called .fits: HERCULES writes
# .fit, so "*.fit*" matches both.
FRAME_PATTERN = "*.fits"
ARC_PATTERN = "*ThAr*.fits"

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

# Counts at or above which an arc line is discarded as saturated. A
# saturated line is flat-topped and asymmetric and its centroid is
# meaningless, so it is thrown away rather than fitted.
#
# In the same units as the frames the pipeline reads, which is raw ADU
# with APPLY_BIAS off and ADU above bias with it on. On this detector the
# pedestal is about 1000 ADU, so lower this by that much when bias
# subtraction is on if you want the limit to mean the same thing.
ARC_SATURATION = 55000.0

# Reference lines must be strong, and must dominate their own resolution
# element. The atlas carries about four lines per Angstrom in the blue
# against a resolution element of a tenth of one.
ATLAS_AMPLITUDE_MIN = 150.0
ATLAS_DOMINANCE = 4.0

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
MAX_ARC_GAP_MINUTES = 120.0     # warn beyond this; flexure has had time to act

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
# Bias and dark
# ----------------------------------------------------------------------
# Applied in frames.read_image, so every frame that enters the pipeline
# is corrected the same way and nothing downstream has to know. Both are
# off by default: a wavelength solution is blind to an additive pedestal,
# because line centroids are measured above a median-filtered continuum,
# so leaving the bias in costs a wavelength axis nothing.
#
# Turn APPLY_BIAS on when flat fielding. A flat field divides, and
# (S + bias) / (F + bias) is not S / F, so an un-debiased flat field is
# wrong by an amount that depends on how bright the pixel is. On this
# detector the pedestal is around 1000 ADU against flats peaking near
# 49000, so the error is not subtle.
#
# APPLY_DARK matters much less here. Dark current measured from the 2026
# masters is about 0.15 ADU/s at 0 C, so a 300 s exposure collects tens
# of ADU. Worth having for long exposures and for hot pixels; not worth
# scaling a 1 s dark up to 900 s to get, since that scales its read noise
# with it.
APPLY_BIAS = True
APPLY_DARK = True
FLAT_FIELD = True

# A master FITS file, a directory to build one from, or a list of paths.
# A directory is filtered on IMAGETYP where the frames carry one, so a
# folder holding both a master bias and several master darks can be given
# to both settings. Frames are grouped by exposure time and the group
# nearest the frame being calibrated is used.
MASTER_BIAS = os.path.join(SPECTRA_ROOT, "2026-ASTR3010", "Master_Bias_Darks_2026")
MASTER_DARK = os.path.join(SPECTRA_ROOT, "2026-ASTR3010", "Master_Bias_Darks_2026")

# True where the master dark still carries its own bias pedestal, which
# is what a dark written straight off the camera looks like. The 2026
# masters do: MasterDark_1s has a median of 1002 ADU against a bias of
# 1004. With this set the bias is removed from the dark before the dark
# current is scaled, so the pedestal is never subtracted twice or scaled.
DARK_INCLUDES_BIAS = True

# Scale the dark current by the ratio of exposure times. Dark current is
# linear in time to well within its own noise over this range.
SCALE_DARK_BY_EXPTIME = True
DARK_EXPTIME_TOLERANCE = 4.0    # warn when the scale factor leaves 1/4 .. 4

# How several calibration frames become one. "median" is safe; "mean" is
# a 5-sigma-clipped mean, which has lower noise once there are more than
# about ten frames.
CALIBRATION_COMBINE = "median"

# ----------------------------------------------------------------------
# Flat fielding
# ----------------------------------------------------------------------
# The white lamp has its own steep continuum, so the flat is split into a
# smooth part (the lamp's colour, the blaze, the fibre throughput) and a
# pixel-to-pixel part, and only the second is divided out. See
# flat_field.py. The lamp SED therefore never reaches the science
# spectrum, and the blaze is left in the flux and saved alongside it so
# it can be removed later if wanted.


# Everything varying more slowly than this many pixels is called blaze
# and kept out of the correction. It has to be much longer than a line
# (about 8 px here) and much shorter than the blaze (thousands), so
# anything from about 51 to 301 gives the same answer.
FLAT_SMOOTH_WINDOW = 101

# Leave the ends of an order alone, where the blaze has fallen away and
# the measured response is mostly noise.
FLAT_MIN_RELATIVE = 0.15

# A response further from 1 than this is a defect, not a sensitivity.
FLAT_MAX_CORRECTION = 1.5

# Flat field the arcs as well as the science frames. A gradient in pixel
# response across a line profile pulls its centroid, so this is the part
# that touches the wavelength solution.
FLAT_FIELD_ARCS = True

# Counts at or above which the coadded flat is called saturated. Used
# only to warn: a saturated flat flattens the order profile and both the
# tracing and the flat field take that as real. Same units as
# ARC_SATURATION, so it means ADU above bias once APPLY_BIAS is on.
FLAT_SATURATION = 55000.0

# ----------------------------------------------------------------------
# Wavelength solution: how the fit is done
# ----------------------------------------------------------------------
# These change the numbers a rebuild produces. Each is separately
# switchable so a master can be built both ways and the quality report
# compared, and each defaults to the better behaviour.
#
# They cover the judgement calls only. Outright bug fixes are not
# switchable and are applied always: a NaN sample no longer discards its
# whole order in detect_arc_lines, an order with no order number no
# longer raises in match_lines, and overlap_agreement no longer
# interpolates on a reversed axis. Turning every switch here off gets
# back the old fitting choices, not the old bugs.

# Fit the line's local background as a slope rather than a constant, and
# fit it to the raw spectrum instead of to the median-filter residual.
# A 201 px running median through a line-dense ThAr spectrum tracks the
# local line density, not the continuum, so what is left under a line is
# a tilted pedestal. A tilt under a symmetric Gaussian moves its fitted
# centre, and the amount varies with where the line sits in the blaze, so
# it does not average out over many lines.
ARC_LINE_LINEAR_BACKGROUND = True

# Set the detection threshold from the noise near each line rather than
# from one number for the whole order. Flux across an echelle order
# varies by an order of magnitude, so a single threshold is too strict at
# the ends of the order and too loose at the blaze peak. The ends are
# where pixel coverage and order overlap are decided.
ARC_LOCAL_NOISE = True

# Clip on residuals divided by their own uncertainty rather than on
# residuals in m*lambda. m runs 137 down to 52 here, so a fixed cut in
# m*lambda is a factor of 2.6 tighter on the bluest order than the
# reddest, and throws away good blue lines while keeping bad red ones.
WEIGHTED_CLIPPING = True

# Fit the physical surface and its Chebyshev correction together instead
# of fitting the correction to the leftovers of the physical fit. Fitting
# them in sequence is one step of backfitting, which does not reach the
# least-squares optimum unless the two bases are orthogonal, and they are
# not. It also lets the camera focal length be chosen against residuals
# the correction has not yet absorbed.
JOINT_CORRECTION_FIT = True

# Apply the drift across the order range that measure_arc_shift already
# measures, as a per-order shift, instead of only reporting it.
APPLY_ARC_TILT = True

# ----------------------------------------------------------------------
# Cosmic rays
# ----------------------------------------------------------------------

CLEAN_COSMIC_RAYS = True
COSMIC_RAY_MAX_WIDTH = 2      # pixels; anything wider is left alone
COSMIC_RAY_SIGMA = 8.0        # how far above the local scatter counts as a spike