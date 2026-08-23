"""Identify the orders holding H-alpha, H-beta and Na D, and solve for m.

Run once per spectrograph. It produces the three config.py settings that
cannot be derived from an arc, because an arc has no line of known identity:
DIRECTION, ANCHORS and NAD_TRACE.

Inputs: white-light flats to trace with, and one science frame of a star
showing H-alpha, H-beta and Na D. A bright, moderately warm star is easiest,
since the Balmer lines are broad. Arcturus works.

Procedure:

  1. a map of all orders opens; the two widest dark features are the Balmer
     lines. Note their trace indices and close the map.
  2. a browser opens for each line. Arrow keys change order, a click accepts.
     The click does not need to be precise, it is refined by a Gaussian fit.
  3. the absolute order numbers are solved and printed ready to paste into
     config.py.

The solve does not require the grating constant. Setting BLAZE_ANGLE_DEG and
GROOVE_DENSITY_MM below adds the only independent check that no order was
missed between the two Balmer lines.

Set ALPHA, BETA and NAD to re-run the solve without clicking.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt

import one_off.anchors as anchors
import config
import frames
import wavelength_solution as ws
from order_tracing import trace_orders

# ----------------------------------------------------------------------
# What to look at
# ----------------------------------------------------------------------
NIGHT = os.path.join(config.SPECTRA_ROOT, "2025-03-12")
WHITE_LOC = os.path.join(NIGHT, "White")
SCIENCE_FILE = os.path.join(NIGHT, "Light", "Arcturus",
                            "Arcturus_Light_300_secs_2025-03-13T00-56-21_005.fits")

# ----------------------------------------------------------------------
# The cross-check
# ----------------------------------------------------------------------
# Grating constant K = 2 d sin(blaze), in Angstrom. Set both values to None
# to skip the cross-check. The solve still works, but nothing then confirms
# the integer it lands on.
# BLAZE_ANGLE_DEG = config.BLAZE_ANGLE_DEG
# GROOVE_DENSITY_MM = config.GROOVE_DENSITY_MM
BLAZE_ANGLE_DEG = None
GROOVE_DENSITY_MM = None

# A rough guess at the order number of trace 0, only used to open each
# browser near the right order instead of in the middle. Wrong is harmless.
M0_GUESS = None
DIRECTION_GUESS = -1

# ----------------------------------------------------------------------
# Already know where the lines are?
# ----------------------------------------------------------------------
# Fill these in to skip the clicking and re-run the solve directly, for
# example after the tracer picks up a different number of orders. Format:
#   ALPHA = (trace_index, pixel)
#   BETA  = (trace_index, pixel)
#   NAD   = (trace_index, pixel_of_D2_5889.95, pixel_of_D1_5895.92)
ALPHA = None        # this instrument: (50, 2367.6)
BETA = None         # this instrument: (20, 1222.0)
NAD = None          # this instrument: (40, 2559.0, 2868.9)


K = None
if BLAZE_ANGLE_DEG is not None and GROOVE_DENSITY_MM is not None:
    K = ws.compute_grating_K(BLAZE_ANGLE_DEG, GROOVE_DENSITY_MM)
    print(f"Grating constant K = 2 d sin(blaze) = {K:.0f} A "
          f"(cross-check only, not an input to the answer)\n")

orders, white = trace_orders(WHITE_LOC)
n_pixels = white.shape[0]

print("\nChecking the trace list for a missing order:")
ws.check_trace_spacing(orders)
print("  A missed order between the two Balmer lines breaks the solve, so "
      "read the above before trusting the result.\n")

print("Extracting the science spectrum along every trace ...")
science = frames.read_image(SCIENCE_FILE)
for order in orders:
    order.science_spectrum = order.extract_weighted(
        science, n_sigma=config.SCIENCE_EXTRACT_NSIGMA)

if ALPHA and BETA and NAD:
    print("\nUsing the positions set in this file, no clicking:\n")
    anchors.solve_order_numbers(ALPHA, BETA, NAD, n_pixels, K=K)
else:
    anchors.find_anchor_lines(orders, n_pixels, K=K, m0_guess=M0_GUESS,
                              direction_guess=DIRECTION_GUESS)

plt.close("all")
