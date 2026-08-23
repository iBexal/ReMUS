"""
one_off/find_anchors.py

Run this once for a spectrograph the pipeline has never seen, to work out
which traced order holds H-alpha, H-beta and the Na D doublet, and what the
absolute echelle order numbers are.

It produces the three settings in config.py that nothing else can derive:
DIRECTION, ANCHORS and NAD_TRACE. The arc cannot give them -- a ThAr frame
has no line whose identity you know in advance -- so they come from a star,
once, by eye.

What you need: white-light flats to trace with, and one science frame of a
star showing H-alpha, H-beta and Na D. A bright, fairly warm star is easiest
(the Balmer lines are broad and unmistakable); Arcturus works.

How it goes:

  1. a map of every order at once opens. The two broad dark smudges are the
     Balmer lines -- note their trace indices and close it;
  2. a browser opens for each line in turn. Arrow keys change order, a click
     accepts. You do not need to click precisely;
  3. the absolute order numbers are solved and printed ready to paste.

The solve needs no grating constant. Set K below anyway if you have it --
it is the only independent check that no order was missed between the two
Balmer lines, which is the one error the arithmetic cannot see.
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
# Grating constant K = 2 d sin(blaze), in Angstrom. Leave the two values
# below as None to skip it -- the solve still works, it just has nothing
# independent confirming the integer it lands on.
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
# Fill these in to skip the clicking entirely and just re-run the solve --
# useful for checking a change, or for re-deriving after the tracer picks up
# a different number of orders. Format:
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
print("  A missed order BETWEEN the two Balmer lines is the one thing that "
      "breaks this, so read the above before trusting the result.\n")

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
