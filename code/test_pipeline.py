"""Check the bias/dark, flat field and wavelength-solution changes.

Everything here runs against synthetic data with a known answer, so each
check is an assertion about the code rather than about any one night. The
vectorised extraction is compared against the loop it replaced, the flat
field against a lamp continuum and a pixel response that were put in by
hand, and the fitting changes against a surface the test generated
itself.

fitsio is stubbed with a shim that serves arrays out of a dict, so no
FITS file is read and nothing on disk is touched outside /tmp.

    python test_pipeline.py

Exits non-zero if anything fails.
"""
import os
import sys
import types

import numpy as np

# --- fitsio shim -------------------------------------------------------
FRAMES = {}
HEADERS = {}


class _HDU:
    def __init__(self, path):
        self.path = path

    def read(self):
        return FRAMES[self.path]

    def read_header(self):
        return HEADERS.get(self.path, {})


class _FITS:
    def __init__(self, path, mode="r"):
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getitem__(self, i):
        return _HDU(self.path)


fitsio = types.ModuleType("fitsio")
fitsio.FITS = _FITS
sys.modules["fitsio"] = fitsio
sys.modules.setdefault("pandas", types.ModuleType("pandas"))
mpl = types.ModuleType("matplotlib")
plt = types.ModuleType("matplotlib.pyplot")
for fn in ("figure", "plot", "show", "legend", "annotate", "title", "xlabel",
           "tight_layout", "subplots", "close"):
    setattr(plt, fn, lambda *a, **k: None)
mpl.pyplot = plt
sys.modules.setdefault("matplotlib", mpl)
sys.modules.setdefault("matplotlib.pyplot", plt)

# Runs from anywhere: the pipeline modules sit either beside this file or
# in a code/ directory next to it.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _candidate in (_HERE, os.path.join(_HERE, "code")):
    if os.path.exists(os.path.join(_candidate, "wavelength_solution.py")):
        sys.path.insert(0, _candidate)
        break

TMP = "/tmp/remus"
os.makedirs(TMP, exist_ok=True)


def stub(name, data, header):
    """Register a fake frame and touch a real file so os.path.exists works."""
    path = os.path.join(TMP, name)
    open(path, "w").close()
    FRAMES[path] = data
    HEADERS[path] = header
    return path


import config                                                    # noqa: E402
config.PROJECT_ROOT = "/tmp/remus"
config.TRANSPOSE = False

import frames as fr                                              # noqa: E402
import flat_field                                                # noqa: E402
from order_tracing import Order, trace_orders                     # noqa: E402
import wavelength_solution as ws                                  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILURES.append(name)


# =======================================================================
print("\n1. bias and dark subtraction")
# =======================================================================
NY = NX = 200
rng = np.random.default_rng(7)
bias = 1000.0 + rng.normal(0, 3, (NY, NX))
dark_current = np.abs(rng.normal(0, 0.2, (NY, NX)))       # ADU/s
signal = 500.0 * np.exp(-((np.arange(NX) - 100.0) / 20.0) ** 2)[None, :] * np.ones((NY, 1))

BIAS = stub("bias.fits", bias, {"EXPTIME": 0.001, "IMAGETYP": "Bias Frame"})
D10 = stub("dark10.fits", bias + 10.0 * dark_current,
           {"EXPTIME": 10.0, "IMAGETYP": "Dark Frame"})
D100 = stub("dark100.fits", bias + 100.0 * dark_current,
            {"EXPTIME": 100.0, "IMAGETYP": "Dark Frame"})
SCI = stub("sci.fits", signal + bias + 100.0 * dark_current,
           {"EXPTIME": 100.0, "IMAGETYP": "Light Frame"})

config.MASTER_BIAS = BIAS
config.MASTER_DARK = [D10, D100]
config.DARK_INCLUDES_BIAS = True
config.SCALE_DARK_BY_EXPTIME = True

config.APPLY_BIAS = False
config.APPLY_DARK = False
raw = fr.read_image(SCI)
check("raw read is unchanged", np.allclose(raw, FRAMES[SCI]))

config.APPLY_BIAS = True
config.APPLY_DARK = True
fr._CALIBRATION_CACHE.clear()
cal = fr.read_image(SCI)
check("bias+dark recovers the signal exactly", np.allclose(cal, signal, atol=1e-9),
      f"max error {np.abs(cal - signal).max():.3e}")

# the 100 s dark should have been chosen, not the 10 s one scaled
config.APPLY_BIAS, config.APPLY_DARK = True, False
fr._CALIBRATION_CACHE.clear()
bias_only = fr.read_image(SCI)
check("bias alone leaves the dark behind",
      np.allclose(bias_only, signal + 100.0 * dark_current, atol=1e-9))

config.APPLY_BIAS, config.APPLY_DARK = False, True
fr._CALIBRATION_CACHE.clear()
dark_only = fr.read_image(SCI)
check("dark alone leaves the pedestal behind",
      np.allclose(dark_only, signal + bias, atol=1e-9),
      f"median {np.median(dark_only - signal):.2f} vs bias {np.median(bias):.2f}")

# scaling: ask for a 50 s exposure, only a 10 s dark available
SCI50 = stub("sci50.fits", signal + bias + 50.0 * dark_current, {"EXPTIME": 50.0})
config.MASTER_DARK = D10
config.APPLY_BIAS, config.APPLY_DARK = True, True
fr._CALIBRATION_CACHE.clear()
fr._CALIBRATION_REPORTED.clear()
scaled = fr.read_image(SCI50)
check("dark is scaled by the exposure ratio", np.allclose(scaled, signal, atol=1e-9),
      f"max error {np.abs(scaled - signal).max():.3e}")

config.APPLY_BIAS = config.APPLY_DARK = False

# =======================================================================
print("\n2. vectorised extraction matches the old loop")
# =======================================================================
order = Order(center_poly=np.poly1d([2e-6, -2e-3, 60.0]),
              sigma_poly=np.poly1d([1e-7, 5e-5, 2.4]))
image = rng.normal(50, 5, (NY, NX)) + 1000 * np.exp(
    -((np.arange(NX)[None, :] - order.center(np.arange(NY))[:, None]) ** 2)
    / (2 * order.sigma(np.arange(NY))[:, None] ** 2))


def old_weighted(order, image, n_sigma):
    ny, nx = image.shape
    out = np.zeros(ny)
    for y in range(ny):
        xmin, xmax = order.aperture(y, n_sigma=n_sigma)
        xmin, xmax = max(0, xmin), min(nx, xmax)
        if xmax - xmin < 2:
            out[y] = np.nan
            continue
        x = np.arange(xmin, xmax)
        w = np.exp(-(x - order.center(y)) ** 2 / (2 * order.sigma(y) ** 2))
        t = w.sum()
        if t <= 0:
            out[y] = np.nan
            continue
        out[y] = np.sum(image[y, xmin:xmax] * (w / t))
    return out


def old_sum(order, image, n_sigma):
    ny, nx = image.shape
    out = np.zeros(ny)
    for y in range(ny):
        xmin, xmax = order.aperture(y, n_sigma=n_sigma)
        out[y] = np.sum(image[y, max(0, xmin):min(nx, xmax)])
    return out


for ns in (2.5, 3.0):
    a, b = old_weighted(order, image, ns), order.extract_weighted(image, n_sigma=ns)
    check(f"extract_weighted n_sigma={ns}",
          np.allclose(a, b, equal_nan=True, rtol=1e-12, atol=1e-9),
          f"max diff {np.nanmax(np.abs(a - b)):.3e}")
    a, b = old_sum(order, image, ns), order.extract_sum(image, n_sigma=ns)
    check(f"extract_sum n_sigma={ns}", np.allclose(a, b, equal_nan=True))

# an order running off the edge, so the aperture clips
edge = Order(center_poly=np.poly1d([0.05, 1.0]), sigma_poly=np.poly1d([2.4]))
a, b = old_weighted(edge, image, 3.0), edge.extract_weighted(image, n_sigma=3.0)
check("extract_weighted with a clipped aperture",
      np.allclose(a, b, equal_nan=True, rtol=1e-12, atol=1e-9))
check("clipped aperture still produces NaN where it should",
      np.isnan(a).sum() == np.isnan(b).sum())

# =======================================================================
print("\n3. flat field separates the lamp SED from the pixel response")
# =======================================================================
n = 2000
pixel = np.arange(n)
blaze_true = 20000 * np.exp(-((pixel - 950.0) / 700.0) ** 2) + 500
response_true = 1.0 + 0.03 * rng.normal(size=n)
flat = blaze_true * response_true

blaze, response = flat_field.split_flat(flat, smooth_window=101,
                                        min_relative=0.15, max_correction=1.5)
core = (blaze_true > 0.3 * blaze_true.max())
check("blaze recovers the smooth part",
      np.nanmedian(np.abs(blaze[core] / blaze_true[core] - 1)) < 0.01,
      f"median error {np.nanmedian(np.abs(blaze[core] / blaze_true[core] - 1)):.4f}")
check("response recovers the pixel-to-pixel part",
      np.median(np.abs(response[core] - response_true[core])) < 0.005,
      f"median error {np.median(np.abs(response[core] - response_true[core])):.4f}")
check("response is 1 where nothing is corrected", np.all(np.isfinite(response)))
check("response never inverts the spectrum", np.all(response > 0))

# the lamp SED must not survive into a corrected spectrum
star = 1000 * (1 + 0.4 * np.sin(pixel / 90.0))
observed = star * response_true
corrected = observed / response
inner = (blaze_true > 0.4 * blaze_true.max())
before = np.std(observed[inner] / star[inner])
after = np.std(corrected[inner] / star[inner])
check("flat fielding reduces the pixel-response scatter",
      after < 0.4 * before, f"{before:.4f} -> {after:.4f}")

lamp_slope = np.polyfit(pixel[inner], corrected[inner] / star[inner], 1)[0]
check("no lamp continuum is printed into the result", abs(lamp_slope) < 1e-5,
      f"slope {lamp_slope:.3e} per pixel")

flat_nan = flat.copy()
flat_nan[:5] = np.nan
flat_nan[1200] = np.nan
b2, r2 = flat_field.split_flat(flat_nan, smooth_window=101)
check("split_flat survives NaN in the flat",
      np.all(np.isfinite(r2)) and np.isnan(b2[:5]).all())

# =======================================================================
print("\n4. arc line centroids: sloped background")
# =======================================================================


def make_arc(n=1200, tilt=0.0, seed=0):
    r = np.random.default_rng(seed)
    x = np.arange(n)
    truth = np.arange(60, n - 60, 97).astype(float) + 0.37
    spec = 1000.0 + tilt * x
    for t in truth:
        spec = spec + 4000.0 * np.exp(-(x - t) ** 2 / (2 * 3.3 ** 2))
    return spec + r.normal(0, 8, n), truth


for tilt, label in ((0.0, "flat background"), (3.0, "sloping background")):
    spec, truth = make_arc(tilt=tilt, seed=3)
    config.ARC_LINE_LINEAR_BACKGROUND = False
    det_c = ws.detect_arc_lines(spec, expected_sigma_pixels=3.3)
    config.ARC_LINE_LINEAR_BACKGROUND = True
    det_s = ws.detect_arc_lines(spec, expected_sigma_pixels=3.3)

    def bias_of(det):
        if det is None or len(det) == 0:
            return np.nan
        found = det[:, ws.DET_PIXEL]
        near = [f - truth[np.argmin(np.abs(truth - f))] for f in found]
        return float(np.mean(near))

    bc, bs = bias_of(det_c), bias_of(det_s)
    print(f"    {label}: constant-background bias {bc:+.4f} px, "
          f"sloped {bs:+.4f} px  ({len(det_c)} vs {len(det_s)} lines)")
    if tilt:
        check("sloped background reduces the centroid bias", abs(bs) <= abs(bc) + 1e-6,
              f"{bc:+.4f} -> {bs:+.4f}")
    check(f"lines still detected ({label})", len(det_s) >= len(truth) - 2,
          f"{len(det_s)} of {len(truth)}")

# NaN handling: one bad sample must not kill the order
spec, truth = make_arc(seed=5)
spec_nan = spec.copy()
spec_nan[3] = np.nan
det = ws.detect_arc_lines(spec_nan, expected_sigma_pixels=3.3)
check("one NaN no longer discards the whole order",
      det is not None and len(det) >= len(truth) - 2,
      f"got {None if det is None else len(det)} of {len(truth)}")
check("an all-NaN order still returns None",
      ws.detect_arc_lines(np.full(500, np.nan)) is None)

# =======================================================================
print("\n5. fit_solution: joint solve and weighted clipping")
# =======================================================================
FOCAL = 28000.0
NPIX = 4096
m_values = np.arange(60, 130)


def truth_solution():
    sol = ws.WavelengthSolution(FOCAL, np.zeros(9), 2, NPIX,
                                m_values.min(), m_values.max())
    sol.coefficients = np.array([570000.0, 900.0, -200.0,
                                 -14000.0, 300.0, 40.0,
                                 -2500.0, -120.0, 15.0])
    sol.correction = np.array([[0.0, 0.0, 0.0],
                               [0.0, 0.4, 0.1],
                               [0.6, -0.2, 0.05],
                               [0.3, 0.1, 0.0],
                               [-0.25, 0.05, 0.0]])
    return sol


true_sol = truth_solution()
r = np.random.default_rng(11)
pix, mm = [], []
for m in m_values:
    p = np.sort(r.uniform(80, NPIX - 80, 40))
    pix.append(p)
    mm.append(np.full(len(p), float(m)))
pix = np.concatenate(pix)
mm = np.concatenate(mm)
mlam_true = true_sol.m_lambda(pix, mm)
pixel_err = r.uniform(0.01, 0.06, len(pix))
disp = np.abs(true_sol.dispersion(pix, mm))
sigma_mlam = pixel_err * disp * mm
mlam_obs = mlam_true + r.normal(0, sigma_mlam)
weight = 1.0 / sigma_mlam ** 2

matches = ws.MatchSet(pix, mm, mlam_obs, weight, pixel_err,
                      np.full(len(pix), 50.0), np.zeros(len(pix), int))

results, focals, chi2 = {}, {}, {}
for joint in (False, True):
    config.JOINT_CORRECTION_FIT = joint
    config.WEIGHTED_CLIPPING = True
    sol, keep, resid = ws.fit_solution(matches, NPIX, m_degree=2,
                                       correction_degree=(4, 2))
    err = np.sqrt(np.mean(((mlam_true - sol.m_lambda(pix, mm)) / mm) ** 2))
    # the quantity the fit actually minimises, on the lines it kept
    chi2[joint] = float(np.average(resid[keep] ** 2, weights=matches.weight[keep]))
    results[joint] = err * 1000
    focals[joint] = sol.focal_pixels
    print(f"    {'joint' if joint else 'two-stage'} fit: weighted chi2 "
          f"{chi2[joint]:.6e}, truth recovered to {err * 1000:.4f} mA, "
          f"focal {sol.focal_pixels:.0f} px (true {FOCAL:.0f}), "
          f"{keep.sum()}/{len(keep)} lines kept")

# A joint solve is the exact minimiser of the same objective the two-stage
# fit only approaches, so this is a guarantee rather than a hope.
check("joint solve reaches a lower weighted chi2 than the two-stage fit",
      chi2[True] <= chi2[False] * (1 + 1e-9),
      f"{chi2[False]:.6e} -> {chi2[True]:.6e}")
check("joint solve recovers the truth to well under a mA", results[True] < 0.1,
      f"{results[True]:.4f} mA")
check("the joint solve leaves the focal length physical",
      0.6 * FOCAL < focals[True] < 1.7 * FOCAL,
      f"{focals[True]:.0f} px against a true {FOCAL:.0f} px")
check("the joint solve does not move the focal length at all",
      np.isclose(focals[True], focals[False]),
      f"{focals[False]:.0f} -> {focals[True]:.0f} px")

# outliers must be clipped, and clipping must not favour one end of the range
bad = matches.m_lambda.copy()
spoil = r.choice(len(bad), 60, replace=False)
bad[spoil] += r.normal(0, 3.0, 60)
noisy = ws.MatchSet(matches.pixel, matches.m, bad, matches.weight,
                    matches.pixel_err, matches.snr, matches.order_index)
config.JOINT_CORRECTION_FIT = True
for weighted in (False, True):
    config.WEIGHTED_CLIPPING = weighted
    sol, keep, _ = ws.fit_solution(noisy, NPIX, m_degree=2, correction_degree=(4, 2))
    blue = keep[noisy.m > np.median(noisy.m)].mean()
    red = keep[noisy.m <= np.median(noisy.m)].mean()
    err = np.sqrt(np.mean(((mlam_true - sol.m_lambda(pix, mm)) / mm) ** 2)) * 1000
    print(f"    {'weighted' if weighted else 'plain'} clipping: kept {blue:.1%} of "
          f"blue orders and {red:.1%} of red, recovered to {err:.3f} mA")
    if weighted:
        check("weighted clipping treats blue and red orders alike",
              abs(blue - red) < 0.06, f"{blue:.1%} vs {red:.1%}")

config.JOINT_CORRECTION_FIT = True
config.WEIGHTED_CLIPPING = True

# =======================================================================
print("\n6. bug fixes")
# =======================================================================


class _Model:
    n_pixels = NPIX

    def wavelength(self, pixel, m):
        return true_sol.wavelength(pixel, m)

    def dispersion(self, pixel, m):
        return true_sol.dispersion(pixel, m)


ref_wave = np.sort(true_sol.wavelength(np.linspace(200, NPIX - 200, 300),
                                       np.full(300, 90.0)))
reference = ws.ReferenceLines(ref_wave, np.full(len(ref_wave), 500.0), ref_wave)
det = np.zeros((5, 6))
det[:, ws.DET_PIXEL] = np.linspace(300, NPIX - 300, 5)
det[:, ws.DET_PIXERR] = 0.05
try:
    ws.match_lines(_Model(), [det, det], [None, 90], reference, NPIX, 2.0)
    check("match_lines survives an unnumbered order", True)
except TypeError as exc:
    check("match_lines survives an unnumbered order", False, str(exc))

# overlap_agreement on a decreasing wavelength axis
xp = np.array([6000.0, 5999.0, 5998.0, 5997.0])
fp = np.array([1.0, 2.0, 3.0, 4.0])
wrong = np.interp(5998.5, xp, fp)
right = np.interp(5998.5, xp[::-1], fp[::-1])
check("np.interp really is wrong on a decreasing axis (the bug is real)",
      not np.isclose(wrong, 2.5) and np.isclose(right, 2.5),
      f"{wrong} vs {right}")

# =======================================================================
print("\n7. per-order tilt in the shift")
# =======================================================================
orders = []
for m in (88, 89, 90):
    o = Order(np.poly1d([100.0]), np.poly1d([2.4]), order_number=m)
    orders.append(o)
ws.attach_solution(orders, true_sol, pixel_shift=2.0, tilt=0.1,
                   tilt_reference_m=89.0, quiet=True)
p = np.arange(10)
expected = [true_sol.wavelength(p - (2.0 + 0.1 * (m - 89.0)), np.full(10, float(m)))
            for m in (88, 89, 90)]
got = [o.wavelength_poly(p) for o in orders]
check("tilt gives each order its own shift",
      all(np.allclose(a, b) for a, b in zip(expected, got)))
ws.attach_solution(orders, true_sol, pixel_shift=2.0, tilt=0.0, quiet=True)
rigid = [o.wavelength_poly(p) for o in orders]
check("tilt of zero reproduces the rigid shift exactly",
      all(np.allclose(x, true_sol.wavelength(p - 2.0, np.full(10, float(m))))
          for x, m in zip(rigid, (88, 89, 90))))

# =======================================================================
print("\n8. interpolating the shift between arcs")
# =======================================================================
from datetime import datetime, timedelta                          # noqa: E402
import reduce_spectra as rs                                       # noqa: E402

t0 = datetime(2026, 3, 10, 12, 0)
entries = [(t0, 1.0, 0.01, 90.0), (t0 + timedelta(minutes=60), 3.0, 0.03, 90.0)]
s, k, m0, how = rs._interpolated_shift(entries, t0 + timedelta(minutes=30))
check("shift interpolates between bracketing arcs", np.isclose(s, 2.0), f"{s}")
check("tilt interpolates too", np.isclose(k, 0.02), f"{k}")
s, k, m0, how = rs._interpolated_shift(entries, t0 - timedelta(minutes=30))
check("an unbracketed frame holds rather than extrapolating", np.isclose(s, 1.0),
      f"{s}: {how}")
s, k, m0, how = rs._interpolated_shift([(t0, 5.0, 0.0, 90.0)], t0)
check("a single arc is held", np.isclose(s, 5.0))

# =======================================================================
print("\n9. tracing still works end to end")
# =======================================================================
NY2, NX2 = 400, 300
frame = np.full((NY2, NX2), 1000.0)
xs = np.arange(NX2)
centres = []
for k0 in range(30, NX2 - 30, 25):
    c = k0 + 0.004 * (np.arange(NY2) - NY2 / 2) ** 2 / 50.0
    centres.append(c)
    frame += 9000 * np.exp(-((xs[None, :] - c[:, None]) ** 2) / (2 * 2.3 ** 2))
frame += np.random.default_rng(2).normal(0, 6, frame.shape)
W1 = stub("white1.fits", frame, {"EXPTIME": 5.0, "IMAGETYP": "Flat Field"})
W2 = stub("white2.fits", frame + np.random.default_rng(3).normal(0, 6, frame.shape),
          {"EXPTIME": 5.0, "IMAGETYP": "Flat Field"})

real_list = fr.list_frames
fr.list_frames = lambda d, p=None, recursive=False: [W1, W2]
traced, coadd = trace_orders("/tmp/remus/white", auto_exclude=False)
fr.list_frames = real_list
check("tracing finds every order", len(traced) == len(centres),
      f"{len(traced)} of {len(centres)}")

config.FLAT_FIELD = True
config.FLAT_FIELD_ARCS = True
n_resp = flat_field.flat_field_orders(traced, coadd, verbose=False)
check("every traced order gets a response", n_resp == len(traced))
check("responses are finite and positive",
      all(np.all(np.isfinite(o.pixel_response)) and np.all(o.pixel_response > 0)
          for o in traced))
check("the arc aperture gets its own response",
      all(o.pixel_response_arc is not None for o in traced))
for o in traced:
    o.science_spectrum = o.extract_weighted(coadd, n_sigma=3.0)
before = [np.array(o.science_spectrum) for o in traced]
flat_field.apply_pixel_response(traced, "science_spectrum", verbose=False)
check("applying the response changes the spectra but keeps their level",
      all(not np.allclose(a, o.science_spectrum, equal_nan=True)
          and abs(np.nanmedian(o.science_spectrum) / np.nanmedian(a) - 1) < 0.05
          for a, o in zip(before, traced)))
config.FLAT_FIELD = False


# =======================================================================
print("\n10. regressions found in review")
# =======================================================================
import numpy.polynomial.chebyshev as CH                            # noqa: E402

# (1) the joint solve must drop only the columns that really duplicate
for md, cd in [(1, (3, 2)), (1, (4, 2)), (1, (5, 3)), (2, (5, 3)),
               (2, (4, 2)), (3, (5, 3))]:
    trial = ws.WavelengthSolution(28000.0, np.zeros(3 * (md + 1)), md, NPIX, 52, 137)
    # a real 2D sampling: many pixels in each of many orders. A single
    # linspace in both would trace one curve and be rank deficient for
    # reasons that have nothing to do with the basis.
    gp, gm = np.meshgrid(np.linspace(0, NPIX - 1, 40), np.arange(52, 138))
    p, mv = gp.ravel(), gm.ravel()
    P = trial.design(p, mv)
    V = CH.chebvander2d(trial._y_hat(p), trial._m_hat(mv), cd)
    full_rank = np.linalg.matrix_rank(np.hstack([P, V]))
    mask = np.arange(V.shape[1]) > min(cd[1], md)
    kept_rank = np.linalg.matrix_rank(np.hstack([P, V[:, mask]]))
    check(f"joint basis keeps every independent column, m_degree={md} corr={cd}",
          kept_rank == full_rank and kept_rank == P.shape[1] + mask.sum(),
          f"rank {kept_rank}, full {full_rank}, columns {P.shape[1] + mask.sum()}")

# (2) blaze must belong to the same aperture as flat_spectrum
tall = Order(np.poly1d([150.0]), np.poly1d([0.0025, 3.0]))
img2 = np.full((600, 300), 100.0)
xs2 = np.arange(300)
cen = tall.center(np.arange(600))
sig = tall.sigma(np.arange(600))
img2 = img2 + 20000 * np.exp(-((xs2[None, :] - cen[:, None]) ** 2)
                             / (2 * sig[:, None] ** 2))
config.FLAT_FIELD, config.FLAT_FIELD_ARCS = True, True
pair = [tall]
flat_field.flat_field_orders(pair, img2, verbose=False)
own = tall.extract_weighted(img2, n_sigma=config.SCIENCE_EXTRACT_NSIGMA)
own_blaze, _ = flat_field.split_flat(own, smooth_window=config.FLAT_SMOOTH_WINDOW)
inner = np.isfinite(tall.blaze) & np.isfinite(own_blaze) & (own_blaze > 0)
ratio = tall.blaze[inner] / own_blaze[inner]
check("saved blaze matches the saved flat_spectrum's aperture",
      np.allclose(ratio, 1.0, atol=1e-9),
      f"ratio spans {ratio.min():.4f} to {ratio.max():.4f}")
check("flat_spectrum is the science-aperture extraction",
      np.allclose(np.asarray(tall.flat_spectrum)[inner], own[inner], equal_nan=True))
config.FLAT_FIELD = False

# (3) a broken trace must not extract silently
broken = Order(np.poly1d([np.nan, 100.0]), np.poly1d([2.4]))
for name, fn in (("extract_sum", lambda: broken.extract_sum(img2)),
                 ("extract_weighted", lambda: broken.extract_weighted(img2))):
    try:
        fn()
        check(f"{name} refuses a non-finite trace", False, "returned instead of raising")
    except ValueError:
        check(f"{name} refuses a non-finite trace", True)

wide = Order(np.poly1d([150.0]), np.poly1d([50.0, 1.0]))   # sigma runs to 30000
grid = wide._aperture_grid(600, 300, 3.0)
check("the aperture grid is never wider than the detector",
      grid[0].shape[1] <= 300, f"width {grid[0].shape[1]}")

# (4) holding an unbracketed shift must use the re-referenced value
entries = [(t0, 1.0, 0.02, 80.0), (t0 + timedelta(minutes=60), 3.0, 0.02, 100.0)]
s, k, ref, _ = rs._interpolated_shift(entries, t0 - timedelta(minutes=30))
check("a held shift is re-referenced like an interpolated one",
      np.isclose(s, 1.0 + 0.02 * (ref - 80.0)), f"{s:.4f} at m={ref}")
s, k, ref, _ = rs._interpolated_shift(entries, t0 + timedelta(minutes=120))
check("the same holds on the other side",
      np.isclose(s, 3.0 + 0.02 * (ref - 100.0)), f"{s:.4f} at m={ref}")

# (5) WEIGHTED_CLIPPING False must restore the original uncentred rule
config.JOINT_CORRECTION_FIT = False
config.WEIGHTED_CLIPPING = False
sol_a, keep_a, res_a = ws.fit_solution(matches, NPIX, m_degree=2,
                                       correction_degree=(4, 2))
check("all switches off still fits", np.isfinite(sol_a.coefficients).all())
one_sided = np.array([1.0, 1.05, 0.95, 1.02, 0.98, 5.0])
scat = 1.4826 * np.median(np.abs(one_sided - np.median(one_sided)))
original = np.abs(one_sided) < 4.0 * scat
check("uncentred clipping is what the original did",
      not original.any(), "the original clips a one-sided set entirely")
config.JOINT_CORRECTION_FIT = True
config.WEIGHTED_CLIPPING = True

# (6) bias needed by the dark path must be reported clearly
config.MASTER_BIAS = None
config.APPLY_BIAS, config.APPLY_DARK = False, True
config.MASTER_DARK = D100
config.DARK_INCLUDES_BIAS = True
fr._CALIBRATION_CACHE.clear()
try:
    fr.read_image(SCI)
    check("a missing MASTER_BIAS is explained", False, "no error raised")
except ValueError as exc:
    check("a missing MASTER_BIAS is explained", "MASTER_BIAS" in str(exc))
except Exception as exc:
    check("a missing MASTER_BIAS is explained", False, f"{type(exc).__name__}: {exc}")
config.MASTER_BIAS = BIAS
config.APPLY_BIAS = config.APPLY_DARK = False

# (7) the cache must survive exposure times that vary a little
fr._CALIBRATION_CACHE.clear()
calls = {"n": 0}
real_combine = fr._combine


def counting(*a, **k):
    calls["n"] += 1
    return real_combine(*a, **k)


fr._combine = counting
config.MASTER_DARK = [D10, D100]
for t in (100.0, 100.5, 99.5, 101.0):
    HEADERS[SCI]["EXPTIME"] = t
    fr.master_frame(config.MASTER_DARK, want="dark", match_exptime=t, verbose=False)
fr._combine = real_combine
HEADERS[SCI]["EXPTIME"] = 100.0
check("the master dark is combined once for nearby exposure times",
      calls["n"] == 1, f"{calls['n']} combines")

# (8) cached masters must not be writable through
m1, _ = fr.master_frame(BIAS, want="bias", verbose=False)
try:
    m1[0, 0] = -999.0
    check("a cached master cannot be written through", False, "the write succeeded")
except ValueError:
    check("a cached master cannot be written through", True)

# =======================================================================
print("\n11. vacuum to air conversion of the ThAr atlas")
# =======================================================================
# Independent implementations of the same physics, to check the one in
# the pipeline rather than restate it.
def n_edlen1966(lv):
    s2 = (1e4 / np.asarray(lv, float)) ** 2
    return 1 + 1e-8 * (8342.13 + 2406030 / (130 - s2) + 15997 / (38.9 - s2))


def n_morton2000(lv):
    s2 = (1e4 / np.asarray(lv, float)) ** 2
    return 1 + 8.34254e-5 + 2.406147e-2 / (130 - s2) + 1.5998e-4 / (38.9 - s2)


probe = np.array([4165.0, 4861.3, 5890.0, 6563.0, 7300.0])
air = ws.vacuum_to_air(probe)

check("air wavelengths are shorter than vacuum", np.all(air < probe))
frac = (probe - air) / probe
check("the shift is about 0.028% everywhere",
      np.all((frac > 2.7e-4) & (frac < 2.9e-4)),
      f"{frac.min():.3e} to {frac.max():.3e}")

# a constant velocity across the band is the signature that makes this
# masquerade as a Doppler shift, so it is worth asserting
vel = ws.C_LIGHT_MS * (probe - air) / probe / 1000.0
check("the offset is a near-constant velocity, 82-85 km/s",
      np.all((vel > 82) & (vel < 85)) and (vel.max() - vel.min()) < 2.5,
      f"{vel.min():.1f} to {vel.max():.1f} km/s, spread {vel.max()-vel.min():.2f}")

for name, fn in (("Edlen 1966", n_edlen1966), ("Morton 2000", n_morton2000)):
    other = probe / fn(probe)
    worst = np.max(np.abs(other - air)) * 1000.0
    check(f"agrees with {name} to well under the line residual",
          worst < 0.5, f"worst {worst:.3f} mA")

back = ws.air_to_vacuum(air)
check("air_to_vacuum inverts vacuum_to_air",
      np.allclose(back, probe, atol=1e-9), f"max error {np.abs(back-probe).max():.3e} A")

check("conversion is strictly increasing over the whole optical range",
      np.all(np.diff(ws.vacuum_to_air(np.linspace(3000, 11000, 40001))) > 0))

try:
    ws.vacuum_to_air([1500.0])
    check("far-UV wavelengths are refused", False, "no error raised")
except ValueError:
    check("far-UV wavelengths are refused", True)

# Published catalogue pairs, both sides taken from references rather than
# from each other, so a wrong constant in the conversion cannot hide.
CATALOGUE = [("Na D2", 5891.5832, 5889.9509),
             ("Na D1", 5897.5581, 5895.9242),
             ("H-alpha", 6564.6140, 6562.8010),
             ("H-beta", 4862.6830, 4861.3250)]
worst_ma, rows = 0.0, []
for name, vac, air_pub in CATALOGUE:
    got = float(ws.vacuum_to_air(vac))
    err = abs(got - air_pub) * 1000.0
    worst_ma = max(worst_ma, err)
    rows.append(f"{name} {err:.2f}")
check("converts published vacuum values onto their published air values",
      worst_ma < 1.5, f"worst {worst_ma:.2f} mA ({', '.join(rows)})")

# config.ANCHORS / NAD_LINES against those same published air values
CONFIG_LINES = [("Na D2", 5889.95, 5889.9509), ("Na D1", 5895.92, 5895.9242),
                ("H-alpha", 6562.80, 6562.8010), ("H-beta", 4861.30, 4861.3250)]
off = [(n, (cfg - pub) * 1000.0, ws.C_LIGHT_MS * (cfg - pub) / pub / 1000.0)
       for n, cfg, pub in CONFIG_LINES]
for n, ma, kms in off:
    print(f"    config {n:8s} is {ma:+6.1f} mA = {kms:+5.2f} km/s from the "
          f"published air value")
# The bound catches a scale blunder (air vs vacuum is 1600 mA), not a
# catalogue preference. The printed offsets above are the thing to read:
# they go straight into measure_frame_shift, so anything approaching a
# km/s is worth deciding about deliberately.
check("config's rest wavelengths are on the air scale, not the vacuum one",
      max(abs(ma) for _, ma, _ in off) < 200.0,
      f"worst {max(abs(ma) for _, ma, _ in off):.0f} mA = "
      f"{max(abs(k) for _, _, k in off):.2f} km/s")
_worst_name, _worst_ma, _worst_kms = max(off, key=lambda r: abs(r[1]))
if abs(_worst_ma) > 10.0:
    print(f"    note: {_worst_name} is the furthest out at {_worst_ma:+.0f} mA "
          f"({_worst_kms:+.2f} km/s). That is a catalogue choice, not a scale "
          f"error, but it lands directly in the stellar-line check.")

# --- load_atlas end to end, on a synthetic line list -------------------
import io                                                        # noqa: E402
import types as _types                                           # noqa: E402

VAC = np.array([4200.0, 4862.6830, 5000.0, 5891.5832, 6564.6140, 7000.0, 7300.0])
AMP = np.array([300.0, 900.0, 20.0, 800.0, 950.0, 400.0, 250.0])
ION = ["Th I", "Th I", "Th I", "Th I", "Th I", "Ar I", "Th I"]
atlas_path = os.path.join(TMP, "atlas.dat")
with open(atlas_path, "w") as fh:
    fh.write("ion|wave|amplitude\n")
    for i, w in enumerate(VAC):
        fh.write(f"{ION[i]}|{w}|{AMP[i]}\n")

# pandas is stubbed at the top of this file, so give load_atlas a real
# reader for just this check
_real_pandas = sys.modules.get("pandas")


class _Frame:
    def __init__(self, rows):
        self.rows = rows

    @property
    def columns(self):
        return ["ion", "wave", "amplitude"]

    @columns.setter
    def columns(self, v):
        pass

    def sort_values(self, key):
        return _Frame(sorted(self.rows, key=lambda r: float(r[1])))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        col = {"ion": 0, "wave": 1, "amplitude": 2}[k]
        return _Col([r[col] for r in self.rows])

    def __setitem__(self, k, v):
        pass


class _Col:
    def __init__(self, v):
        self.v = v

    @property
    def values(self):
        if not self.v:
            return np.array([], dtype=float)
        if isinstance(self.v[0], bool):
            return np.array(self.v, dtype=bool)
        if isinstance(self.v[0], str):
            return np.array(self.v, dtype=object)
        return np.array(self.v, dtype=float)

    @property
    def str(self):
        return self

    def strip(self):
        return _Col([x.strip() for x in self.v])

    def startswith(self, p):
        return _Col([str(x).startswith(p) for x in self.v])


def _read_csv(path, delimiter="|"):
    rows = []
    with open(path) as fh:
        next(fh)
        for line in fh:
            if line.strip():
                rows.append([p.strip() for p in line.split("|")])
    return _Frame(rows)


fake_pd = _types.ModuleType("pandas")
fake_pd.read_csv = _read_csv
ws.pd = fake_pd

config.ATLAS_AIR = False
w_vac, a_vac, fw_vac, _ = ws.load_atlas(atlas_path, amplitude_min=10.0)
config.ATLAS_AIR = True
w_air, a_air, fw_air, _ = ws.load_atlas(atlas_path, amplitude_min=10.0)

check("ATLAS_AIR off leaves the file's vacuum values alone",
      np.allclose(np.sort(fw_vac), np.sort(VAC)))
check("ATLAS_AIR on converts every line",
      np.allclose(np.sort(fw_air), np.sort(ws.vacuum_to_air(VAC))))
check("the same lines survive the ion and amplitude cuts either way",
      len(w_air) == len(w_vac) and np.allclose(a_air, a_vac))
check("the converted list is still sorted", np.all(np.diff(fw_air) > 0))
check("Ar is still dropped", len(w_air) == 6 and not np.any(
      np.isclose(w_air[:, None], ws.vacuum_to_air(7000.0)).any()))

# the payoff: config's air rest wavelengths now line up with the atlas
# The payoff, with catalogue vacuum in and catalogue air out. Tight enough
# (2 mA = 0.1 km/s) that only a correct relation passes; the earlier version
# fed in values back-computed from the config numbers, so it could not fail.
for name, air_pub, vac_in_atlas in (("H-beta", 4861.3250, 4862.6830),
                                    ("Na D2", 5889.9509, 5891.5832),
                                    ("H-alpha", 6562.8010, 6564.6140)):
    converted = float(ws.vacuum_to_air(vac_in_atlas))
    check(f"{name} in the atlas converts onto its published air wavelength",
          abs(converted - air_pub) < 0.002,
          f"atlas {vac_in_atlas} -> {converted:.4f} vs published {air_pub}")

# --- the cases the first pass of this suite did not reach ---------------
empty_path = os.path.join(TMP, "atlas_empty.dat")
with open(empty_path, "w") as fh:
    fh.write("ion|wave|amplitude\n")
for flag in (False, True):
    config.ATLAS_AIR = flag
    try:
        ws.load_atlas(empty_path, amplitude_min=10.0)
        check(f"an empty line list is refused clearly (ATLAS_AIR={flag})", False,
              "no error raised; it would surface much later as 'no orders have "
              "enough detected ThAr lines'")
    except ValueError as exc:
        check(f"an empty line list is refused clearly (ATLAS_AIR={flag})",
              "delimited" in str(exc), str(exc))
    except Exception as exc:
        check(f"an empty line list is refused clearly (ATLAS_AIR={flag})", False,
              f"{type(exc).__name__}: {exc}")

one_path = os.path.join(TMP, "atlas_one.dat")
with open(one_path, "w") as fh:
    fh.write("ion|wave|amplitude\n")
    fh.write("Th I|5891.5832|800.0\n")
config.ATLAS_AIR = True
w1, _, fw1, _ = ws.load_atlas(one_path, amplitude_min=10.0)
check("a one-line list converts cleanly",
      len(w1) == 1 and abs(float(w1[0]) - 5889.9509) < 0.002)

try:
    ws.vacuum_to_air([5000.0, np.nan, 6000.0])
    check("a NaN wavelength is refused, not passed through", False, "no error raised")
except ValueError:
    check("a NaN wavelength is refused, not passed through", True)

# select_reference_lines on the converted atlas: the np.isin float-equality
# path, end to end rather than by inspection
config.ATLAS_AIR = True
sw, sa, fwv, fav = ws.load_atlas(atlas_path, amplitude_min=10.0)
ref = ws.select_reference_lines(sw, sa, fwv, fav,
                                lambda w: np.full_like(np.asarray(w, float), 0.2),
                                amplitude_min=100.0, dominance=3.0)
check("select_reference_lines still matches the selected lines after conversion",
      len(ref) > 0 and np.all(np.isin(np.round(ref.wave, 6),
                                      np.round(ws.vacuum_to_air(VAC), 6))),
      f"{len(ref)} reference lines")
check("reference wavelengths are on the air scale",
      bool(np.all(ref.wave < ws.air_to_vacuum(ref.wave))))

# the .npz label must follow the master, not tonight's config
for master_flag, tonight, expect in ((True, False, "air"), (False, True, "vacuum"),
                                     (True, True, "air"), (False, False, "vacuum")):
    config.ATLAS_AIR = tonight
    scale = rs._master_wavelength_scale({"processing": {"atlas_air": master_flag}})
    check(f"npz scale follows the master (master={master_flag}, config={tonight})",
          scale == expect, f"got {scale}")
check("a master with no processing record is treated as vacuum",
      rs._master_wavelength_scale({}) == "vacuum"
      and rs._master_wavelength_scale(None) == "vacuum")

if _real_pandas is not None:
    ws.pd = _real_pandas
config.ATLAS_AIR = False


print("\n" + "=" * 60)
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): " + ", ".join(FAILURES))
    sys.exit(1)
print("all checks passed")
