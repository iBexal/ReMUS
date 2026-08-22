"""
wavelength_solution.py

The wavelength model, and everything that fits or judges one. Building a
master lives in build_master_thar.py, reusing one in apply_master_thar.py;
both are built out of what is here.

Every class that ends up inside a saved master is defined in THIS module on
purpose. Pickle records the module a class came from and looks it up by
that name when loading, so moving one of these elsewhere would quietly
break every master already written.

The whole module is built on one idea: for a grating spectrograph the
quantity that behaves simply is not the wavelength, it is the product

    m * lambda

Along a single order the grating equation says

    m * lambda = d * (sin(alpha) + sin(beta))

with alpha (incidence) fixed by the instrument, and beta (diffraction
angle) set purely by where the light lands on the detector. Every order
is imaged by the same camera, so beta is the same function of detector
row for all of them. That means a SINGLE smooth curve

    m * lambda(y) = a0 + (a1 * u + a2 * f) / sqrt(f**2 + u**2),   u = y - y_centre

describes every order at once: f is the camera focal length in pixels,
and the sqrt() is just tan->sin for the camera's mapping of row to angle.
The only order-to-order freedom left is the small out-of-plane (cos gamma)
term and any detector rotation, which enter as a slowly varying function
of m -- here a low-degree polynomial multiplying each of the three basis
terms.

Two things follow, and they are the reason this module is shaped the way
it is:

  * Orders cannot disagree at their overlaps. Adjacent orders are not
    fitted separately and then hoped to agree; they are two evaluations
    of one surface. Overlap agreement becomes a measurement of how good
    the surface is, not something to be patched up afterwards.

  * The model extrapolates honestly. A high-degree polynomial fitted to
    the pixel range where ThAr lines happen to be dense goes wild outside
    it. This basis is the actual optics, so the ends of each order stay
    sane even before lines are matched there -- which is what lets the
    line matching reach the ends at all (see solve()).

A small Chebyshev correction in (pixel, order) is fitted on top to absorb
whatever the idealised optics misses. It is deliberately low degree so it
cannot undo the good extrapolation behaviour.

The pipeline:

    1. order numbers          assign_order_numbers()   -- from Halpha/Hbeta
    2. seed dispersion        seed_from_doublet()      -- from clicked Na D
    3. global lock            lock_seed()              -- template correlation
                                                          of every order at
                                                          once against the atlas
    4. iterative solve        solve()                  -- match, fit, tighten
    5. quality gate           assess()                 -- CV + overlap + trends
    6. save                    save_solution()
    7. reuse                   apply_master_thar.reduce_science()

Nothing here touches order tracing or extraction; that is order_tracing.py.
"""

import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev as C
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

C_LIGHT_MS = 299792458.0


# ======================================================================
# grating geometry
# ======================================================================

def compute_grating_K(blaze_angle_deg, groove_density_mm):
    """Littrow grating constant K, in Angstrom, such that m * lambda ~ K at
    the blaze peak: K = 2 * d * sin(blaze), d = groove spacing.

    Used only as a starting guess and as a sanity check on the fitted
    solution -- nothing downstream depends on it being exactly right.
    """
    d_angstrom = 1e7 / groove_density_mm
    return 2.0 * d_angstrom * np.sin(np.deg2rad(blaze_angle_deg))


def assign_order_numbers(orders, K, anchors, direction):
    """Give every traced order its physical echelle order number m.

    anchors : list of (trace_index, rest_wavelength) for lines whose order
        you are confident about -- Halpha and Hbeta. Each anchor gives
        m ~ K / lambda; rounding that to an integer and stepping back to
        trace index 0 gives m0. Two anchors far apart in trace index is a
        strong test: they must agree on m0, and the number of orders
        between them must equal the difference in their order numbers.

    direction : +1 or -1 in m_i = m0 + direction * i.

    Returns m0. Mutates orders in place.
    """
    implied = []
    print("Order numbering from anchors:")
    for i, wave in anchors:
        m_float = K / wave
        m_round = int(round(m_float))
        m0_i = m_round - direction * i
        implied.append(m0_i)
        print(f"  trace {i:3d}  lambda={wave:9.2f} A  ->  m = K/lambda = {m_float:7.3f}"
              f"  -> m = {m_round}  -> implies m0 = {m0_i}")

    if len(set(implied)) > 1:
        raise ValueError(
            f"Anchors disagree on m0 ({implied}). Either the grating constant is wrong "
            f"for this configuration, `direction` has the wrong sign, or an order is "
            f"missing from the trace list between the two anchor traces (which would "
            f"shift every trace index past it). Fix this before going any further -- "
            f"every wavelength downstream is built on it.")

    m0 = implied[0]
    for i, order in enumerate(orders):
        order.order_number = int(m0 + direction * i)

    ms = np.array([o.order_number for o in orders])
    print(f"  -> m0 = {m0}, orders run m={ms.max()} (trace 0) down to m={ms.min()} "
          f"(trace {len(orders) - 1})")
    print(f"  -> nominal central wavelengths K/m: {K / ms.max():.0f} A (bluest) to "
          f"{K / ms.min():.0f} A (reddest)")
    return m0


def check_trace_spacing(orders, tolerance=0.35):
    """Warn if the spacing between adjacent traces suggests a MISSED order.

    Order numbering is 'm0 + direction * trace_index', so an order that the
    tracer failed to pick up *between* two traced orders silently shifts
    every order number past it by one and wrecks the solution. Missed
    orders at the very blue or red end are harmless by comparison -- they
    cost coverage, not correctness.

    The separation between adjacent orders on the detector varies smoothly,
    so a gap that is far larger than its neighbours is the signature of a
    missing order. Returns a list of trace indices after which a gap looks
    suspicious (empty is what you want).
    """
    x = np.array([o.trace_center_pixel for o in orders], dtype=float)
    gaps = np.diff(x)
    if len(gaps) < 5:
        return []
    smooth = median_filter(gaps, size=7, mode="nearest")
    ratio = gaps / smooth
    bad = np.where(ratio > 1.0 + tolerance)[0]
    for b in bad:
        print(f"  WARNING: gap between trace {b} (x={x[b]:.0f}) and {b + 1} (x={x[b + 1]:.0f}) "
              f"is {gaps[b]:.0f} px, {ratio[b]:.2f}x the local trend -- an order may be "
              f"missing here, which would shift every order number past it.")
    if len(bad) == 0:
        print(f"  trace spacing is smooth across all {len(orders)} orders "
              f"(no missing order between them).")
    return bad.tolist()


# ======================================================================
# ThAr atlas
# ======================================================================

class ReferenceLines:
    """The subset of the atlas that is actually usable for calibration.

    `wave` is the catalogue wavelength; `eff_wave` is the amplitude-weighted
    centroid of everything the instrument blends into that line at its
    resolution. A centroid measured on the detector measures the blend, so
    `eff_wave` is what it should be compared against.
    """

    def __init__(self, wave, eff_wave, amplitude):
        order = np.argsort(wave)
        self.wave = np.asarray(wave, float)[order]
        self.eff_wave = np.asarray(eff_wave, float)[order]
        self.amplitude = np.asarray(amplitude, float)[order]

    def __len__(self):
        return len(self.wave)


def load_atlas(path, ion_prefix="Th", amplitude_min=10.0):
    """Read a ThAr line list (the '|'-delimited NIST/Murphy format).

    Ar lines are dropped by default. They are genuinely worse for precision
    work -- more prone to blending, self-absorption and pressure shifts than
    Th -- and with 15000 Th lines available there is no need for them.

    Returns (wave, amplitude, full_wave, full_amplitude): the selected lines,
    plus the complete list, which is kept because the blend/dominance test
    below has to know about the lines it is rejecting, not just the ones it
    is keeping.
    """
    atlas = pd.read_csv(path, delimiter="|")
    atlas.columns = [c.strip() for c in atlas.columns]
    atlas["ion"] = atlas["ion"].str.strip()
    atlas = atlas.sort_values("wave")

    full_wave = atlas["wave"].values.astype(float)
    full_amp = atlas["amplitude"].values.astype(float)

    keep = atlas["ion"].str.startswith(ion_prefix).values & (full_amp > amplitude_min)
    print(f"load_atlas: {len(atlas)} lines, kept {keep.sum()} with ion '{ion_prefix}*' "
          f"and amplitude > {amplitude_min}")
    return full_wave[keep], full_amp[keep], full_wave, full_amp


def select_reference_lines(sel_wave, sel_amp, full_wave, full_amp,
                           resolution_angstrom, amplitude_min=200.0,
                           dominance=5.0):
    """Keep only lines that a spectrograph of this resolution can actually
    measure without bias, and give each one an effective wavelength.

    This matters more than it sounds. This atlas carries roughly four lines
    per Angstrom in the blue while a resolution element is about a tenth of
    an Angstrom, so most catalogue lines are unresolvable neighbours of a
    stronger one. Matching a measured centroid to whichever catalogue entry
    happens to be nearest then pulls the fit around by an amount that
    depends on the local blend -- a systematic that looks like noise and
    does not average away.

    A line survives if it is strong (`amplitude_min`) and dominates its own
    resolution element: its amplitude must exceed `dominance` times the sum
    of every other line within one resolution width. What survives gets an
    effective wavelength, the amplitude-weighted centroid of that whole
    group, which is what the detector actually sees.

    resolution_angstrom : callable lambda -> resolution width in Angstrom.
    """
    w = np.asarray(full_wave, float)
    a = np.asarray(full_amp, float)
    width = np.asarray(resolution_angstrom(w), float)

    lo = np.searchsorted(w, w - width)
    hi = np.searchsorted(w, w + width)
    cum_a = np.concatenate([[0.0], np.cumsum(a)])
    cum_aw = np.concatenate([[0.0], np.cumsum(a * w)])
    group_amp = cum_a[hi] - cum_a[lo]
    group_awave = cum_aw[hi] - cum_aw[lo]
    neighbours = group_amp - a
    eff = group_awave / np.maximum(group_amp, 1e-30)

    selected = np.isin(w, sel_wave)
    keep = selected & (a > amplitude_min) & (a > dominance * neighbours)

    ref = ReferenceLines(w[keep], eff[keep], a[keep])
    shift = np.abs(ref.wave - ref.eff_wave)
    print(f"select_reference_lines: {len(ref)} isolated, dominant lines "
          f"(amplitude > {amplitude_min}, > {dominance}x its blend neighbours); "
          f"blend correction median {np.median(shift) * 1000:.2f} mA, "
          f"max {shift.max() * 1000:.1f} mA")
    return ref


# ======================================================================
# arc line detection
# ======================================================================

def _gaussian(x, amp, mu, sigma, offset):
    return amp * np.exp(-(x - mu) ** 2 / (2.0 * sigma ** 2)) + offset


# columns of the detection array
DET_PIXEL, DET_SIGMA, DET_AMP, DET_SNR, DET_PIXERR, DET_FITRES = range(6)


def reference_lines_for(atlas_path, m_lambda_centre, m_lambda_half_range, n_pixels,
                        line_sigma_pixels, amplitude_min=200.0, dominance=5.0):
    """Load the atlas and keep the lines this instrument can actually use.

    Needs a rough idea of the dispersion, because how isolated a line has to
    be depends on what the spectrograph can separate, and that width scales
    with wavelength. A seed provides it when building; the saved surface
    provides it when applying.
    """
    sel_wave, sel_amp, full_wave, full_amp = load_atlas(atlas_path)
    fwhm_pixels = 2.355 * line_sigma_pixels

    def resolution_angstrom(wave):
        return (fwhm_pixels * wave * 2.0 * m_lambda_half_range
                / ((n_pixels - 1) * m_lambda_centre))

    return select_reference_lines(sel_wave, sel_amp, full_wave, full_amp,
                                  resolution_angstrom, amplitude_min=amplitude_min,
                                  dominance=dominance)


def detect_arc_lines(spectrum, expected_sigma_pixels=3.3, detection_sigma=7.0,
                     saturation=45000.0, min_separation=8, half_window=9,
                     width_tolerance=(0.5, 2.0), max_fit_residual=0.15,
                     continuum_window=201):
    """Find ThAr emission lines in one extracted order and centroid them.

    Everything here exists to make sure a line that reaches the fit is one
    whose centroid means what it says. Rejected outright:

      * saturated lines -- the profile is clipped, so the centroid is
        biased by however the clipping happens to be distributed. In the
        red these are the brightest lines, and keeping them at reduced
        weight (rather than dropping them) is a false economy: they bias
        the fit in exactly the orders where there are fewest other lines
        to outvote them.
      * lines much narrower or wider than the instrumental profile -- a
        cosmic ray or an unresolved blend, respectively.
      * lines whose Gaussian fit leaves large structured residuals -- an
        asymmetric blend, whose centroid is pulled off the true line.

    Returns an (n, 6) array of columns
    (pixel, sigma, amplitude, snr, pixel_error, fractional_fit_residual),
    or None if the spectrum has non-finite values (a dead trace).
    """
    s = np.asarray(spectrum, float)
    if not np.all(np.isfinite(s)):
        return None

    continuum = median_filter(s, continuum_window)
    resid = s - continuum
    mad = np.median(np.abs(resid - np.median(resid)))
    noise = 1.4826 * mad if mad > 0 else np.std(resid)
    if noise <= 0:
        return np.zeros((0, 6))

    peaks, _ = find_peaks(resid, prominence=detection_sigma * noise,
                          distance=min_separation)

    out = []
    for p in peaks:
        if s[p] >= saturation:
            continue
        lo = max(0, p - half_window)
        hi = min(len(s), p + half_window + 1)
        if hi - lo < 7:
            continue
        x = np.arange(lo, hi)
        y = resid[lo:hi]
        try:
            popt, pcov = curve_fit(_gaussian, x, y,
                                   p0=(y.max(), float(p), expected_sigma_pixels, 0.0))
        except Exception:
            continue
        amp, mu, sigma, _ = popt
        sigma = abs(sigma)
        if amp <= 0 or not (lo + 1 < mu < hi - 1):
            continue
        if not (width_tolerance[0] * expected_sigma_pixels < sigma
                < width_tolerance[1] * expected_sigma_pixels):
            continue
        model = _gaussian(x, *popt)
        frac_resid = np.sqrt(np.mean((y - model) ** 2)) / amp
        if frac_resid > max_fit_residual:
            continue
        try:
            pixel_err = float(np.sqrt(abs(pcov[1, 1])))
        except Exception:
            pixel_err = np.nan
        if not np.isfinite(pixel_err) or pixel_err <= 0:
            pixel_err = 1.0
        out.append((mu, sigma, amp, amp / noise, pixel_err, frac_resid))

    return np.array(out) if out else np.zeros((0, 6))


def detect_all_orders(orders, **kwargs):
    """Run detect_arc_lines over every order's extracted ThAr spectrum.

    Returns a list parallel to `orders` (None for dead/blank traces) and
    prints a one-line summary, since a sudden collapse in the number of
    lines found is the first sign something is wrong with the extraction.
    """
    detections = []
    n_total = 0
    n_dead = 0
    for order in orders:
        if order.thar_spectrum is None:
            detections.append(None)
            n_dead += 1
            continue
        d = detect_arc_lines(order.thar_spectrum, **kwargs)
        if d is None:
            n_dead += 1
        else:
            n_total += len(d)
        detections.append(d)
    counts = [0 if d is None else len(d) for d in detections]
    print(f"detect_all_orders: {n_total} usable ThAr lines over {len(orders) - n_dead} orders "
          f"(median {int(np.median(counts))} per order, "
          f"range {min(counts)}-{max(counts)}, {n_dead} dead traces)")
    return detections


# ======================================================================
# the solution itself
# ======================================================================

class WavelengthSolution:
    """pixel, order number -> wavelength, for the whole detector at once.

    m * lambda = [physical camera terms] x [low-degree polynomial in m]
                 + [small Chebyshev correction in (pixel, m)]

    The three physical terms are 1, u/sqrt(f^2+u^2) and f/sqrt(f^2+u^2)
    with u = pixel - pixel_centre: constant, sin(beta) and cos(beta) for a
    camera of focal length f pixels. Each is allowed its own low-degree
    polynomial in normalised order number, which covers the out-of-plane
    cos(gamma) term and any detector rotation.

    Module-level class on purpose: it has to survive pickling into the
    saved master solution, which a closure would not.
    """

    def __init__(self, focal_pixels, coefficients, m_degree, n_pixels,
                 m_min, m_max, correction=None):
        self.focal_pixels = float(focal_pixels)
        self.coefficients = np.asarray(coefficients, float)
        self.m_degree = int(m_degree)
        self.n_pixels = int(n_pixels)
        self.m_min = float(m_min)
        self.m_max = float(m_max)
        self.correction = None if correction is None else np.asarray(correction, float)

    # -- coordinate normalisation ------------------------------------
    def _y_hat(self, pixel):
        return 2.0 * (np.asarray(pixel, float) - (self.n_pixels - 1) / 2.0) / (self.n_pixels - 1)

    def _m_hat(self, m):
        mid = (self.m_max + self.m_min) / 2.0
        half = max((self.m_max - self.m_min) / 2.0, 1.0)
        return (np.asarray(m, float) - mid) / half

    def design(self, pixel, m):
        """Design matrix of the physical part, shape (n, 3 * (m_degree + 1))."""
        u = np.asarray(pixel, float) - (self.n_pixels - 1) / 2.0
        d = np.sqrt(self.focal_pixels ** 2 + u ** 2)
        basis = np.vstack([np.ones_like(u), u / d, self.focal_pixels / d])
        mh = self._m_hat(m)
        cols = [basis[k] * mh ** j for k in range(3) for j in range(self.m_degree + 1)]
        return np.vstack(cols).T

    # -- evaluation --------------------------------------------------
    def m_lambda(self, pixel, m):
        pixel = np.asarray(pixel, float)
        m = np.asarray(m, float)
        pixel, m = np.broadcast_arrays(pixel, m)
        flat_p, flat_m = pixel.ravel(), m.ravel()
        value = self.design(flat_p, flat_m) @ self.coefficients
        if self.correction is not None:
            value = value + C.chebval2d(self._y_hat(flat_p), self._m_hat(flat_m),
                                        self.correction)
        return value.reshape(pixel.shape)

    def wavelength(self, pixel, m):
        pixel = np.asarray(pixel, float)
        m = np.asarray(m, float)
        pixel, m = np.broadcast_arrays(pixel, m)
        return self.m_lambda(pixel, m) / m

    def dispersion(self, pixel, m):
        """Angstrom per pixel at (pixel, m)."""
        pixel = np.asarray(pixel, float)
        return (self.wavelength(pixel + 0.5, m) - self.wavelength(pixel - 0.5, m))

    def order_axis(self, m, n_pixels=None):
        """The full wavelength axis of one order."""
        n = self.n_pixels if n_pixels is None else n_pixels
        pix = np.arange(n)
        return self.wavelength(pix, np.full(n, float(m)))

    def for_order(self, m):
        """A one-argument callable pixel -> wavelength for a single order,
        so per-order code (plots, extraction, the saved solution) can hold
        something that behaves like the old per-order polynomial."""
        return OrderWavelength(self, m)


class OrderWavelength:
    """One order's slice of a WavelengthSolution. Picklable and callable.

    pixel_shift moves the axis across the detector -- the right correction
    for flexure, and the wrong one for anything Doppler. velocity_ms scales
    it instead, which is the right correction for a Doppler shift and, being
    the same fraction in every order, cannot disturb order overlap. See
    diagnose_frame_offset for which one an exposure needs.
    """

    def __init__(self, solution, m, pixel_shift=0.0, velocity_ms=0.0):
        self.solution = solution
        self.m = float(m)
        self.pixel_shift = float(pixel_shift)
        self.velocity_ms = float(velocity_ms)

    def __call__(self, pixel):
        pixel = np.asarray(pixel, float) - self.pixel_shift
        w = self.solution.wavelength(pixel, np.full(pixel.shape, self.m))
        return w / (1.0 + self.velocity_ms / C_LIGHT_MS)


class PolynomialSurface:
    """m * lambda as a plain Chebyshev surface in (pixel, order number).

    Used only to warm the solution up. It is well conditioned at low degree
    and does not care how bad the starting matches are, which is what the
    first couple of passes need. It is a poor final model -- outside the
    pixel range where lines happened to be matched it does whatever a
    polynomial does -- so solve() hands over to WavelengthSolution as soon
    as there are enough matches to pin the camera geometry down.
    """

    def __init__(self, coefficients, n_pixels, m_min, m_max):
        self.coefficients = np.asarray(coefficients, float)
        self.n_pixels = int(n_pixels)
        self.m_min = float(m_min)
        self.m_max = float(m_max)

    def _y_hat(self, pixel):
        return 2.0 * (np.asarray(pixel, float) - (self.n_pixels - 1) / 2.0) / (self.n_pixels - 1)

    def _m_hat(self, m):
        mid = (self.m_max + self.m_min) / 2.0
        half = max((self.m_max - self.m_min) / 2.0, 1.0)
        return (np.asarray(m, float) - mid) / half

    def m_lambda(self, pixel, m):
        pixel, m = np.broadcast_arrays(np.asarray(pixel, float), np.asarray(m, float))
        return C.chebval2d(self._y_hat(pixel), self._m_hat(m), self.coefficients)

    def wavelength(self, pixel, m):
        pixel, m = np.broadcast_arrays(np.asarray(pixel, float), np.asarray(m, float))
        return self.m_lambda(pixel, m) / m

    def dispersion(self, pixel, m):
        pixel = np.asarray(pixel, float)
        return self.wavelength(pixel + 0.5, m) - self.wavelength(pixel - 0.5, m)


def fit_polynomial_surface(matches, n_pixels, degrees, clip_sigma=4.0, iterations=4):
    """Robust weighted least squares for the warm-up surface."""
    keep = np.ones(len(matches), bool)
    solution = None
    residuals = np.zeros(len(matches))
    n_free = (degrees[0] + 1) * (degrees[1] + 1)
    for _ in range(iterations):
        V = C.chebvander2d(
            2.0 * (matches.pixel[keep] - (n_pixels - 1) / 2.0) / (n_pixels - 1),
            (matches.m[keep] - (matches.m.max() + matches.m.min()) / 2.0)
            / max((matches.m.max() - matches.m.min()) / 2.0, 1.0),
            degrees)
        sw = np.sqrt(matches.weight[keep])
        coef, *_ = np.linalg.lstsq(V * sw[:, None], matches.m_lambda[keep] * sw, rcond=None)
        solution = PolynomialSurface(coef.reshape(degrees[0] + 1, degrees[1] + 1),
                                     n_pixels, matches.m.min(), matches.m.max())
        residuals = matches.m_lambda - solution.m_lambda(matches.pixel, matches.m)
        scatter = 1.4826 * np.median(np.abs(residuals[keep] - np.median(residuals[keep])))
        new_keep = np.abs(residuals) < clip_sigma * max(scatter, 1e-12)
        if np.array_equal(new_keep, keep) or new_keep.sum() < 3 * n_free:
            break
        keep = new_keep
    return solution, keep, residuals


class LinearSeed:
    """m * lambda = A + B * y_hat -- the starting guess only.

    Deliberately the simplest thing that can be pinned down from a single
    clicked doublet, and never used past the first matching pass.
    """

    def __init__(self, A, B, n_pixels):
        self.A = float(A)
        self.B = float(B)
        self.n_pixels = int(n_pixels)

    def _y_hat(self, pixel):
        return 2.0 * (np.asarray(pixel, float) - (self.n_pixels - 1) / 2.0) / (self.n_pixels - 1)

    def m_lambda(self, pixel, m):
        pixel, m = np.broadcast_arrays(np.asarray(pixel, float), np.asarray(m, float))
        return self.A + self.B * self._y_hat(pixel)

    def wavelength(self, pixel, m):
        pixel, m = np.broadcast_arrays(np.asarray(pixel, float), np.asarray(m, float))
        return self.m_lambda(pixel, m) / m

    def dispersion(self, pixel, m):
        return self.wavelength(np.asarray(pixel, float) + 0.5, m) - \
               self.wavelength(np.asarray(pixel, float) - 0.5, m)


# ======================================================================
# seeding
# ======================================================================

def seed_from_doublet(pixels, wavelengths, m, n_pixels, K=None):
    """Turn one clicked doublet in one order into a seed for EVERY order.

    Two lines of known wavelength at known pixels in order m give the local
    dispersion, and because m * lambda is a shared function of pixel, that
    single measurement fixes the linear seed for the whole detector:

        B = m * (dlambda/dy) * (n_pixels - 1) / 2      (slope, shared)
        A = m * lambda_1 - B * y_hat(pixel_1)          (zero point, shared)

    The Na D doublet is a good choice: 5.97 A apart, both deep, and the two
    components cannot be confused with each other. Fitting for them in a
    star as line-rich as Arcturus is harder than just pointing at them,
    which is why this takes clicked positions.

    K, if given, is only used to report how far the seed lands from the
    nominal grating constant -- a large disagreement means a misclick or
    the wrong order number, and is worth seeing before anything else runs.
    """
    pixels = np.asarray(pixels, float)
    wavelengths = np.asarray(wavelengths, float)
    if len(pixels) != 2 or len(wavelengths) != 2:
        raise ValueError("seed_from_doublet needs exactly two lines")

    order = np.argsort(wavelengths)
    pixels, wavelengths = pixels[order], wavelengths[order]
    d_pixel = pixels[1] - pixels[0]
    if abs(d_pixel) < 5:
        raise ValueError(f"clicked doublet is only {d_pixel:.1f} pixels apart -- "
                         f"that is almost certainly the same line clicked twice")

    dispersion = (wavelengths[1] - wavelengths[0]) / d_pixel   # A / pixel
    B = m * dispersion * (n_pixels - 1) / 2.0

    y_hat = 2.0 * (pixels - (n_pixels - 1) / 2.0) / (n_pixels - 1)
    A = float(np.mean(m * wavelengths - B * y_hat))

    print(f"seed_from_doublet: order m={m}, {abs(d_pixel):.1f} px between "
          f"{wavelengths[0]:.2f} and {wavelengths[1]:.2f} A")
    print(f"  -> dispersion {dispersion:.5f} A/px at m={m}; "
          f"m*lambda = {A:.0f} + {B:.0f} * y_hat")
    print(f"  -> implies {2 * B / m:.1f} A of coverage per order at m={m} "
          f"(free spectral range there is {A / m ** 2:.1f} A)")
    if K is not None:
        print(f"  -> nominal grating constant K = {K:.0f} A, seed zero point differs by "
              f"{A - K:+.0f} A ({100 * (A - K) / K:+.2f}%)")
    return LinearSeed(A, B, n_pixels)


def lock_seed(detections, order_numbers, reference, seed, n_pixels,
              zero_point_range=3000.0, slope_fraction=0.06, n_slope=121,
              grid_step=1.0, expected_sigma_pixels=3.3, verbose=True):
    """Refine the linear seed by correlating every order against the atlas
    simultaneously, and report how convincingly it locked.

    Why all orders at once. A single order carries perhaps thirty lines
    against an atlas with thousands of candidates in reach, so its
    correlation has plenty of near-ties -- this is exactly how a per-order
    search ends up confidently wrong, and one wrong order poisons anything
    fitted across orders. Here each order is correlated against its own
    atlas window (order m's lines live at m * lambda, which is different for
    every m, so this stays specific) and the correlations are SUMMED. The
    true zero point is the one place where all ~80 orders agree, so it wins
    by a margin nothing else can fake.

    The reported SNR -- peak height over the robust scatter of the summed
    correlation -- is the number to look at. Tens means locked. Single
    digits means the seed, the order numbers, or the atlas selection is
    wrong, and there is no point fitting anything until that is fixed.

    Returns (LinearSeed, snr).
    """
    y_hat = 2.0 * (np.arange(n_pixels) - (n_pixels - 1) / 2.0) / (n_pixels - 1)

    # observed line combs, one per order, in "m*lambda relative to A" space
    live = [(i, d) for i, d in enumerate(detections) if d is not None and len(d) >= 5]
    if not live:
        raise ValueError("no orders have enough detected ThAr lines to lock a seed")

    best = None
    slopes = seed.B * np.linspace(1.0 - slope_fraction, 1.0 + slope_fraction, n_slope)
    for B in slopes:
        u_lo = seed.A - zero_point_range - B
        u_hi = seed.A + zero_point_range + B
        u_obs = np.arange(-B, B + grid_step, grid_step)
        u_ref = np.arange(u_lo, u_hi + grid_step, grid_step)
        n_obs, n_ref = len(u_obs), len(u_ref)
        sigma_u = expected_sigma_pixels * 2.0 * B / (n_pixels - 1)
        half = max(int(4 * sigma_u / grid_step), 1)
        kernel = np.exp(-0.5 * (np.arange(-half, half + 1) * grid_step / sigma_u) ** 2)

        size = 1
        while size < n_obs + n_ref:
            size *= 2
        obs = np.zeros((len(live), size))
        ref_grid = np.zeros((len(live), size))

        for k, (i, d) in enumerate(live):
            m = float(order_numbers[i])
            # observed comb: unit-height marks at the detected line positions
            pos = np.interp(d[:, DET_PIXEL], np.arange(n_pixels), B * y_hat)
            idx = np.round((pos + B) / grid_step).astype(int)
            idx = idx[(idx >= 0) & (idx < n_obs)]
            row = np.zeros(n_obs)
            np.add.at(row, idx, 1.0)
            row = np.convolve(row, kernel, mode="same")
            if row.std() > 0:
                obs[k, :n_obs] = (row - row.mean()) / row.std()

            j0, j1 = np.searchsorted(reference.wave, [u_lo / m, u_hi / m])
            if j1 - j0 < 3:
                continue
            centres = reference.wave[j0:j1] * m
            amps = np.sqrt(reference.amplitude[j0:j1])
            jdx = np.round((centres - u_lo) / grid_step).astype(int)
            ok = (jdx >= 0) & (jdx < n_ref)
            row = np.zeros(n_ref)
            np.add.at(row, jdx[ok], amps[ok])
            row = np.convolve(row, kernel, mode="same")
            if row.std() > 0:
                ref_grid[k, :n_ref] = (row - row.mean()) / row.std()

        cc = np.fft.irfft(np.conj(np.fft.rfft(obs, size, axis=1))
                          * np.fft.rfft(ref_grid, size, axis=1), size, axis=1)
        total = cc.sum(axis=0)[:n_ref]
        lag = int(np.argmax(total))
        A = u_lo + lag * grid_step + B
        med = np.median(total)
        mad = 1.4826 * np.median(np.abs(total - med))
        snr = (total[lag] - med) / mad if mad > 0 else 0.0
        if best is None or snr > best[2]:
            best = (A, B, snr)

    A, B, snr = best
    if verbose:
        print(f"lock_seed: m*lambda = {A:.0f} + {B:.0f} * y_hat  "
              f"(seed was {seed.A:.0f} + {seed.B:.0f}); "
              f"zero point moved {A - seed.A:+.0f} A, dispersion "
              f"{100 * (B / seed.B - 1):+.2f}%")
        print(f"  correlation SNR = {snr:.1f} over {len(live)} orders "
              f"({'locked' if snr > 15 else 'WEAK -- do not trust this'})")
    return LinearSeed(A, B, n_pixels), snr


def check_order_number_offset(detections, order_numbers, reference, locked, n_pixels,
                              offsets=(-2, -1, 0, 1, 2), **lock_kwargs):
    """Confirm the order numbers by brute force: re-lock with m shifted by a
    constant and see which shift the atlas prefers.

    Absolute order number is the one thing overlap agreement cannot check --
    shift every m by one and a smooth surface will happily re-absorb it,
    producing a self-consistent solution with every wavelength wrong by
    roughly one free spectral range. Only the atlas can tell the difference,
    and it does so emphatically: the correct offset should win by a wide
    margin. If it does not, stop and fix the numbering.

    The zero-point search has to be wide here -- relabelling every order by
    one moves the best zero point by a whole order's worth of m*lambda, so a
    narrow window would never find the rival solution it is supposed to be
    comparing against. Widening it also lets each offset find its own
    aliases, which is the point: the true numbering is the only one where
    all eighty-odd orders agree on a single zero point, and the aliases
    cannot fake that.
    """
    print("Order-number check (re-locking with m shifted by a constant):")
    median_m = float(np.median([m for m in order_numbers if m is not None]))
    kwargs = dict(slope_fraction=0.04, n_slope=17,
                  zero_point_range=2.6 * locked.A / median_m)
    kwargs.update(lock_kwargs)
    results = {}
    for off in offsets:
        shifted = [m + off for m in order_numbers]
        _, snr = lock_seed(detections, shifted, reference, locked, n_pixels,
                           verbose=False, **kwargs)
        results[off] = snr
        print(f"  m {off:+d}: lock SNR = {snr:6.1f}{'   <-- current numbering' if off == 0 else ''}")
    best = max(results, key=results.get)
    runner_up = max(snr for off, snr in results.items() if off != 0)
    margin = results[0] / max(runner_up, 1e-9)
    if best != 0:
        print(f"  WARNING: offset {best:+d} locks better than the current numbering. "
              f"Either an anchor is wrong or an order is missing from the trace list. "
              f"Every wavelength is out by about one free spectral range until this "
              f"is fixed, and no internal check will notice.")
    elif margin >= 2.0:
        print(f"  current numbering wins by {margin:.1f}x -- "
              f"order numbers confirmed against the atlas.")
    else:
        print(f"  current numbering wins, but only by {margin:.1f}x. That is not a "
              f"confirmation: the atlas cannot tell these numberings apart, which "
              f"usually means the lock itself is poor.")
    return results, margin


# ======================================================================
# matching and fitting
# ======================================================================

class MatchSet:
    """Detected lines paired with reference wavelengths, ready to fit."""

    def __init__(self, pixel, m, m_lambda, weight, pixel_err, snr, order_index):
        self.pixel = pixel
        self.m = m
        self.m_lambda = m_lambda
        self.weight = weight
        self.pixel_err = pixel_err
        self.snr = snr
        self.order_index = order_index

    def __len__(self):
        return len(self.pixel)

    def subset(self, mask):
        return MatchSet(self.pixel[mask], self.m[mask], self.m_lambda[mask],
                        self.weight[mask], self.pixel_err[mask], self.snr[mask],
                        self.order_index[mask])


def match_lines(model, detections, order_numbers, reference, n_pixels,
                tolerance_pixels, ambiguity_factor=2.5, max_pixel_error=1.0):
    """Pair each detected line with a reference line, refusing anything
    ambiguous.

    A match is kept only if the nearest reference line is within
    `tolerance_pixels` AND the next-nearest is at least `ambiguity_factor`
    times the tolerance further away. Where the atlas is dense this throws
    away a lot of detections, which is the point: a match that could
    plausibly have been its neighbour contributes a wrong wavelength as
    readily as a right one, and no amount of sigma clipping downstream
    reliably finds those again once they are in the fit.

    The tolerance is specified in PIXELS and converted per order using the
    current model's local dispersion, so one number means the same thing in
    the blue and in the red.
    """
    pix, ms, mlam, wgt, perr, snr, oidx = [], [], [], [], [], [], []
    ref_w = reference.wave
    ref_eff = reference.eff_wave

    for i, det in enumerate(detections):
        if det is None or len(det) == 0:
            continue
        m = float(order_numbers[i])
        mu = det[:, DET_PIXEL]
        m_col = np.full(len(mu), m)
        lam = model.wavelength(mu, m_col)

        disp = np.abs(model.dispersion(mu, m_col))
        tol = tolerance_pixels * disp

        j = np.clip(np.searchsorted(ref_w, lam), 1, len(ref_w) - 1)
        cand = np.vstack([j - 1, j])
        dist = np.abs(ref_w[cand] - lam[None, :])
        which = np.argmin(dist, axis=0)
        cols = np.arange(len(lam))
        nearest = cand[which, cols]
        d1 = dist[which, cols]

        neigh = np.clip(np.vstack([nearest - 2, nearest - 1, nearest + 1, nearest + 2]),
                        0, len(ref_w) - 1)
        d2 = np.min(np.abs(ref_w[neigh] - lam[None, :]), axis=0)

        ok = (d1 < tol) & (d2 > ambiguity_factor * tol) & (det[:, DET_PIXERR] < max_pixel_error)
        if not ok.any():
            continue

        pix.append(mu[ok])
        ms.append(m_col[ok])
        mlam.append(ref_eff[nearest[ok]] * m)
        perr.append(det[ok, DET_PIXERR])
        snr.append(det[ok, DET_SNR])
        oidx.append(np.full(ok.sum(), i))
        # weight in m*lambda space: a pixel error becomes a wavelength error
        # through the local dispersion, and m*lambda scales it by m
        sigma_mlam = det[ok, DET_PIXERR] * disp[ok] * m
        wgt.append(1.0 / np.maximum(sigma_mlam, 1e-6) ** 2)

    if not pix:
        return MatchSet(*[np.array([]) for _ in range(7)])
    return MatchSet(np.concatenate(pix), np.concatenate(ms), np.concatenate(mlam),
                    np.concatenate(wgt), np.concatenate(perr), np.concatenate(snr),
                    np.concatenate(oidx))


FOCAL_LIMITS = (3000.0, 200000.0)   # plausible camera focal lengths, in pixels


def fit_solution(matches, n_pixels, m_degree=2, correction_degree=None,
                 focal_guess=None, clip_sigma=4.0, max_iterations=6):
    """Fit the physical surface (and its optional Chebyshev correction) to a
    set of matched lines, with robust rejection.

    Only the camera focal length is non-linear, so it is scanned on a grid
    with an exact weighted least-squares solve inside -- no optimiser to get
    stuck, no starting-value sensitivity.

    Returns (solution, keep_mask, residuals_in_m_lambda).
    """
    if len(matches) < 3 * (m_degree + 1) * 2:
        raise ValueError(f"only {len(matches)} matched lines -- not enough to fit")

    m_min, m_max = matches.m.min(), matches.m.max()
    # The focal length is the only non-linear parameter, and the only one
    # that can run away: as f grows the basis flattens into a straight line
    # and the Chebyshev correction can imitate whatever curvature was lost,
    # so with poor matches the fit happily wanders off to f = infinity and
    # throws away the very thing the physical basis was for. Bounded here,
    # and searched on a grid rather than by an optimiser so it cannot creep.
    keep = np.ones(len(matches), bool)
    solution = None
    residuals = np.zeros(len(matches))

    def best_focal(grid):
        best = None
        for f in grid:
            trial = WavelengthSolution(f, np.zeros(3 * (m_degree + 1)), m_degree,
                                       n_pixels, m_min, m_max)
            A = trial.design(matches.pixel[keep], matches.m[keep])
            sw = np.sqrt(matches.weight[keep])
            coef, *_ = np.linalg.lstsq(A * sw[:, None], matches.m_lambda[keep] * sw,
                                       rcond=None)
            r = matches.m_lambda[keep] - A @ coef
            chi = np.sqrt(np.average(r ** 2, weights=matches.weight[keep]))
            if best is None or chi < best[0]:
                best = (chi, f, coef)
        return best

    for _ in range(max_iterations):
        # Full range every time, then a local refinement. Walking f in from
        # wherever the previous pass left it is how it gets stranded at a
        # bound: the early passes have bad matches and prefer no curvature,
        # and a local search can never walk back from there.
        coarse = best_focal(np.geomspace(*FOCAL_LIMITS, 80))
        best = best_focal(np.clip(np.linspace(0.85 * coarse[1], 1.18 * coarse[1], 40),
                                  *FOCAL_LIMITS))
        if coarse[0] < best[0]:
            best = coarse

        solution = WavelengthSolution(best[1], best[2], m_degree, n_pixels, m_min, m_max)

        if correction_degree is not None:
            resid = matches.m_lambda - solution.m_lambda(matches.pixel, matches.m)
            V = C.chebvander2d(solution._y_hat(matches.pixel[keep]),
                               solution._m_hat(matches.m[keep]), correction_degree)
            sw = np.sqrt(matches.weight[keep])
            cc, *_ = np.linalg.lstsq(V * sw[:, None], resid[keep] * sw, rcond=None)
            solution.correction = cc.reshape(correction_degree[0] + 1,
                                             correction_degree[1] + 1)

        residuals = matches.m_lambda - solution.m_lambda(matches.pixel, matches.m)
        scatter = 1.4826 * np.median(np.abs(residuals[keep] - np.median(residuals[keep])))
        new_keep = np.abs(residuals) < clip_sigma * max(scatter, 1e-12)
        if new_keep.sum() == keep.sum() and np.array_equal(new_keep, keep):
            break
        keep = new_keep
        if keep.sum() < 3 * (m_degree + 1) * 3:
            break

    return solution, keep, residuals


def _rms_angstrom(matches, residuals, mask):
    return float(np.sqrt(np.mean((residuals[mask] / matches.m[mask]) ** 2)))


def default_schedule(correction_degree=(4, 2)):
    """The tightening ladder for the physical-model passes. Two uncorrected
    passes first so the camera geometry is fitted to the lines alone, then
    the correction comes in and the tolerance closes to two pixels."""
    return [(4.0, None), (3.0, None),
            (2.5, correction_degree), (2.0, correction_degree),
            (2.0, correction_degree), (2.0, correction_degree)]


def solve(orders, detections, reference, seed, n_pixels,
          schedule=None, warmup=None, m_degree=2, correction_degree=None,
          verbose=True):
    """Go from a linear seed to the final solution.

    The schedule alternates matching and fitting, starting loose and
    tightening. Two details that matter more than they look:

      * the model is refitted before the tolerance shrinks, never after, so
        every tightening step is applied to a model that has already
        improved;
      * the physical basis is used from the second pass onward. That is what
        lets matching reach the ends of each order. ThAr lines are matched
        first wherever they are densest, and a polynomial fitted only there
        misbehaves outside that range, so the ends never get matched and
        never get fitted -- the failure feeds itself, and it shows up
        precisely as neighbouring orders disagreeing where they overlap,
        because overlaps live at the ends. A model made of the actual optics
        predicts the ends correctly from the middle, so the lines there get
        matched on the next pass.

    Returns (solution, matches, keep_mask, residuals).
    """
    if warmup is None:
        # Tolerance comes down slowly and the degree goes up slowly. Cutting
        # either corner strands the fit: too big a drop in tolerance throws
        # away the lines at the ends of each order before the model is good
        # enough to reach them, and too high a degree too early lets the
        # polynomial chase the badly-matched ones.
        warmup = [(30.0, (1, 0)), (15.0, (2, 1)), (8.0, (3, 1)),
                  (5.0, (4, 2)), (4.0, (4, 2))]
    if schedule is None:
        schedule = default_schedule((4, 2) if correction_degree is None
                                    else correction_degree)

    order_numbers = [o.order_number for o in orders]
    model = seed
    solution = matches = keep = residuals = None

    def report(tag, tol, extra):
        rms = _rms_angstrom(matches, residuals, keep)
        lam = matches.m_lambda[keep] / matches.m[keep]
        velocity = np.sqrt(np.mean(((residuals[keep] / matches.m[keep]) / lam
                                    * C_LIGHT_MS) ** 2))
        print(f"  {tag}: tol={tol:4.1f} px  {extra:16s}  matched={len(matches):5d}  "
              f"kept={keep.sum():5d}  rms={rms * 1000:6.2f} mA ({velocity:6.0f} m/s)")

    # --- warm-up on a plain polynomial surface -------------------------
    # The physical basis must not be fitted to bad matches: its curvature
    # is only pinned down by lines that are actually right, and given wrong
    # ones it prefers a huge focal length (i.e. no curvature at all), which
    # then has to be faked by the correction term and destroys exactly the
    # extrapolation the basis was chosen for. So: get the matches honest
    # first with something that cannot be led astray, then hand over.
    for step, (tolerance, degrees) in enumerate(warmup):
        matches = match_lines(model, detections, order_numbers, reference, n_pixels,
                              tolerance)
        if len(matches) < 60:
            raise RuntimeError(
                f"only {len(matches)} lines matched at tolerance {tolerance} px. "
                f"The seed is not close enough to the truth -- check the lock SNR, the "
                f"order numbers, and the clicked doublet before anything else.")
        solution, keep, residuals = fit_polynomial_surface(matches, n_pixels, degrees)
        model = solution
        if verbose:
            report(f"warm-up {step + 1}", tolerance, f"cheb{degrees}")

    # --- the real fit ---------------------------------------------------
    focal_guess = None
    for step, (tolerance, correction_degree) in enumerate(schedule):
        matches = match_lines(model, detections, order_numbers, reference, n_pixels,
                              tolerance)
        solution, keep, residuals = fit_solution(
            matches, n_pixels, m_degree=m_degree,
            correction_degree=correction_degree, focal_guess=focal_guess)
        focal_guess = solution.focal_pixels
        model = solution
        if verbose:
            report(f"pass    {step + 1}", tolerance,
                   f"corr={correction_degree}")

    if verbose:
        f = solution.focal_pixels
        if FOCAL_LIMITS[0] * 1.02 < f < FOCAL_LIMITS[1] * 0.98:
            note = "inside the plausible range"
        else:
            note = ("AT A BOUND -- the physical basis is not constraining anything "
                    "here, so treat this as a plain polynomial fit and do not trust "
                    "it beyond the pixels that were matched")
        print(f"  camera focal length fitted at {f:.0f} px ({note})")
    return solution, matches, keep, residuals


# ======================================================================
# quality: does this solution deserve to be believed
# ======================================================================

class QualityReport:
    """Everything measured about a solution, and whether it passes.

    Kept as an object rather than printed and forgotten so the driver can
    refuse to save a bad solution -- the failure mode worth designing
    against is a plausible-looking wavelength axis that is quietly wrong.
    """

    def __init__(self):
        self.checks = []      # (name, passed, message)
        self.stats = {}

    def add(self, name, passed, message):
        self.checks.append((name, bool(passed), message))

    @property
    def passed(self):
        return all(p for _, p, _ in self.checks)

    def show(self):
        print("\n" + "=" * 72)
        print("WAVELENGTH SOLUTION QUALITY")
        print("=" * 72)
        for name, ok, message in self.checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:24s} {message}")
        print("-" * 72)
        print(f"  OVERALL: {'PASS' if self.passed else 'FAIL'}")
        print("=" * 72 + "\n")


def cross_validate(matches, keep, n_pixels, m_degree, correction_degree,
                   folds=5, seed=0):
    """Hold out a fifth of the lines, fit on the rest, predict the held-out
    ones -- repeated over all five folds.

    The RMS of a fit against its own training lines always improves with
    more free parameters, so it cannot tell you whether the extra freedom is
    describing the instrument or the noise. This can. It is also the number
    to quote: it is what the solution does to a line it has never seen,
    which is what it will do to your science spectrum.
    """
    sub = matches.subset(keep)
    idx = np.arange(len(sub))
    np.random.default_rng(seed).shuffle(idx)
    errors = []
    for f in range(folds):
        test = idx[f::folds]
        train = np.setdiff1d(idx, test)
        try:
            sol, _, _ = fit_solution(sub.subset(train), n_pixels, m_degree=m_degree,
                                     correction_degree=correction_degree)
        except ValueError:
            continue
        pred = sol.m_lambda(sub.pixel[test], sub.m[test])
        errors.append((sub.m_lambda[test] - pred) / sub.m[test])
    if not errors:
        return np.nan
    e = np.concatenate(errors)
    scatter = 1.4826 * np.median(np.abs(e - np.median(e)))
    e = e[np.abs(e) < 5 * scatter]          # a fold can leave a gap it must extrapolate into
    return float(np.sqrt(np.mean(e ** 2)))


def choose_degrees(matches, keep, n_pixels,
                   m_degrees=(1, 2, 3), correction_degrees=(None, (2, 1), (3, 2), (4, 2), (5, 3)),
                   folds=5, tolerance=0.05, verbose=True):
    """Pick the model complexity by cross-validation rather than by taste.

    Among models that cross-validate within `tolerance` of the best, the one
    with the fewest free parameters wins. Extra degrees that buy nothing
    measurable are not free: they are what lets a fit wander in the pixel
    ranges where lines are sparse, which is exactly where you cannot see it
    happening.
    """
    if verbose:
        print(f"Choosing model complexity by {folds}-fold cross-validation:")
    results = []
    for md in m_degrees:
        for cd in correction_degrees:
            try:
                cv = cross_validate(matches, keep, n_pixels, md, cd, folds=folds)
            except Exception:
                continue
            if not np.isfinite(cv):
                continue
            n_free = 3 * (md + 1) + (0 if cd is None else (cd[0] + 1) * (cd[1] + 1))
            results.append((cv, n_free, md, cd))
            if verbose:
                print(f"    m_degree={md}  correction={str(cd):8s} "
                      f"({n_free:2d} free) -> cross-validated rms {cv * 1000:6.2f} mA")
    if not results:
        raise RuntimeError("cross-validation failed for every model tried")
    best_cv = min(r[0] for r in results)
    within = [r for r in results if r[0] <= best_cv * (1.0 + tolerance)]
    cv, n_free, md, cd = min(within, key=lambda r: (r[1], r[0]))
    if verbose:
        print(f"  -> m_degree={md}, correction={cd}: cross-validated rms "
              f"{cv * 1000:.2f} mA with {n_free} free parameters "
              f"(best was {best_cv * 1000:.2f} mA; the simplest model within "
              f"{tolerance:.0%} of it wins)")
    return md, cd, cv


def overlap_agreement(orders, solution, spectrum_attr="thar_spectrum",
                      pixel_shift=0.0, velocity_ms=0.0,
                      min_overlap_angstrom=2.0, max_velocity_ms=20000.0,
                      oversample=5.0, min_contrast=3.0):
    """Measure whether adjacent orders agree in wavelength space, using the
    ThAr spectra themselves and no atlas at all.

    Where two orders overlap they observe the same lamp lines on different
    parts of the detector. Put both on a common log-wavelength grid and
    cross-correlate: the offset of the peak from zero is the disagreement,
    in velocity, and it is completely independent of the line list.

    This is the check that answers "do the orders line up". A solution can
    have a small residual against the atlas and still be wrong here if the
    order-to-order behaviour is off. In practice a global surface makes this
    nearly structural -- which is the argument for fitting one -- so a
    failure means something more basic is broken.

    Returns (velocities, pairs) for the pairs that overlap. Orders red of
    m ~ A / (2 * B) do not overlap at all -- their free spectral range is
    wider than the detector -- and are silently skipped.
    """
    velocities, pairs = [], []
    usable = [o for o in orders if o.order_number is not None
              and getattr(o, spectrum_attr, None) is not None]
    usable.sort(key=lambda o: -o.order_number)

    def axis_of(order, n):
        pixels = np.arange(n) - pixel_shift
        w = solution.wavelength(pixels, np.full(n, float(order.order_number)))
        return w / (1.0 + velocity_ms / C_LIGHT_MS)

    for a, b in zip(usable[:-1], usable[1:]):
        if abs(a.order_number - b.order_number) != 1:
            continue
        spec_a = getattr(a, spectrum_attr)
        spec_b = getattr(b, spectrum_attr)
        n = len(spec_a)
        wa = axis_of(a, n)
        wb = axis_of(b, n)
        lo = max(wa.min(), wb.min())
        hi = min(wa.max(), wb.max())
        if hi - lo < min_overlap_angstrom:
            continue

        # Log-wavelength grid, so one lag is one constant velocity, sampled
        # several times per detector pixel: quantising this at the pixel
        # scale would report "they agree to within a pixel" and no more,
        # which is far too blunt to be a check on anything.
        pixel_step = np.median(np.abs(np.diff(wa)))
        step = pixel_step / hi / oversample
        grid = np.arange(np.log(lo), np.log(hi), step)
        if len(grid) < 128:
            continue
        fa = np.interp(np.exp(grid), wa, spec_a)
        fb = np.interp(np.exp(grid), wb, spec_b)
        fa = fa - median_filter(fa, max(9, len(fa) // 8) | 1)
        fb = fb - median_filter(fb, max(9, len(fb) // 8) | 1)
        if fa.std() <= 0 or fb.std() <= 0:
            continue
        fa /= fa.std()
        fb /= fb.std()

        max_lag = int(min(len(grid) // 3, max_velocity_ms / C_LIGHT_MS / step))
        if max_lag < 5:
            continue
        lags = np.arange(-max_lag, max_lag + 1)
        cc = np.array([np.dot(fa[max(0, l):len(fa) + min(0, l)],
                              fb[max(0, -l):len(fb) + min(0, -l)]) /
                       max(len(fa) - abs(l), 1) for l in lags])
        if not np.isfinite(cc).all():
            continue
        k = int(np.argmax(cc))
        if not (0 < k < len(cc) - 1):
            continue                      # peak is outside the search range
        # a peak that does not stand out means these two orders share no
        # lines worth correlating -- reporting a number for it is worse
        # than reporting nothing
        scatter = 1.4826 * np.median(np.abs(cc - np.median(cc)))
        if scatter <= 0 or (cc[k] - np.median(cc)) / scatter < min_contrast:
            continue
        denom = cc[k - 1] - 2 * cc[k] + cc[k + 1]
        sub = 0.5 * (cc[k - 1] - cc[k + 1]) / denom if denom != 0 else 0.0
        velocities.append((lags[k] + sub) * step * C_LIGHT_MS)
        pairs.append((a.order_number, b.order_number))

    return np.array(velocities), pairs


def residual_trends(matches, keep, residuals, n_pixels, n_bins=8):
    """Bin the residuals against pixel and against order number.

    A correct model leaves residuals with no structure. Systematic drift
    with pixel means the dispersion shape is underfitted; drift with order
    number means the cross-order term is. Both are things extra degrees can
    fix, which is why this is reported next to the cross-validation rather
    than instead of it.

    Returns (pixel_bins, order_bins), each a list of (label, n, mean_mA, rms_mA).
    """
    r = residuals[keep] / matches.m[keep] * 1000.0
    p = matches.pixel[keep]
    m = matches.m[keep]

    pixel_bins = []
    edges = np.linspace(0, n_pixels, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (p >= lo) & (p < hi)
        if s.sum() > 5:
            pixel_bins.append((f"{int(lo)}-{int(hi)}", int(s.sum()),
                               float(r[s].mean()), float(np.sqrt(np.mean(r[s] ** 2)))))

    order_bins = []
    m_edges = np.linspace(m.min(), m.max() + 1e-9, n_bins + 1)
    for lo, hi in zip(m_edges[:-1], m_edges[1:]):
        s = (m >= lo) & (m < hi)
        if s.sum() > 5:
            order_bins.append((f"m={int(lo)}-{int(hi)}", int(s.sum()),
                               float(r[s].mean()), float(np.sqrt(np.mean(r[s] ** 2)))))
    return pixel_bins, order_bins


def assess(orders, solution, matches, keep, residuals, n_pixels, lock_snr,
           m_degree, correction_degree, order_number_margin=None,
           max_rms_ma=15.0, max_cv_ma=20.0, max_overlap_ms=600.0,
           min_orders_with_lines=0.6, min_pixel_coverage=0.75,
           max_trend_ma=6.0, min_lock_snr=15.0, min_order_number_margin=2.0,
           verbose=True):
    """Run every check and return a QualityReport.

    The thresholds are defaults, not laws -- set them from what this
    instrument actually achieves. What matters is that they exist and that
    the driver refuses to save a solution that fails them, so a bad night
    announces itself instead of quietly producing a wavelength axis that
    looks fine on a plot.
    """
    report = QualityReport()
    lam = matches.m_lambda[keep] / matches.m[keep]
    resid_ang = residuals[keep] / matches.m[keep]
    rms = float(np.sqrt(np.mean(resid_ang ** 2)))
    velocity = float(np.sqrt(np.mean((resid_ang / lam * C_LIGHT_MS) ** 2)))

    report.stats["n_matched"] = int(keep.sum())
    report.stats["rms_angstrom"] = rms
    report.stats["rms_ms"] = velocity
    report.stats["focal_pixels"] = solution.focal_pixels

    report.add("atlas lock", lock_snr >= min_lock_snr,
               f"correlation SNR {lock_snr:.1f} (need >= {min_lock_snr:.0f})")

    if order_number_margin is not None:
        report.stats["order_number_margin"] = order_number_margin
        report.add("order numbers", order_number_margin >= min_order_number_margin,
                   f"the assigned numbering locks {order_number_margin:.1f}x better than "
                   f"m +/-1 or +/-2 (need >= {min_order_number_margin:.0f}x)")

    report.add("line residuals", rms * 1000 <= max_rms_ma,
               f"{keep.sum()} lines, rms {rms * 1000:.2f} mA = {velocity:.0f} m/s "
               f"(need <= {max_rms_ma:.0f} mA)")

    cv = cross_validate(matches, keep, n_pixels, m_degree, correction_degree)
    report.stats["cv_angstrom"] = cv
    report.add("cross-validation", np.isfinite(cv) and cv * 1000 <= max_cv_ma,
               f"held-out rms {cv * 1000:.2f} mA "
               f"({cv / max(rms, 1e-12):.2f}x the fitted rms; "
               f"a large ratio means over-fitting)")

    # how much of each order is actually constrained by matched lines
    coverages = []
    per_order = {}
    for order in orders:
        if order.order_number is None:
            continue
        s = keep & (matches.m == order.order_number)
        if s.sum() >= 4:
            span = matches.pixel[s].max() - matches.pixel[s].min()
            coverages.append(span / n_pixels)
            per_order[order.order_number] = (
                int(s.sum()),
                float(np.sqrt(np.mean((residuals[s] / order.order_number) ** 2)) * 1000),
                span / n_pixels)
    report.stats["per_order"] = per_order

    n_with = len(per_order)
    n_total = sum(1 for o in orders if o.order_number is not None)
    report.add("order coverage", n_with >= min_orders_with_lines * n_total,
               f"{n_with}/{n_total} orders carry >= 4 matched lines "
               f"(need >= {min_orders_with_lines:.0%})")

    median_cov = float(np.median(coverages)) if coverages else 0.0
    report.stats["median_pixel_coverage"] = median_cov
    report.add("pixel coverage", median_cov >= min_pixel_coverage,
               f"matched lines span {median_cov:.0%} of the detector in the median order "
               f"(need >= {min_pixel_coverage:.0%}; less means the ends are extrapolated, "
               f"and the ends are where orders overlap)")

    pixel_bins, order_bins = residual_trends(matches, keep, residuals, n_pixels)
    worst_pixel = max((abs(b[2]) for b in pixel_bins), default=0.0)
    worst_order = max((abs(b[2]) for b in order_bins), default=0.0)
    report.stats["pixel_bins"] = pixel_bins
    report.stats["order_bins"] = order_bins
    report.add("residual trends", max(worst_pixel, worst_order) <= max_trend_ma,
               f"largest binned mean residual {max(worst_pixel, worst_order):.2f} mA "
               f"vs pixel/order (need <= {max_trend_ma:.0f} mA; structure here means "
               f"the model is too stiff)")

    velocities, pairs = overlap_agreement(orders, solution)
    if len(velocities):
        med = float(np.median(np.abs(velocities)))
        worst = float(np.max(np.abs(velocities)))
        report.stats["overlap_ms"] = velocities
        report.stats["overlap_pairs"] = pairs
        report.add("order overlap", med <= max_overlap_ms,
                   f"{len(velocities)} overlapping pairs agree to {med:.0f} m/s median, "
                   f"{worst:.0f} m/s worst (need <= {max_overlap_ms:.0f} m/s)")
    else:
        report.add("order overlap", False,
                   "no adjacent orders overlap in wavelength -- cannot check")

    if verbose:
        report.show()
        print("Residuals by pixel:")
        for label, n, mean, rms_b in pixel_bins:
            print(f"    {label:>12s}  n={n:4d}  mean={mean:+7.2f} mA  rms={rms_b:6.2f} mA")
        print("Residuals by order number:")
        for label, n, mean, rms_b in order_bins:
            print(f"    {label:>12s}  n={n:4d}  mean={mean:+7.2f} mA  rms={rms_b:6.2f} mA")
        if len(velocities):
            print(f"Adjacent-order overlap (independent of the atlas): "
                  f"median {np.median(np.abs(velocities)):.0f} m/s over {len(velocities)} pairs")
            worst_idx = np.argsort(-np.abs(velocities))[:5]
            for k in worst_idx:
                print(f"    m={pairs[k][0]}/{pairs[k][1]}: {velocities[k]:+8.0f} m/s")

        thin = [(m, v) for m, v in sorted(per_order.items()) if v[0] < 8 or v[2] < 0.5]
        worst = sorted(per_order.items(), key=lambda kv: -kv[1][1])[:6]
        print("Weakest orders (few lines, or lines covering little of the order):")
        if thin:
            for m, (n, rms_o, cov) in thin:
                print(f"    m={m:4d}  {n:3d} lines  rms {rms_o:6.2f} mA  covering {cov:4.0%} "
                      f"of the order -- {solution.order_axis(m)[0]:.0f}-"
                      f"{solution.order_axis(m)[-1]:.0f} A")
        else:
            print("    none -- every order has 8+ lines spanning at least half of it")
        print("Highest per-order residuals:")
        for m, (n, rms_o, cov) in worst:
            print(f"    m={m:4d}  {n:3d} lines  rms {rms_o:6.2f} mA  covering {cov:4.0%}")
        missing = [o.order_number for o in orders
                   if o.order_number is not None and o.order_number not in per_order]
        if missing:
            print(f"Orders with too few matched lines to check ({len(missing)}): "
                  f"{sorted(missing)}")
            print("    these still get a wavelength axis from the global surface, which is "
                  "the point of fitting one -- but nothing in their own data confirms it")
    return report


def measure_frame_shift(anchors, solution, verbose=True):
    """Compare stellar lines of known wavelength against the solution, and
    report the pixel shift between the frame they were measured in and the
    ThAr frame the solution was built from.

    This does double duty. It is an independent check -- lines in different
    orders, from a different exposure, that never touched the fit -- and if
    they all agree on one shift, the solution is right in a way no internal
    residual can demonstrate. It is also the correction: a science exposure
    taken hours from its arc has moved, and applying the arc's wavelength
    axis to it unshifted puts every line in the wrong place by that amount.

    anchors : list of (order_number, measured_pixel, rest_wavelength).
        The clicked Na D pair is two of these; Halpha and Hbeta, if you
        have their pixels, are two more in two other orders, which is what
        makes the agreement meaningful.

    What it does NOT tell you is which of two very different things the
    offset is, and that distinction decides how it may be applied:

      * the spectrum has moved across the detector (flexure). Every order
        moves by the same number of PIXELS, and evaluating the arc solution
        at (pixel - shift) undoes it exactly.
      * the light is Doppler shifted -- the target's radial velocity, the
        Earth's motion, anything that scales with wavelength. Every order
        moves by the same FRACTION of its wavelength, which is a different
        number of pixels in each order and at each position along it.

    Correcting one as though it were the other is not a small error. A
    pixel shift applied to a Doppler offset leaves each order wrong by
    shift x (its own dispersion), and dispersion goes as 1/m, so adjacent
    orders end up disagreeing by roughly shift x d(lambda)/dy / m -- worst
    in the red, shrinking towards the blue. Use diagnose_frame_offset() to
    find out which one you have before applying anything.

    Returns (shift_pixels, scatter_pixels, per-anchor list).
    """
    rows = []
    for m, pixel, wave in anchors:
        axis = solution.order_axis(m)
        if not (axis.min() < wave < axis.max()):
            print(f"  {wave:.2f} A is outside order m={m} ({axis.min():.1f}-"
                  f"{axis.max():.1f} A) -- skipping this anchor")
            continue
        predicted = float(np.interp(wave, axis, np.arange(len(axis)))
                          if axis[0] < axis[-1] else
                          np.interp(wave, axis[::-1], np.arange(len(axis))[::-1]))
        disp = float(np.abs(solution.dispersion(predicted, m)))
        shift = pixel - predicted
        rows.append((m, wave, pixel, predicted, shift, shift * disp / wave * C_LIGHT_MS))

    if not rows:
        return np.nan, np.nan, rows

    shifts = np.array([r[4] for r in rows])
    shift = float(np.median(shifts))
    scatter = float(np.max(np.abs(shifts - shift))) if len(shifts) > 1 else 0.0

    if verbose:
        print("Stellar-line check (lines that took no part in the fit):")
        for m, wave, pixel, predicted, s, v in rows:
            print(f"  m={int(m):4d}  {wave:9.2f} A  measured px {pixel:8.2f}  "
                  f"solution px {predicted:8.2f}  ->  {s:+7.2f} px ({v / 1000:+7.1f} km/s)")
        velocity = float(np.median([r[5] for r in rows]))
        print(f"  median offset {shift:+.2f} px = {velocity / 1000:+.1f} km/s; "
              f"anchors agree to within {scatter:.2f} px")
        if scatter > 3.0:
            print("  WARNING: the anchors do NOT agree on one shift. That is not "
                  "flexure -- suspect a misidentified line or a wrong order number.")
        elif abs(shift) > 5.0:
            print("  Do not apply this as a pixel shift until diagnose_frame_offset() "
                  "says it is flexure -- see that function.")
    return shift, scatter, rows


def diagnose_frame_offset(orders, solution, shift_pixels, spectrum_attr="science_spectrum",
                          verbose=True):
    """Decide whether an exposure's offset from the arc is a movement across
    the detector or a Doppler shift, by asking the orders themselves.

    The two are trivially separable if you look in the right place. Adjacent
    orders see the same wavelengths where they overlap, so whichever
    correction is right will leave them agreeing there and the wrong one
    will pull them apart -- and the wrong one pulls them apart hard, because
    a pixel is worth a different amount of wavelength in every order.

      * flexure: the orders agree once the pixel shift is applied, and
        disagree without it.
      * Doppler: the orders agree WITHOUT any correction (a Doppler shift
        moves every order by the same fraction, so they still line up), and
        a pixel shift breaks that agreement.

    A Doppler offset must not be removed with a pixel shift, and in general
    should not be removed from the wavelength axis at all: the arc solution
    already gives observed wavelengths, and the target's velocity is
    something to measure from the data, not to calibrate out of it.

    Returns a dict of the two measurements plus a verdict string.
    """
    uncorrected, _ = overlap_agreement(orders, solution, spectrum_attr=spectrum_attr)
    shifted, _ = overlap_agreement(orders, solution, spectrum_attr=spectrum_attr,
                                   pixel_shift=shift_pixels)
    if len(uncorrected) == 0 or len(shifted) == 0:
        if verbose:
            print("diagnose_frame_offset: not enough overlapping orders to tell")
        return {"verdict": "unknown", "uncorrected_ms": np.nan, "shifted_ms": np.nan}

    a = float(np.median(np.abs(uncorrected)))
    b = float(np.median(np.abs(shifted)))
    verdict = "doppler" if a < b else "flexure"

    if verbose:
        print(f"diagnose_frame_offset ({spectrum_attr}, offset {shift_pixels:+.1f} px):")
        print(f"  adjacent orders agree to {a:7.0f} m/s with NO correction")
        print(f"  adjacent orders agree to {b:7.0f} m/s with the pixel shift applied")
        if verdict == "doppler":
            print(f"  -> Doppler-like, by a factor of {b / max(a, 1e-9):.0f}. The light is "
                  f"shifted in wavelength, not moved across the detector, so applying "
                  f"this as a pixel shift would break the order overlap rather than "
                  f"fix anything. Leave the axis alone; the offset is a velocity to be "
                  f"measured, not calibrated away.")
        else:
            print(f"  -> flexure, by a factor of {a / max(b, 1e-9):.0f}. The spectrum has "
                  f"moved on the detector, and the pixel shift is the right correction.")
    return {"verdict": verdict, "uncorrected_ms": a, "shifted_ms": b}


# ======================================================================
# applying the solution to the orders
# ======================================================================

def attach_solution(orders, solution, pixel_shift=0.0, velocity_ms=0.0, quiet=False):
    """Give every order a callable pixel -> wavelength.

    Both corrections default to zero, which is the arc frame: observed
    wavelengths, exactly as measured, and what gets saved as the master.
    Only move away from it for a reason diagnose_frame_offset supports.
    """
    n = 0
    for order in orders:
        if order.order_number is None:
            continue
        order.wavelength_poly = OrderWavelength(solution, order.order_number,
                                                pixel_shift=pixel_shift,
                                                velocity_ms=velocity_ms)
        n += 1
    if not quiet:
        extra = ""
        if pixel_shift:
            extra += f" (shifted {pixel_shift:+.2f} px)"
        if velocity_ms:
            extra += f" (rest frame, {velocity_ms / 1000:+.1f} km/s removed)"
        print(f"attach_solution: wavelength axis attached to {n} orders{extra}")


def store_matches(orders, matches, keep):
    """Record which lines each order was calibrated with (for plots and for
    the saved master solution)."""
    for order in orders:
        if order.order_number is None:
            continue
        s = keep & (matches.m == order.order_number)
        order.thar_pixels = matches.pixel[s].tolist()
        order.thar_wavelengths = (matches.m_lambda[s] / matches.m[s]).tolist()


# ======================================================================
# save / reuse
# ======================================================================

class OrderIdentifier:
    """Turns a position across the detector into a physical order number.

    Order spacing varies smoothly across the detector (76 px between the
    bluest orders here, 30 px between the reddest), so position -> order
    number is a smooth monotonic curve. Fitting it does two things a lookup
    table cannot: it tolerates the small cross-dispersion drift between
    nights, and it extends past the orders the master itself traced, so a
    later night that picks up an extra faint order at either end still gets
    the right number for it.
    """

    def __init__(self, trace_x, order_number, degree=4):
        self.trace_x = np.asarray(trace_x, float)
        self.order_number = np.asarray(order_number, float)
        order = np.argsort(self.trace_x)
        self.trace_x = self.trace_x[order]
        self.order_number = self.order_number[order]
        degree = min(degree, max(1, len(self.trace_x) - 2))
        self.coefficients = np.polyfit(self.trace_x, self.order_number, degree)
        residual = self.order_number - np.polyval(self.coefficients, self.trace_x)
        self.max_residual = float(np.max(np.abs(residual)))

    def __call__(self, x):
        """Order number (not rounded) at cross-dispersion position x."""
        return np.polyval(self.coefficients, np.asarray(x, float))

    def spacing_at(self, x):
        """Local spacing between orders, in pixels, at position x."""
        slope = np.polyval(np.polyder(self.coefficients), np.asarray(x, float))
        return np.abs(1.0 / np.where(np.abs(slope) < 1e-12, np.nan, slope))


def save_solution(path, solution, orders, report=None, white=None, atlas_path=None):
    """Save a master solution.

    The thing that makes this reusable on a night whose traces are not the
    same list is what identifies an order. Not its index -- a faint order
    the tracer misses shifts every index after it, and nothing downstream
    can tell. Not even its order number by itself, since that has to come
    from somewhere. What is saved instead is where each order physically
    sits across the detector, plus the white-light cross-section it was
    measured from. An order's position is set by the optics, so on any
    later night a trace found at that position IS that order, however many
    orders were or were not traced around it.

    Stored: the m*lambda surface; per order its number, spatial position,
    matched lines and reference arc spectrum; and the spatial map (the
    white-light profile, the row it came from, and the position -> order
    number relation with the polynomial that lets it extrapolate to orders
    the master itself never traced).

    A wavelength image, lambda(x, y), would be the other way to be
    index-independent, and is worse on every count: sixty-odd megabytes
    instead of a few hundred numbers, not smooth across order boundaries
    so it cannot be interpolated or extended, and it would still have to be
    registered against the new night's traces before it could be used. The
    registration is the actual content, so store that.
    """
    numbered = [o for o in orders if o.order_number is not None
                and o.trace_center_pixel is not None]
    x = np.array([o.trace_center_pixel for o in numbered], float)
    m = np.array([o.order_number for o in numbered], float)
    identifier = OrderIdentifier(x, m)

    spatial = {
        "trace_x": x,
        "order_number": m,
        "identifier": identifier,
        "profile": None,
        "profile_row": None,
    }
    if white is not None:
        row = white.shape[0] // 2
        spatial["profile"] = np.asarray(white[row, :], float)
        spatial["profile_row"] = int(row)

    payload = {
        "solution": solution,
        "spatial": spatial,
        "atlas_path": atlas_path,
        "orders": [
            {
                "order_number": o.order_number,
                "trace_center_pixel": o.trace_center_pixel,
                "thar_pixels": o.thar_pixels,
                "thar_wavelengths": o.thar_wavelengths,
                "reference_thar_spectrum": o.thar_spectrum,
            }
            for o in numbered
        ],
        "quality": None if report is None else {"checks": report.checks,
                                                "stats": report.stats},
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"save_solution: wrote {len(payload['orders'])} orders to {path}")
    print(f"  spatial map: orders identified by position across the detector "
          f"(x = {x.min():.0f} to {x.max():.0f} px, m = {int(m.max())} to {int(m.min())})"
          + ("" if spatial["profile"] is None
             else f", white-light profile from row {spatial['profile_row']} saved for registration"))


def load_solution(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ======================================================================
# interactive helpers
# ======================================================================

def refine_line_at_guess(spectrum, guess, window=15, kind="absorption"):
    """Snap an approximate pixel to the sub-pixel centroid of the nearest
    line, by fitting a Gaussian in a window around it."""
    s = np.asarray(spectrum, float)
    guess = int(round(guess))
    lo = max(0, guess - window)
    hi = min(len(s), guess + window)
    x = np.arange(lo, hi)
    y = s[lo:hi]
    baseline = np.median(y)
    extreme = y.min() if kind == "absorption" else y.max()
    try:
        popt, _ = curve_fit(_gaussian, x, y,
                            p0=(extreme - baseline, guess, max(2.0, window / 4), baseline))
        if lo < popt[1] < hi:
            return float(popt[1])
    except Exception:
        pass
    print(f"  Gaussian refine failed near pixel {guess}; using the local extremum")
    return float(x[np.argmin(y)] if kind == "absorption" else x[np.argmax(y)])


def click_line(spectrum, title, window=15, kind="absorption"):
    """Show a spectrum, take one click, return the refined line centre.

    Clicking beats fitting for this: in a spectrum as line-rich as Arcturus
    an automatic search has no way to know which deep line is the one you
    meant, and the whole seed rests on getting that right. You only need to
    land inside the window; the Gaussian does the rest.
    """
    spectrum = np.asarray(spectrum, float)
    fig = plt.figure(figsize=(13, 4))
    plt.plot(spectrum, lw=0.7)
    plt.title(f"{title}  (click on the line; close the window to skip)")
    plt.xlabel("pixel")
    plt.tight_layout()
    points = plt.ginput(1, timeout=0)
    plt.close(fig)
    if not points:
        print(f"  no click for {title}")
        return None
    pixel = refine_line_at_guess(spectrum, points[0][0], window=window, kind=kind)
    print(f"  {title}: clicked {points[0][0]:.0f} -> centroid {pixel:.2f}")
    return pixel


def plot_calibrated_orders(orders, spectrum_attr="science_spectrum", title=None,
                           mark_lines=True):
    """Every calibrated order on a common wavelength axis. Overlapping
    orders should lie on top of each other; that is the eyeball version of
    the overlap check."""
    plt.figure(figsize=(15, 6))
    n = 0
    for order in orders:
        if order.wavelength_poly is None:
            continue
        spectrum = getattr(order, spectrum_attr, None)
        if spectrum is None:
            continue
        pix = np.arange(len(spectrum))
        wave = order.wavelength_poly(pix)
        peak = np.nanmax(spectrum)
        if not np.isfinite(peak) or peak <= 0:
            continue
        plt.plot(wave, spectrum / peak, lw=0.6)
        n += 1
        if mark_lines and order.thar_pixels:
            plt.plot(order.wavelength_poly(np.array(order.thar_pixels)),
                     np.full(len(order.thar_pixels), 1.05), "kx", ms=3)
    plt.xlabel("wavelength (Angstrom)")
    plt.ylabel("normalised flux")
    plt.title(title or f"{n} calibrated orders (x = ThAr lines used)")
    plt.tight_layout()
