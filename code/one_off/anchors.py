"""
anchors.py

Finding which traced order holds Halpha, Hbeta and the Na D doublet, and
turning that into absolute echelle order numbers.

This is the bootstrap step for a spectrograph the pipeline has never seen.
Everything downstream -- the whole master solution -- rests on three
settings in config.py:

    ANCHORS     which trace holds Halpha, which holds Hbeta
    NAD_TRACE   which trace holds the Na D doublet
    DIRECTION   whether order number rises or falls with trace index

and those cannot be derived from the arc, because the arc has no lines whose
identity you know in advance. They have to come from a star, once, by eye.

The pleasant surprise is that once you have found the three lines, the
absolute order number follows from geometry alone -- no grating constant
needed. See solve_order_numbers.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, uniform_filter1d

H_ALPHA = 6562.8
H_BETA = 4861.3
NA_D2 = 5889.95
NA_D1 = 5895.92


# ======================================================================
# the maths
# ======================================================================

def solve_order_numbers(alpha, beta, nad, n_pixels, K=None, verbose=True):
    """Absolute order numbers from three identified lines.

    alpha, beta : (trace_index, pixel) for Halpha and Hbeta
    nad         : (trace_index, pixel_of_D2, pixel_of_D1)
    K           : the grating constant, if you have it. Used only as an
                  independent cross-check, never as an input to the answer.

    How it works. Write the model as m*lambda = A + B*y_hat, and let the
    trace indices give the DIFFERENCES in order number between the three
    lines -- those are just integers you can count off the trace list. That
    leaves three unknowns (A, B and the order number of Halpha) and three
    facts: Halpha's position, Hbeta's position, and the Na D doublet's
    separation, which fixes B directly because m*lambda is shared across
    orders. Solving gives

        m_alpha = [dm_beta*lambda_beta + dm_D*G*dy] / [(lambda_alpha - lambda_beta) - G*dy]

    with G the dispersion scale from the doublet and dy the difference in
    normalised pixel position between Halpha and Hbeta.

    Two things make this trustworthy. It should land on an integer, and how
    close it lands is a real check rather than a formality -- on this
    instrument it comes out at 86.97. And it is insensitive to almost
    everything: a 1% error in the doublet dispersion moves it by 0.013, and
    a whole pixel of error on Halpha by 0.001. Even the science frame's
    velocity offset cancels, because a Doppler shift moves all three lines
    by nearly the same number of pixels and only differences are used.

    What it IS sensitive to is the trace count between the lines. An order
    the tracer missed between Halpha and Hbeta moves the answer by about 3
    -- and lands it near a different integer, so the integer test alone will
    not catch it. Two things do: the same solve run against Na D instead of
    Hbeta uses a different trace count and so disagrees when either count is
    wrong, and K, which owes nothing to the data at all.

    Returns a dict with m0, direction, and the diagnostics.
    """
    i_alpha, p_alpha = alpha
    i_beta, p_beta = beta
    i_nad, p_d2, p_d1 = nad

    if i_alpha == i_beta:
        raise ValueError("Halpha and Hbeta cannot be in the same order")

    # Hbeta is bluer, so it has the higher order number. That fixes which
    # way order number runs against trace index -- no assumption needed.
    direction = 1 if i_beta > i_alpha else -1
    dm_beta = direction * (i_beta - i_alpha)      # positive by construction
    dm_nad = direction * (i_nad - i_alpha)

    if abs(p_d2 - p_d1) < 5:
        raise ValueError(f"the two Na D positions are only {abs(p_d2 - p_d1):.1f} px "
                         f"apart -- that is the same line clicked twice")

    dispersion = (NA_D1 - NA_D2) / (p_d1 - p_d2)          # Angstrom per pixel
    G = dispersion * (n_pixels - 1) / 2.0                  # B = m_nad * G

    def y_hat(p):
        return 2.0 * (p - (n_pixels - 1) / 2.0) / (n_pixels - 1)

    def m_from_pair(lam2, dm2, p2):
        """Order number of Halpha, using one other line as the second fact."""
        dy = y_hat(p_alpha) - y_hat(p2)
        return ((dm2 * lam2 + dm_nad * G * dy)
                / ((H_ALPHA - lam2) - G * dy))

    m_alpha = m_from_pair(H_BETA, dm_beta, p_beta)
    # The same solve run against Na D instead of Hbeta. It uses a different
    # trace count (Halpha to Na D rather than Halpha to Hbeta), so the two
    # answers agree only if BOTH counts are right -- and disagreement says
    # which span the missing order is in. Landing on an integer cannot do
    # this on its own: an order missed between the Balmer lines still gives
    # a near-integer answer, just the wrong integer.
    m_alpha_nad = m_from_pair(NA_D2, dm_nad, p_d2) if dm_nad != 0 else np.nan

    m_alpha_int = int(round(m_alpha))
    m0 = m_alpha_int - direction * i_alpha
    B = (m_alpha_int + dm_nad) * G
    A = m_alpha_int * H_ALPHA - B * y_hat(p_alpha)

    result = {
        "m0": m0,
        "direction": direction,
        "m_alpha": m_alpha,
        "m_alpha_int": m_alpha_int,
        "m_beta": m_alpha_int + dm_beta,
        "m_nad": m_alpha_int + dm_nad,
        "A": A,
        "B": B,
        "dispersion_at_nad": dispersion,
        "integer_miss": abs(m_alpha - m_alpha_int),
        "m_alpha_from_nad": m_alpha_nad,
    }

    if verbose:
        print("Solving for absolute order numbers from the three lines:")
        print(f"  H-alpha  trace {i_alpha:3d}, pixel {p_alpha:7.1f}")
        print(f"  H-beta   trace {i_beta:3d}, pixel {p_beta:7.1f}   "
              f"-> {abs(i_beta - i_alpha)} traces from H-alpha")
        print(f"  Na D     trace {i_nad:3d}, pixels {p_d2:.1f} and {p_d1:.1f}  "
              f"-> {dispersion:.5f} A/px there")
        print(f"  order number rises with trace index: "
              f"{'yes' if direction > 0 else 'no'}  (DIRECTION = {direction:+d})")
        print()
        print(f"  m(H-alpha) = {m_alpha:.3f}  ->  {m_alpha_int}")
        miss = result["integer_miss"]
        verdict = ("lands on an integer" if miss < 0.15 else
                   "NOT close to an integer -- something is wrong" if miss > 0.35 else
                   "close-ish; treat with caution")
        print(f"    misses the nearest integer by {miss:.3f} -- {verdict}")
        if np.isfinite(m_alpha_nad):
            gap = abs(m_alpha_nad - m_alpha)
            print(f"    same solve using Na D instead of H-beta: {m_alpha_nad:.3f}")
            if gap < 0.5:
                print(f"      agrees to {gap:.3f}, so the trace counts on BOTH sides "
                      f"of H-alpha are right -- no order missed between the anchors")
            else:
                print(f"      DISAGREES by {gap:.2f}. The two use different trace "
                      f"counts, so one of the two spans (H-alpha to H-beta, or "
                      f"H-alpha to Na D) is missing an order. Fix the trace list "
                      f"before going on.")
        print(f"  m(H-beta) = {result['m_beta']}, m(Na D) = {result['m_nad']}")
        print(f"  -> m0 = {m0}, DIRECTION = {direction:+d}")
        print(f"  -> m*lambda ~ {A:.0f} + {B:.0f} * y_hat, "
              f"{2 * B / result['m_nad']:.1f} A of coverage per order at Na D")

        if K is not None:
            m_from_K = K / H_ALPHA
            print()
            print(f"  Cross-check against the grating constant K = {K:.0f} A:")
            print(f"    K/lambda(H-alpha) = {m_from_K:.2f} -> {int(round(m_from_K))}")
            if int(round(m_from_K)) == m_alpha_int:
                print("    agrees. This one owes nothing to the data -- it comes from "
                      "the grating alone -- so the integer is now confirmed by the "
                      "optics as well as by the trace counts.")
            else:
                print(f"    DISAGREES by {int(round(m_from_K)) - m_alpha_int:+d}. "
                      f"Either an order was missed between the two anchor traces "
                      f"(which shifts this answer by about 3 per order), a line was "
                      f"misidentified, or K is not right for this configuration. "
                      f"Do not go on until this resolves.")
        else:
            print("\n  No K supplied, so nothing independent confirms the integer. "
                  "Worth doing once with the grating constant to hand.")

        print()
        print("  Paste into config.py:")
        print(f"    DIRECTION = {direction}")
        print(f"    ANCHORS   = [({i_alpha}, {H_ALPHA}), ({i_beta}, {H_BETA})]")
        print(f"    NAD_TRACE = {i_nad}")
        print(f"    # first run of make_master_thar: "
              f"NAD_PIXEL_GUESSES = [{p_d2:.1f}, {p_d1:.1f}]")
    return result


def predict_trace(K, m0, direction, wavelength):
    """Which trace index should hold a given wavelength, from a coarse model.

    Only a starting point for the browser -- it assumes each line sits near
    its order's centre, which is true to about an order.
    """
    return int(round(((K / wavelength) - m0) / direction))


# ======================================================================
# looking at the data
# ======================================================================

def normalised_stack(orders, spectrum_attr="science_spectrum", continuum=801,
                     smooth=51):
    """Every order's spectrum as one 2D array, with the narrow lines removed.

    Rows are trace index, columns are pixel. Two things happen here, and
    both are needed. Dividing by a running median takes out the blaze, so
    the faint orders at each end are visible at all. Then a boxcar wider
    than a metal line but narrower than a Balmer line wipes out the forest
    of hairlines that otherwise fills the whole map -- in a star as
    line-rich as Arcturus the raw map is unreadable hash, and Halpha is
    nowhere to be seen in it.

    Pass smooth=1 for the unsmoothed version.
    """
    rows = []
    for order in orders:
        spectrum = getattr(order, spectrum_attr, None)
        if spectrum is None:
            rows.append(np.full(1, np.nan))
            continue
        s = np.asarray(spectrum, float)
        base = median_filter(s, continuum)
        with np.errstate(invalid="ignore", divide="ignore"):
            norm = np.where(base > 0, s / base, np.nan)
        if smooth > 1:
            norm = uniform_filter1d(np.nan_to_num(norm, nan=1.0), smooth)
        rows.append(norm)
    width = max(len(r) for r in rows)
    stack = np.full((len(rows), width), np.nan)
    for i, r in enumerate(rows):
        stack[i, :len(r)] = r
    return stack


def suggest_broad_lines(orders, spectrum_attr="science_spectrum", n=8, edge=200):
    """Rank the traces by their deepest BROAD absorption feature.

    Balmer lines are much wider than the metal lines around them, so once
    the narrow ones are smoothed away whatever is left standing is usually
    Halpha. On this data it comes out top, at the right pixel.

    Take it as a head start, not an answer. Hbeta is weaker and sits where
    the metal lines are densest and the signal is poorest, so it often does
    not make the list at all -- find Halpha here, then page bluewards.

    Returns [(trace_index, depth, pixel), ...], deepest first.
    """
    stack = normalised_stack(orders, spectrum_attr)
    out = []
    for i, row in enumerate(stack):
        core = row[edge:-edge]
        if not np.any(np.isfinite(core)):
            continue
        k = int(np.nanargmin(core))
        out.append((i, float(1.0 - core[k]), int(k + edge)))
    out.sort(key=lambda item: -item[1])
    return out[:n]


def plot_order_map(orders, spectrum_attr="science_spectrum", title=None,
                   suggestions=None):
    """The whole spectrum as one image: trace index against pixel.

    The quickest way to find the Balmer lines. With the narrow lines
    smoothed out, what remains dark and wide is a Balmer line; read its
    trace index off the y axis and confirm it in the browser.
    """
    stack = normalised_stack(orders, spectrum_attr)
    finite = stack[np.isfinite(stack)]
    lo, hi = np.percentile(finite, [1, 99]) if finite.size else (0, 1)

    plt.figure(figsize=(14, 8))
    plt.imshow(stack, aspect="auto", origin="lower", cmap="magma",
               vmin=lo, vmax=hi, interpolation="nearest")
    plt.colorbar(label="continuum-divided, narrow lines smoothed away", shrink=0.8)
    if suggestions:
        for rank, (i, depth, pixel) in enumerate(suggestions[:4]):
            plt.plot(pixel, i, "o", ms=13, mfc="none", mec="#39d0ff", mew=1.6)
            plt.annotate(f"{rank + 1}", (pixel, i), color="#39d0ff", fontsize=9,
                         xytext=(11, 6), textcoords="offset points")
    plt.xlabel("pixel along the order")
    plt.ylabel("trace index (0 = first traced order)")
    plt.title(title or "Every order at once — the widest dark features are the "
                       "Balmer lines (circles: deepest broad features found)")
    plt.tight_layout()


class LineBrowser:
    """Page through orders and click the line you are after.

    Left and right arrows change order, up and down jump ten, and a click
    accepts the position under the cursor after snapping it to the nearest
    line centre. Closing the window without clicking gives up on that line.
    """

    def __init__(self, orders, name, wavelength, start_index=0,
                 spectrum_attr="science_spectrum", kind="absorption", window=15):
        self.orders = orders
        self.name = name
        self.wavelength = wavelength
        self.attr = spectrum_attr
        self.kind = kind
        self.window = window
        self.index = int(np.clip(start_index, 0, len(orders) - 1))
        self.result = None
        self.fig, self.ax = plt.subplots(figsize=(14, 5))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._draw()

    def _spectrum(self):
        s = getattr(self.orders[self.index], self.attr, None)
        return None if s is None else np.asarray(s, float)

    def _draw(self):
        self.ax.clear()
        s = self._spectrum()
        if s is None or not np.any(np.isfinite(s)):
            self.ax.text(0.5, 0.5, "no spectrum on this trace",
                         ha="center", transform=self.ax.transAxes)
        else:
            base = median_filter(s, 301)
            with np.errstate(invalid="ignore", divide="ignore"):
                self.ax.plot(np.where(base > 0, s / base, np.nan), lw=0.8, color="#2a78d6")
            self.ax.set_ylim(0, 1.4)
        self.ax.set_xlabel("pixel")
        self.ax.set_ylabel("continuum-divided flux")
        self.ax.set_title(
            f"{self.name} ({self.wavelength} A)   —   trace {self.index} of "
            f"{len(self.orders) - 1}\n"
            f"left/right = next order, up/down = jump 10, click = accept, "
            f"close window = skip")
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        step = {"left": -1, "right": 1, "down": -10, "up": 10}.get(event.key)
        if step is None:
            return
        self.index = int(np.clip(self.index + step, 0, len(self.orders) - 1))
        self._draw()

    def _on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        s = self._spectrum()
        if s is None:
            return
        pixel = refine(s, event.xdata, window=self.window, kind=self.kind)
        self.result = (self.index, pixel)
        print(f"  {self.name}: trace {self.index}, clicked {event.xdata:.0f} "
              f"-> centroid {pixel:.2f}")
        plt.close(self.fig)

    def run(self):
        plt.show()
        if self.result is None:
            print(f"  {self.name}: skipped")
        return self.result


def refine(spectrum, guess, window=15, kind="absorption"):
    """Snap a clicked position to the nearest line centre, on the
    continuum-divided spectrum so a sloping blaze does not drag the fit."""
    from scipy.optimize import curve_fit

    s = np.asarray(spectrum, float)
    base = median_filter(s, 301)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = np.where(base > 0, s / base, np.nan)
    guess = int(round(guess))
    lo = max(0, guess - window)
    hi = min(len(s), guess + window)
    x = np.arange(lo, hi)
    y = norm[lo:hi]
    good = np.isfinite(y)
    if good.sum() < 5:
        return float(guess)
    x, y = x[good], y[good]

    def gaussian(xx, amp, mu, sigma, offset):
        return amp * np.exp(-(xx - mu) ** 2 / (2 * sigma ** 2)) + offset

    extreme = y.min() if kind == "absorption" else y.max()
    try:
        popt, _ = curve_fit(gaussian, x, y,
                            p0=(extreme - np.median(y), guess,
                                max(2.0, window / 4), np.median(y)))
        if lo < popt[1] < hi:
            return float(popt[1])
    except Exception:
        pass
    return float(x[np.argmin(y) if kind == "absorption" else np.argmax(y)])


def find_anchor_lines(orders, n_pixels, K=None, m0_guess=None, direction_guess=-1,
                      show_map=True):
    """Walk through identifying all four lines, then solve.

    If K and a guessed m0 are given, each browser opens on the order that
    model predicts, which usually means confirming rather than hunting. With
    nothing to go on it opens in the middle and you page across -- the map
    drawn first is what makes that quick.

    Returns whatever solve_order_numbers returns, or None if a line was skipped.
    """
    suggestions = suggest_broad_lines(orders, spectrum_attr="science_spectrum")
    print(f"Red orders should be a higher number than blue orders. H-alpha is usually around 50")
    print(f"Na D is usually around 40. H-beta is usually around 20. The exact numbers depend on the instrument and setup.")
    if suggestions:
        print("Deepest broad features, by trace -- H-alpha is usually at or near "
              "the top of this list:")
        for rank, (i, depth, pixel) in enumerate(suggestions):
            print(f"  {rank + 1}. trace {i:3d}  pixel {pixel:5d}  depth {depth:.2f}")
        print("H-beta is often absent from it: it is weaker, and it sits where the "
              "metal lines are densest and the signal poorest.\n")

    if show_map:
        plot_order_map(orders, suggestions=suggestions)
        print("Note the trace indices of the two widest dark features, close the "
              "map, then confirm each line in the browser.")
        plt.show()

    def start_for(wavelength):
        if K is not None and m0_guess is not None:
            return predict_trace(K, m0_guess, direction_guess, wavelength)
        return len(orders) // 2

    alpha = LineBrowser(orders, "H-alpha", H_ALPHA, start_for(H_ALPHA)).run()
    beta = LineBrowser(orders, "H-beta", H_BETA, start_for(H_BETA)).run()
    if alpha is None or beta is None:
        print("Both Balmer lines are needed. Nothing solved.")
        return None

    d2 = LineBrowser(orders, "Na D2", NA_D2, start_for(NA_D2), window=15).run()
    if d2 is None:
        print("Na D2 is needed. Nothing solved.")
        return None
    d1 = LineBrowser(orders, "Na D1", NA_D1, d2[0], window=15).run()
    if d1 is None:
        print("Na D1 is needed. Nothing solved.")
        return None
    if d1[0] != d2[0]:
        print(f"  WARNING: the two Na D lines were clicked in different traces "
              f"({d2[0]} and {d1[0]}). They are 6 A apart and must share an order.")

    print()
    return solve_order_numbers(alpha, beta, (d2[0], d2[1], d1[1]), n_pixels, K=K)
