"""Identify the traces holding Halpha, Hbeta and the Na D doublet.

Bootstrap step for a spectrograph configuration the pipeline has not seen
before. It produces the config.py settings ANCHORS (which trace holds
Halpha and which holds Hbeta), NAD_TRACE (which trace holds the Na D
doublet) and DIRECTION (whether order number rises or falls with trace
index). These cannot be derived from an arc frame, because no arc line
identity is known in advance, so they are found once from a stellar
frame by eye.

Main entry points: find_anchor_lines, solve_order_numbers,
plot_order_map, LineBrowser.
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
    """Solve for absolute echelle order numbers from three lines.

    The model is m*lambda = A + B*y_hat, where y_hat is the pixel
    position rescaled to the range -1 to 1. Differences in order number
    between the three lines are read off the trace indices, and the Na D
    doublet fixes the dispersion, leaving the order number of Halpha as
    the only unknown. A correct solve lands close to an integer; a trace
    missed between two of the lines shifts the answer by about 3 and is
    caught by m_alpha_from_nad rather than by integer_miss.

    Parameters
    ----------
    alpha : tuple
        (trace_index, pixel) of Halpha, as (int, float), pixel in
        pixels.
    beta : tuple
        (trace_index, pixel) of Hbeta, as (int, float).
    nad : tuple
        (trace_index, pixel_of_D2, pixel_of_D1) of the Na D doublet, as
        (int, float, float).
    n_pixels : int
        Number of pixels along an order.
    K : float, optional
        Grating constant, in Angstrom. Used only for an independent
        printed cross-check, never as an input to the solution. Default
        None, meaning no cross-check is made.
    verbose : bool, optional
        Print the solve, its diagnostics and the config.py settings.
        Default True.

    Returns
    -------
    result : dict
        m0 : int
            Order number at trace index 0, so that the order number of
            trace i is m0 + direction * i.
        direction : int
            +1 if order number rises with trace index, -1 if it falls.
        m_alpha : float
            Unrounded order number of the Halpha trace.
        m_alpha_int : int
            m_alpha rounded to the nearest integer.
        m_beta : int
            Order number of the Hbeta trace.
        m_nad : int
            Order number of the Na D trace.
        A : float
            Constant term of m*lambda, in Angstrom.
        B : float
            Coefficient of y_hat in m*lambda, in Angstrom.
        dispersion_at_nad : float
            Dispersion at the doublet, in Angstrom per pixel.
        integer_miss : float
            Distance from m_alpha to the nearest integer.
        m_alpha_from_nad : float
            The same solve with Na D in place of Hbeta, or NaN if Na D
            shares the Halpha trace. It uses a different trace count, so
            disagreement with m_alpha means one of the two counts is
            wrong.

    Raises
    ------
    ValueError
        If Halpha and Hbeta fall on the same trace, or if the two Na D
        positions are less than 5 pixels apart.
    """
    i_alpha, p_alpha = alpha
    i_beta, p_beta = beta
    i_nad, p_d2, p_d1 = nad

    if i_alpha == i_beta:
        raise ValueError("Halpha and Hbeta cannot be in the same order")

    # Hbeta is bluer and so has the higher order number, which fixes
    # how order number runs against trace index.
    direction = 1 if i_beta > i_alpha else -1
    dm_beta = direction * (i_beta - i_alpha)      # positive by construction
    dm_nad = direction * (i_nad - i_alpha)

    if abs(p_d2 - p_d1) < 5:
        raise ValueError(f"the two Na D positions are only {abs(p_d2 - p_d1):.1f} px "
                         f"apart, so it is the same line clicked twice")

    dispersion = (NA_D1 - NA_D2) / (p_d1 - p_d2)          # Angstrom per pixel
    G = dispersion * (n_pixels - 1) / 2.0                  # B = m_nad * G

    def y_hat(p):
        return 2.0 * (p - (n_pixels - 1) / 2.0) / (n_pixels - 1)

    def m_from_pair(lam2, dm2, p2):
        """Order number of Halpha from one further line."""
        dy = y_hat(p_alpha) - y_hat(p2)
        return ((dm2 * lam2 + dm_nad * G * dy)
                / ((H_ALPHA - lam2) - G * dy))

    m_alpha = m_from_pair(H_BETA, dm_beta, p_beta)
    # The same solve against Na D instead of Hbeta. It uses a different
    # trace count, so the two agree only if both counts are right, and
    # disagreement identifies the span holding the missing order.
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
                   "NOT close to an integer, something is wrong" if miss > 0.35 else
                   "close-ish; treat with caution")
        print(f"    misses the nearest integer by {miss:.3f}, {verdict}")
        if np.isfinite(m_alpha_nad):
            gap = abs(m_alpha_nad - m_alpha)
            print(f"    same solve using Na D instead of H-beta: {m_alpha_nad:.3f}")
            if gap < 0.5:
                print(f"      agrees to {gap:.3f}, so the trace counts on BOTH sides "
                      f"of H-alpha are right, no order missed between the anchors")
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
                print("    agrees. This check owes nothing to the data, it comes "
                      "from the grating alone, so the integer is confirmed by the "
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
    """Predict which trace index holds a given wavelength.

    Coarse model that assumes each line sits near the centre of its
    order, so the result is reliable only to about one order. It serves
    as a starting point for LineBrowser.

    Parameters
    ----------
    K : float
        Grating constant, in Angstrom.
    m0 : int
        Order number at trace index 0.
    direction : int
        +1 if order number rises with trace index, -1 if it falls.
    wavelength : float
        Wavelength of the line, in Angstrom.

    Returns
    -------
    index : int
        Predicted trace index. It is not clipped to the trace list.
    """
    return int(round(((K / wavelength) - m0) / direction))


# ======================================================================
# looking at the data
# ======================================================================

def normalised_stack(orders, spectrum_attr="science_spectrum", continuum=801,
                     smooth=51):
    """Stack the continuum-divided spectrum of every order.

    Dividing by a running median removes the blaze, so the faint orders
    at each end remain visible. The boxcar that follows is wider than a
    metal line but narrower than a Balmer line, so it suppresses the
    narrow lines while leaving the broad ones standing.

    Parameters
    ----------
    orders : list of Order
        Traced orders carrying the named spectrum attribute.
    spectrum_attr : str, optional
        Name of the attribute holding the extracted spectrum. Default
        "science_spectrum".
    continuum : int, optional
        Width of the running median taken as the continuum, in pixels.
        Default 801.
    smooth : int, optional
        Width of the boxcar applied after normalisation, in pixels.
        Default 51. A value of 1 or less leaves the rows unsmoothed.

    Returns
    -------
    stack : ndarray
        Array of shape (len(orders), n_pixels), rows in trace index
        order and columns in pixels. Rows for orders with no spectrum,
        and columns beyond the length of a short row, are NaN.
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
    """Rank the traces by their deepest broad absorption feature.

    Balmer lines are much wider than the metal lines around them, so the
    deepest feature surviving the smoothing in normalised_stack is
    usually Halpha. Hbeta is weaker and sits where the metal lines are
    densest and the signal poorest, so it is often absent from the list.
    The ranking is a starting point for LineBrowser, not an
    identification.

    Parameters
    ----------
    orders : list of Order
        Traced orders carrying the named spectrum attribute.
    spectrum_attr : str, optional
        Name of the attribute holding the extracted spectrum. Default
        "science_spectrum".
    n : int, optional
        Maximum number of traces returned. Default 8.
    edge : int, optional
        Number of pixels ignored at each end of an order. Default 200.

    Returns
    -------
    features : list of tuple
        At most n entries of (trace_index, depth, pixel), deepest first.
        trace_index is an int, depth is 1 minus the normalised flux at
        the minimum, and pixel is the pixel of that minimum. Traces with
        no finite data are omitted.
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
    """Plot every order as one image of trace index against pixel.

    The narrow lines are smoothed away by normalised_stack, so the wide
    dark features are the Balmer lines and their trace indices can be
    read off the y axis.

    Parameters
    ----------
    orders : list of Order
        Traced orders carrying the named spectrum attribute.
    spectrum_attr : str, optional
        Name of the attribute holding the extracted spectrum. Default
        "science_spectrum".
    title : str, optional
        Figure title. Default None, meaning a built-in title is used.
    suggestions : list of tuple, optional
        (trace_index, depth, pixel) entries from suggest_broad_lines;
        the first four are circled and numbered. Default None, meaning
        nothing is marked.

    Returns
    -------
    None
        The figure is created but not shown.
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
    plt.title(title or "Every order at once. The widest dark features are the "
                       "Balmer lines (circles: deepest broad features found)")
    plt.tight_layout()


class LineBrowser:
    """Interactive browser for locating one line among the orders.

    The figure shows one order's continuum-divided spectrum at a time.
    Left and right arrows step one trace, up and down jump ten, and a
    click accepts the position under the cursor after snapping it to the
    nearest line centre. Closing the window without clicking skips the
    line. Call run to display the figure and collect the result.

    Parameters
    ----------
    orders : list of Order
        Traced orders carrying the named spectrum attribute.
    name : str
        Label for the line, used in the title and printed messages.
    wavelength : float
        Wavelength of the line, in Angstrom, shown in the title.
    start_index : int, optional
        Trace index to open on, clipped to the range of orders. Default
        0.
    spectrum_attr : str, optional
        Name of the attribute holding the extracted spectrum. Default
        "science_spectrum".
    kind : str, optional
        "absorption" to snap to a minimum, anything else to snap to a
        maximum. Default "absorption".
    window : int, optional
        Half width of the refinement window, in pixels. Default 15.
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
            f"{self.name} ({self.wavelength} A), trace {self.index} of "
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
        """Display the browser and block until its window closes.

        Returns
        -------
        result : tuple or None
            (trace_index, pixel) of the accepted line, with pixel
            refined to sub-pixel precision, or None if the window was
            closed without a click.
        """
        plt.show()
        if self.result is None:
            print(f"  {self.name}: skipped")
        return self.result


def refine(spectrum, guess, window=15, kind="absorption"):
    """Snap an approximate pixel position to the nearest line centre.

    A Gaussian is fitted to the continuum-divided spectrum, so a sloping
    blaze does not drag the fit.

    Parameters
    ----------
    spectrum : ndarray
        Extracted spectrum of one order, shape (n_pixels,).
    guess : float
        Approximate line position, in pixels.
    window : int, optional
        Half width of the fitting window, in pixels. Default 15.
    kind : str, optional
        "absorption" to fit a minimum, anything else to fit a maximum.
        Default "absorption".

    Returns
    -------
    pixel : float
        Refined line position, in pixels. The rounded guess is returned
        if fewer than 5 finite points fall in the window, and the local
        extremum is returned if the fit fails or lands outside it.
    """
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
    """Identify Halpha, Hbeta and the Na D doublet, then solve.

    A LineBrowser is opened for each of the four lines in turn. If K and
    m0_guess are given, each browser opens on the trace predicted by
    predict_trace; otherwise it opens on the middle trace.

    Parameters
    ----------
    orders : list of Order
        Traced orders carrying science_spectrum.
    n_pixels : int
        Number of pixels along an order.
    K : float, optional
        Grating constant, in Angstrom, passed on to
        solve_order_numbers. Default None, meaning no starting trace is
        predicted and no cross-check is made.
    m0_guess : int, optional
        Guessed order number at trace index 0, used with K to predict
        the starting trace. Default None, meaning the browsers open on
        the middle trace.
    direction_guess : int, optional
        Guessed sign of order number against trace index, used with K
        and m0_guess. Default -1.
    show_map : bool, optional
        Draw the order map before the browsers open. Default True.

    Returns
    -------
    result : dict or None
        The dict returned by solve_order_numbers, or None if either
        Balmer line or either Na D line was skipped.
    """
    suggestions = suggest_broad_lines(orders, spectrum_attr="science_spectrum")
    print(f"Red orders should be a higher number than blue orders. H-alpha is usually around 50")
    print(f"Na D is usually around 40. H-beta is usually around 20. The exact numbers depend on the instrument and setup.")
    if suggestions:
        print("Deepest broad features, by trace. H-alpha is usually at or near "
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
