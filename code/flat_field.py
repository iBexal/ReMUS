"""Flat fielding from the white light frames.

The white lamp is a quartz halogen bulb, so its own spectrum is a steep
continuum, bright in the red and weak in the blue. Dividing a science
spectrum by a raw flat would print that continuum straight into the data
along with the blaze function of the grating. Neither belongs there.

So the flat is split in two before anything is divided by it. Its smooth
part, everything varying on the scale of the blaze and the lamp's own
colour, is measured with a running median and set aside; what is left is
the part that varies pixel to pixel:

    flat = blaze(pixel) * response(pixel)

`blaze` carries the lamp SED, the blaze function and the fibre
throughput, all together and all smooth. `response` carries pixel to
pixel quantum efficiency, dust, and fringing, and averages to one. Only
`response` is divided out, so the lamp's spectral signature never enters
the science spectrum. The separation is by spatial frequency and needs no
model of the lamp: anything smoother than smooth_window is called blaze.

This is worth doing for the wavelength solution and not only for the
fluxes. An arc line is about 7.8 pixels across, so a gradient in pixel
response across the line profile pulls its fitted centroid off centre by
a fraction of the gradient. The bias is small, but it is systematic, and
it does not average down over many lines the way photon noise does.

Two things it does not fix. The blaze is left in the extracted flux
rather than divided out, since removing it needs a flux standard to be
worth anything; `blaze` is returned so it can be divided later. And
because the response is measured on the extracted 1D flat rather than on
the 2D frame, it corrects the response along each trace and not across
it.

One thing to watch. A flat field divides, and the detector's bias
pedestal is an additive constant, so (S + bias) / (F + bias) is not
S / F. Set config.APPLY_BIAS along with config.FLAT_FIELD; without it
the pedestal dilutes the measured response towards 1 and the correction
comes out too small.

Main entry points: split_flat, attach_response, apply_pixel_response,
flat_field_orders.
"""

import numpy as np
from scipy.ndimage import median_filter

import config


def split_flat(flat_spectrum, smooth_window=101, min_relative=0.15,
               max_correction=1.5):
    """Separate one order's flat into its smooth part and its response.

    Parameters
    ----------
    flat_spectrum : ndarray
        Extracted white light spectrum of one order, shape (n_pixels,).
    smooth_window : int, optional
        Width of the running median that defines "smooth", in pixels.
        Default 101. Anything varying more slowly than this is called
        blaze and kept out of the correction; anything faster is called
        response and divided out. Forced odd.
    min_relative : float, optional
        Pixels where the blaze has fallen below this fraction of its own
        maximum are left uncorrected. Default 0.15. The ends of an order
        carry almost no light, so the response measured there is noise.
    max_correction : float, optional
        Largest correction applied, as a factor either way. Default 1.5.
        A response outside 1/max_correction to max_correction is a dead
        column or a cosmic ray in the flat rather than a sensitivity, and
        dividing by it would inject a spike.

    Returns
    -------
    blaze : ndarray
        The smooth part, in counts, shape (n_pixels,). NaN where the flat
        was not finite.
    response : ndarray
        The pixel to pixel part, dimensionless and near 1, shape
        (n_pixels,). Exactly 1 wherever no correction is applied, so it
        is always safe to divide by.
    """
    s = np.asarray(flat_spectrum, float)
    n = len(s)
    window = int(smooth_window) | 1

    finite = np.isfinite(s)
    response = np.ones(n)
    blaze = np.full(n, np.nan)
    if finite.sum() < window:
        return blaze, response

    # Interpolate across the gaps before the median filter, so a NaN does
    # not eat a window's worth of the blaze around it.
    filled = np.interp(np.arange(n), np.flatnonzero(finite), s[finite])
    smooth = median_filter(filled, size=window, mode="nearest")
    blaze = np.where(finite, smooth, np.nan)

    peak = np.nanmax(smooth)
    usable = finite & (smooth > min_relative * peak) & (smooth > 0)
    if not usable.any():
        return blaze, response

    raw = np.ones(n)
    raw[usable] = filled[usable] / smooth[usable]

    # A response is a sensitivity, so it should sit near 1. Anything far
    # from it is a defect, and is left uncorrected rather than divided by.
    lo, hi = 1.0 / max_correction, max_correction
    good = usable & (raw > lo) & (raw < hi)
    response[good] = raw[good]

    # Renormalise over the part actually being corrected, so the
    # correction changes the shape of the spectrum and not its level.
    if good.any():
        response[good] /= np.median(response[good])

    return blaze, response


def attach_response(orders, flat_image, n_sigma, response_attr,
                    flat_attr=None, smooth_window=None, min_relative=None,
                    max_correction=None, label="", verbose=True):
    """Extract the flat through one aperture and turn it into a response.

    The aperture matters. Arcs are extracted through a narrower aperture
    than science frames, so the two see slightly different mixtures of
    the same detector columns, and a response measured through one does
    not exactly describe the other. Each gets its own, from the same flat.

    Parameters
    ----------
    orders : list of Order
        Traces to extract along. Modified in place: response_attr, and
        flat_attr where given, are set on each.
    flat_image : ndarray
        Coadded white light frame, 2D of shape (ny, nx). This is the
        second value returned by order_tracing.trace_orders.
    n_sigma : float
        Half width of the extraction aperture, in units of the profile
        sigma. Use the same value the spectra being corrected are
        extracted with.
    response_attr : str
        Attribute the response is stored in, "pixel_response" or
        "pixel_response_arc".
    flat_attr : str, optional
        Attribute the extracted flat itself is stored in. Default None,
        meaning neither it nor its blaze is kept. The blaze follows the
        flat rather than being written on every call, so that blaze and
        flat_spectrum always describe the same aperture; the arc pass
        would otherwise leave a blaze belonging to a narrower aperture
        beside a science flux extracted through a wider one.
    smooth_window : int, optional
        Passed to split_flat. Default None, meaning
        config.FLAT_SMOOTH_WINDOW.
    min_relative : float, optional
        Passed to split_flat. Default None, meaning
        config.FLAT_MIN_RELATIVE.
    max_correction : float, optional
        Passed to split_flat. Default None, meaning
        config.FLAT_MAX_CORRECTION.
    label : str, optional
        Word for this aperture in the printed summary. Default "".
    verbose : bool, optional
        Print how much of the detector is being corrected and by how
        much. Default True.

    Returns
    -------
    n : int
        Number of orders given a response.
    """
    smooth_window = smooth_window or config.FLAT_SMOOTH_WINDOW
    min_relative = (config.FLAT_MIN_RELATIVE if min_relative is None
                    else min_relative)
    max_correction = max_correction or config.FLAT_MAX_CORRECTION

    n, corrected, total, sizes = 0, 0, 0, []
    for order in orders:
        flat = order.extract_weighted(flat_image, n_sigma=n_sigma)
        blaze, response = split_flat(flat, smooth_window=smooth_window,
                                     min_relative=min_relative,
                                     max_correction=max_correction)
        if flat_attr:
            setattr(order, flat_attr, flat)
            order.blaze = blaze
        setattr(order, response_attr, response)
        n += 1
        touched = response != 1.0
        corrected += int(touched.sum())
        total += len(response)
        if touched.any():
            sizes.append(float(np.median(np.abs(response[touched] - 1.0))))

    if verbose and n:
        pct = 100.0 * corrected / max(total, 1)
        typical = 100.0 * float(np.median(sizes)) if sizes else 0.0
        print(f"flat field{' (' + label + ')' if label else ''}: {n} orders, "
              f"correcting {pct:.0f}% of their pixels by {typical:.1f}% in the median")
        if pct < 50:
            print("  note: less than half of each order is being corrected. That "
                  "means the blaze falls below FLAT_MIN_RELATIVE over much of the "
                  "detector, so either the flats are underexposed or the orders "
                  "run off the end of the frame.")
        if typical > 10:
            print("  note: corrections of more than 10% are larger than pixel "
                  "response usually is. If APPLY_BIAS is off, check it: a pedestal "
                  "left in the flat dilutes the response instead, so a large "
                  "correction here more often means real structure in the lamp.")
    return n


def apply_pixel_response(orders, attr, response_attr="pixel_response",
                         verbose=True):
    """Divide one named spectrum of every order by its pixel response.

    Parameters
    ----------
    orders : list of Order
        Traces carrying the named response. Modified in place: the named
        attribute is replaced by the corrected spectrum.
    attr : str
        Name of the attribute to correct, "thar_spectrum" or
        "science_spectrum".
    response_attr : str, optional
        Attribute holding the response to divide by. Default
        "pixel_response".
    verbose : bool, optional
        Print how many orders were corrected. Default True.

    Returns
    -------
    n : int
        Number of orders corrected.
    """
    n = 0
    for order in orders:
        response = getattr(order, response_attr, None)
        spectrum = getattr(order, attr, None)
        if response is None or spectrum is None:
            continue
        spectrum = np.asarray(spectrum, float)
        if len(response) != len(spectrum):
            continue
        setattr(order, attr, spectrum / response)
        n += 1
    if verbose and n:
        print(f"    flat fielded {n} orders ({attr.replace('_', ' ')})")
    return n


def flat_field_orders(orders, flat_image, verbose=True, **kwargs):
    """Give every order the responses it needs, in one call.

    Two are measured: one through the science aperture and one through
    the narrower arc aperture. The extracted flat from the science
    aperture is kept as flat_spectrum, and the blaze alongside it.

    Parameters
    ----------
    orders : list of Order
        Traces. Modified in place.
    flat_image : ndarray
        Coadded white light frame, 2D of shape (ny, nx).
    verbose : bool, optional
        Print the summary lines. Default True.
    **kwargs
        Passed to attach_response.

    Returns
    -------
    n : int
        Number of orders given a response.
    """
    n = attach_response(orders, flat_image, config.SCIENCE_EXTRACT_NSIGMA,
                        "pixel_response", flat_attr="flat_spectrum",
                        label="science aperture", verbose=verbose, **kwargs)
    if config.FLAT_FIELD_ARCS:
        attach_response(orders, flat_image, config.ARC_EXTRACT_NSIGMA,
                        "pixel_response_arc", label="arc aperture",
                        verbose=verbose, **kwargs)
    return n
