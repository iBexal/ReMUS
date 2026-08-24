# ReMUS

Reduction for the Macquarie University Spectrograph. Currently, ReMUS does
order tracing, wavelength calibration, optional bias and dark subtraction,
optional flat fielding, and an attempt at cosmic ray removal.

## Requirements

Python 3.8 or later, with:

```
numpy  scipy  pandas  matplotlib  fitsio
```

Run everything from the `code/` directory. The saved master is a pickle
holding classes defined in `wavelength_solution.py`, so that module has to be
importable when it is loaded. Loading from elsewhere gives
`ModuleNotFoundError: No module named 'wavelength_solution'`, which is about
the import path and not about the file being damaged.

## Layout

```
code/
  config.py               instrument constants, paths, quality thresholds
  frames.py               finding FITS frames, reading them, bias and dark, times
  order_tracing.py        finding orders on a flat and extracting along them
  flat_field.py           splitting the white lamp into blaze and pixel response
  wavelength_solution.py  the wavelength model, fitting, quality checks, IO
  reduce_spectra.py       registering a master onto new data and writing spectra
  ReMUS.py                RUN: reduce one night
  one_off/
    anchors.py            identifying Halpha, Hbeta and Na D, solving for m
    build_master_thar.py  the master build
    find_anchors.py       RUN: identify the anchor orders (once per spectrograph)
    make_master_thar.py   RUN: build a master (once per instrument configuration)
  thar_linelist/
    ThAr_lines.dat        the ThAr line list (VACUUM wavelengths)

calibration/              the master solution, its summary, and the archive
reduced/                  calibrated spectra, one .npz per science frame
spectra/                  raw data, one folder per night
```

`ReMUS.py`, `make_master_thar.py` and `find_anchors.py` contain settings and
one function call each. All logic lives in the modules.

## Quick start

### Reduce a night

```
python ReMUS.py
```

Set `NIGHT` and `TARGET` at the top. Nothing else is needed: the master is
found where it always is, and arcs are matched to science frames by time.

## How it works

The model is fitted in `m * lambda`, the product of echelle order number and
wavelength, because a single smooth function of detector row describes that
product for every order at once:

```
m * lambda(y) = a0 + (a1 * u + a2 * f) / sqrt(f**2 + u**2),   u = y - y_centre
```

`f` is the camera focal length in pixels. A low-degree Chebyshev correction in
(pixel, order number) is fitted on top. 

Applying a master to a later night takes four steps, each of which reports
itself:

1. orders are identified by their position across the detector, never by
   counting down the trace list, so an order the tracer misses costs that
   order and nothing else;
2. the arc nearest each science frame in time is registered against the
   master's own arc, giving a shift along the dispersion;
3. the shifted solution is matched against the ThAr atlas, which is the only
   step that can show the master no longer fits this night;
4. science spectra are extracted, cleaned of cosmic rays and written.

ThAr line detection and atlas matching are redone from scratch for every arc.

## Building wavelength solutions

### 1. Identify the anchor orders (first time on a spectrograph)

```
python one_off/find_anchors.py
```

Needs white-light flats and one science frame of a star showing H-alpha,
H-beta and Na D. A map of all orders opens, then a browser for each line.
Arrow keys change order, a click accepts. The result is printed ready to
paste into `config.py`:

```
DIRECTION = -1
ANCHORS   = [(50, 6562.8), (20, 4861.3)]
NAD_TRACE = 40
```

The solve does not need the grating constant. Setting `BLAZE_ANGLE_DEG` and
`GROOVE_DENSITY_MM` adds an independent check that no order was missed
between the two Balmer lines.

### 2. Build a master (once per instrument configuration)

```
python one_off/make_master_thar.py
```
The master is a description of the instrument rather than of one night. It is
built once per instrument configuration and then registered onto new data, so
routine reduction needs no interaction and no refitting.

Needs white-light flats, one arc and one science frame. Click the two Na D
lines when the plots open, D2 (5889.95 A) first. The run prints the refined
pixels to paste into `NAD_PIXEL_GUESSES` so the clicking is only needed once.

The solution is written to `calibration/master_wavelength_solution.pkl` with a
plain-text summary beside it and a dated copy in `calibration/archive/`. If
any quality check fails it is reported and not saved.

## Configuration

Everything in `config.py`. The settings most likely to need changing:

| Setting | Meaning |
| --- | --- |
| `PROJECT_ROOT` | project root; every other path derives from it |
| `ANCHORS`, `NAD_TRACE`, `DIRECTION` | which trace holds which line, from `find_anchors.py` |
| `BLAZE_ANGLE_DEG`, `GROOVE_DENSITY_MM` | grating parameters, used for the order-number cross-check |
| `EXPECTED_LINE_SIGMA_PIXELS` | instrumental profile sigma in pixels |
| `ATLAS_AMPLITUDE_MIN`, `ATLAS_DOMINANCE` | how strong and how isolated an atlas line must be |
| `INTERPOLATE_BETWEEN_ARCS`, `MAX_ARC_GAP_MINUTES` | arc selection by time |
| `CLEAN_COSMIC_RAYS`, `COSMIC_RAY_MAX_WIDTH`, `COSMIC_RAY_SIGMA` | spike removal |
| `APPLY_BIAS`, `APPLY_DARK`, `MASTER_BIAS`, `MASTER_DARK` | detector calibration at read-in |
| `FLAT_FIELD`, `FLAT_SMOOTH_WINDOW` | flat fielding from the white lamp |
| `QUALITY`, `APPLY_QUALITY` | the pass/fail thresholds |

## Bias, dark and flat

All three are off by default, so nothing changes until they are turned on.
They are separate switches because they buy different things.

### Bias and dark

Applied in `frames.read_image`, which is the one place a frame enters the
pipeline, so everything downstream sees a corrected frame and none of it
has to know.

```
APPLY_BIAS  = True
APPLY_DARK  = True
MASTER_BIAS = ".../Master_Bias_Darks_2026"
MASTER_DARK = ".../Master_Bias_Darks_2026"
```

Either setting takes a master FITS file, a directory, or a list. A
directory is filtered on `IMAGETYP`, so the same folder can be given to
both. Frames are grouped by exposure time and the group nearest the frame
being calibrated is used, then scaled by the ratio of exposure times; a
scale far from 1 is reported, since stretching a 1 s dark to 900 s scales
its read noise too. `DARK_INCLUDES_BIAS` says whether the master dark
still carries its own pedestal, which the 2026 masters do, and stops the
pedestal being subtracted twice.

A wavelength axis does not need any of this. Line centroids are measured
above a median-filtered continuum, which removes an additive pedestal
along with everything else smooth. The reason to turn `APPLY_BIAS` on is
flat fielding, below, because that divides.

Dark current on this detector, measured from the 2026 masters, is about
0.15 ADU/s at 0 C. A 300 s exposure collects tens of ADU against a bias
of about 1000, so `APPLY_DARK` matters for long exposures and for hot
pixels and not much else.

### Flat fielding

The white lamp is a quartz halogen bulb, so it has its own steep
continuum, and dividing by a raw flat would print that continuum into
every spectrum. It does not, because the flat is split first:

```
flat = blaze(pixel) * response(pixel)
```

`blaze` is everything varying more slowly than `FLAT_SMOOTH_WINDOW`
pixels, measured with a running median: the lamp's colour, the grating's
blaze and the fibre throughput, all together. `response` is what is left
pixel to pixel, and averages to one. Only `response` is divided out, so
the lamp's spectral signature never reaches the science spectrum. The
separation is by spatial frequency alone and needs no model of the lamp.
A line is about 8 pixels wide and the blaze runs over thousands, so any
window between about 51 and 301 gives the same answer.

The blaze is deliberately left in the flux, since removing it properly
needs a flux standard, and saved in the output `.npz` as `blaze` so it
can be divided out later.

Two responses are measured from the same flat, one through the science
extraction aperture and one through the narrower arc aperture, because a
response measured through one does not exactly describe the other.
`FLAT_FIELD_ARCS` controls whether the arcs are corrected; it is the part
that touches the wavelength solution, since a gradient in pixel response
across a line profile pulls its fitted centroid.

Set `APPLY_BIAS` whenever `FLAT_FIELD` is on. A flat field divides and
the pedestal is additive, so it does not cancel: `(S + b) / (F + b)` is
not `S / F`. The pipeline warns if you forget.

A master built with flat fielding on should be used with it on. The
registration cross-correlates tonight's arc against the master's own, and
correcting one side but not the other leaves a fixed mismatch between
them.

## Output

One `.npz` per science frame in `reduced/<night>/<target>/`. Arrays run blue to
red by order number, so row `i` of `wavelength` and row `i` of `flux` belong
together.

| Key | Shape | Contents |
| --- | --- | --- |
| `wavelength` | (n_orders, n_pixels) | Angstrom, in the arc frame |
| `flux` | (n_orders, n_pixels) | extracted counts |
| `order_number` | (n_orders,) | physical echelle order number m |
| `trace_x` | (n_orders,) | order position across the detector, pixels |
| `pixel` | (n_pixels,) | pixel index along the order |
| `cosmic_rays_removed` | (n_orders,) | pixels replaced per order |
| `source_frame` | scalar | the science frame it came from |
| `arc_frames` | (1 or 2,) | the arc or arcs that set the shift |
| `pixel_shift` | scalar | shift applied at `pixel_shift_reference_m`, in pixels |
| `pixel_shift_tilt` | scalar | change in that shift per order, pixels per order |
| `pixel_shift_reference_m` | scalar | order number `pixel_shift` belongs to |
| `blaze` | (n_orders, n_pixels) | smooth part of the white lamp; only when flat fielding |
| `flat_fielded`, `bias_subtracted`, `dark_subtracted` | scalar | what was applied |

```python
import numpy as np
d = np.load("Arcturus_..._wave.npz")
w, f, m = d["wavelength"], d["flux"], d["order_number"]
```

## Quality checks

A master that fails any check is reported and not saved. Values in brackets
are from the 2025-03-12 master, for comparison.

| Check | Threshold | Meaning |
| --- | --- | --- |
| atlas lock | SNR >= 15 (57) | the seed found the atlas, summed over all orders |
| order numbers | margin >= 2x (7.7) | the numbering beats m +/- 1 and m +/- 2 |
| line residuals | <= 15 mA (2.67) | scatter of matched lines about the fit |
| cross-validation | <= 20 mA (2.63) | same, for lines held out of the fit |
| order coverage | >= 60% (77/86) | orders carrying at least 4 matched lines |
| pixel coverage | >= 75% (84%) | fraction of each order spanned by matched lines |
| residual trends | <= 6 mA (0.57) | no systematic drift with pixel or order |
| order overlap | <= 600 m/s (76) | adjacent orders agree where they share wavelengths |

Reuse adds two more: the shifted master must land on the atlas within 15 mA,
and order overlap must still hold. The per-order scatter of the measured shift
is printed as well; tenths of a pixel is ordinary flexure, pixels of
disagreement means the dispersion itself changed and the master needs
rebuilding. One pixel is about 1 km/s here, so the scatter reads directly as
the disagreement a single rigid shift cannot express.

Where the shift varies smoothly across the orders rather than randomly, that
drift is fitted and applied per order rather than only reported. Set
`APPLY_ARC_TILT = False` to go back to one rigid shift for every order.

Order overlap is the only check with no atlas in it. Adjacent orders observe
the same lamp lines on different parts of the detector, so cross-correlating
them measures the disagreement in velocity independently of the line list.

## Known limitations

- Arcs should bracket each science exposure. With arcs on one side only the
  shift is held at its measured value across the gap, and instrument drift
  can become degenerate with the target's motion, to an extent.
- Cosmic-ray removal works on the extracted 1D spectrum and catches spikes one
  or two pixels wide. A long diagonal track crossing several rows is wider
  than that and is left alone.
- Orders redder than about m = 73 on this instrument do not overlap, because
  their free spectral range is wider than the detector. The overlap check
  skips them rather than failing them.
- A master built before a realignment does not describe the instrument
  afterwards. Rebuild rather than stretching it; old masters stay in
  `calibration/archive/` so earlier data can still be reduced with the master
  that described the instrument at the time.