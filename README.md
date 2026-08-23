# ReMUS

Reduction for the Macquarie University Spectrograph. Currently, ReMUS does
order tracing, wavelength calibration, and an attempt at cosmic ray removal.

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
  frames.py               finding FITS frames, reading them, reading their times
  order_tracing.py        finding orders on a flat and extracting along them
  wavelength_solution.py  the wavelength model, fitting, quality checks, IO
  reduce_spectra.py       registering a master onto new data and writing spectra
  ReMUS.py                RUN: reduce one night
  one_off/
    anchors.py            identifying Halpha, Hbeta and Na D, solving for m
    build_master_thar.py  the master build
    find_anchors.py       RUN: identify the anchor orders (once per spectrograph)
    make_master_thar.py   RUN: build a master (once per instrument configuration)
  thar_linelist/
    ThAr_lines.dat        the ThAr line list

calibration/              the master solution, its summary, and the archive
reduced/                  calibrated spectra, one .npz per science frame
spectra/                  raw data, one folder per night
```

`ReMUS.py`, `make_master_thar.py` and `find_anchors.py` contain settings and
one function call each. All logic lives in the modules.

## Quick start

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

### 3. Reduce a night

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
| `QUALITY`, `APPLY_QUALITY` | the pass/fail thresholds |

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
| `pixel_shift` | scalar | shift applied, in pixels |

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
rebuilding.

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
