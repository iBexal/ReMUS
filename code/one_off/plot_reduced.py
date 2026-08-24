import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ----------------------------------------------------------------------
# Which night, which target
# ----------------------------------------------------------------------
NIGHT = os.path.join(config.SPECTRA_ROOT, "2025-03-12")
TARGET = "Arcturus"

reduced_dir = os.path.join(config.REDUCED_ROOT, os.path.basename(NIGHT), TARGET)
files = sorted(glob(os.path.join(reduced_dir, "*.npz")))
for i, file in enumerate(files):
    if i == 0:
        data = np.load(file)

for i in range(len(data['order_number'])):
    plt.plot(data["wavelength"][i], data["flux"][i])
    # data = np.load(file)
    # plt.plot(data["wavelength"], data["flux"], label=os.path.basename(file))
plt.xlabel("Wavelength (A)")
plt.ylabel("Flux (arbitrary units)")
plt.title(f"Reduced spectra for {TARGET} on {os.path.basename(NIGHT)}")
# plt.legend()
plt.show()