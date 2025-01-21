import os
import numpy as np
from parameters import *

for g in g_arr:
    dat_file = f"sff_open/SFFvsTime,j={j},M={M},g={g},beta={β},kappa={γ},ntraj={ntraj}.dat"
    npy_file = f"sff_open/sff_j={j}_M={M}_ω={ω}_ω0={ω0}_gc={gc}_β={β}_γ={γ}_g={g}_ntraj={ntraj}.npy"  # Replace with your desired .npy file path

    # Load the .dat file into a NumPy array
    # Assumes the .dat file contains tabular data separated by whitespace
    data = np.loadtxt(dat_file)

    # Save the array to an .npy file
    np.save(npy_file, data)

    print(f"Data from {dat_file} has been saved to {npy_file}")
