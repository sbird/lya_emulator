"""
Multi-Fidelity Emulation on Lyman alpha flux power spectrum

* Jan 6, 2022: condition emulator at one redshift due to it's too
computational expensive to train multiple MFEmulators.
"""
from typing import List, Optional

import numpy as np

from .dataloader import FluxVectors
from ..coarse_grid import Emulator


class MFEmulator(Emulator):

    """
    A wrapper around the emulator that constructs a MFEmulator at a given redshift.

    Training data are loaded from .dataloader.FluxVectors

    Keep the original attributes, and allow them to be HF training set.
    Keep the MF data as a class instance in another attribute.

    - basedir: directory to load or create emulator
    - param_names: dictionary containing names of the parameters as well as a unique 
                   integer list of positions
    - param_limits: Nx2 array containing upper and lower limits of each parameter,
                    in the order given by the integer stored in param_names
    - kf: k bins to use when getting spectra
    - mf: mean flux object, which takes mean flux parameters and outputs the mean
                    flux in each redshift bin
    - limitfac: factor to uniformly grow the parameter limits by.
    """

    def __init__(
        self,
        basedir: str,
        mfdata: FluxVectors,
        param_names: Optional[List[str]] = {
            "ns": 0,
            "Ap": 1,
            "herei": 2,
            "heref": 3,
            "alphaq": 4,
            "hub": 5,
            "omegamh2": 6,
            "hireionz": 7,
            "bhfeedback": 8,
        },
        param_limits: Optional[np.ndarray] = np.array(
            [
                [
                    0.8,
                    0.995,
                ],  # ns: 0.8 - 0.995. Notice that this is not ns at the CMB scale!
                [
                    1.2e-09,
                    2.6e-09,
                ],  # Ap: amplitude of power spectrum at 8/2pi Mpc scales (see 1812.04654)!
                [3.5, 4.1],  # herei: redshift at which helium reionization starts.
                # 4.0 is default, we use a linear history with 3.5-4.5
                [
                    2.6,
                    3.2,
                ],  # heref: redshift at which helium reionization finishes. 2.8 is default.
                # Thermal history suggests late, HeII Lyman alpha suggests earlier.
                [
                    1.6,
                    2.5,
                ],  # alphaq: quasar spectral index. 1 - 2.5 Controls IGM temperature.
                [0.65, 0.75],  # hub: hubble constant (also changes omega_M)
                [
                    0.14,
                    0.146,
                ],  # omegam h^2: We fix omega_m h^2 = 0.143+-0.001 (Planck 2018 best-fit) and vary omega_m and h^2 to match it.
                # h^2 itself has little effect on the forest.
                [6.5, 8],  # Mid-point of HI reionization
                [0.03, 0.07],  # BH feedback parameter
                #   [3.2, 4.2] # Wind speed
            ]
        ),
        kf: Optional[np.ndarray] = None,
        mf: Optional[str] = None,
        limitfac: int = 1,
        tau_thresh: Optional[float] = None,
        npart: int = 384,
        box: int = 15,
        fullphysics: bool = True,
    ):
        # this is for single-fidelity emulator only; assume here to be highres
        super().__init__(
            basedir,
            param_names=param_names,
            param_limits=param_limits,
            kf=kf,
            mf=mf,
            limitfac=limitfac,
            tau_thresh=tau_thresh,
            npart=npart,
            box=box,
            fullphysics=fullphysics,
        )

        # multi-fidelity data
        # note this is only for one redshift
        self.mfdata = mfdata

    