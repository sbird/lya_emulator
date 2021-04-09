"""Specialization of the Simulation class to galaxy formation simulations."""

import math
import os
import os.path
import shutil
import numpy as np
from . import simulationics
from . import lyasimulation

class GalaxySim(lyasimulation.LymanAlphaSim):
    """Specialise the Simulation class to do full physics runs with galaxy formation (stars, AGN, winds) enabled.
    Mirrors the model used in ASTERIX. No massive neutrinos.
    Extra parameters:
        bhfeedback - Amount of BH feedback."""
    __doc__ = __doc__+simulationics.SimulationICs.__doc__
    def __init__(self, *, bhfeedback = 0.05, windsigma=3.7, **kwargs):
        #super generates the helium reionization table
        super().__init__(**kwargs)
        self.metalcool = "cooling_metal_UVB"
        self.bhfeedback = bhfeedback
        self.windsigma = windsigma

    def _feedback_config_options(self, config, prefix=""):
        """Config options specific to the Lyman alpha forest star formation criterion"""
        #No feedback!
        _,_=config,prefix

    def _feedback_params(self, config):
        """Galaxy formation config options from ASTERIX."""
        #Stellar feedback parameters
        config['StarformationCriterion'] = 'density' #Note no h2 star formation! Different from ASTERIX.
        config['WindModel'] = 'ojft10,decouple'
        config['WindOn'] = 1
        config['MetalCoolFile'] = self.metalcool
        #Wind speed normalisation
        config['WindSigma0'] = 353.0 #km/s
        #Wind speed: controls the strength of the supernova feedback. Default is 3.7
        config['WindSpeedFactor'] = self.windsigma
        config['MetalReturnOn'] = 1
        #SPH parameters
        config['DensityKernelType'] = 'quintic'
        config['DensityIndependentSphOn'] = 1
        config['OutputPotential'] = 0
        #Dynamic friction models for BH
        config['BlackHoleRepositionEnabled'] = 0
        config['BH_DRAG'] = 1
        config['BH_DynFrictionMethod'] = 2
        #Black hole feedback model
        config['BlackHoleFeedbackFactor'] = self.bhfeedback
        config['BlackHoleFeedbackMethod'] = "spline | mass"
        #2 generations only for numerical sanity.
        config['Generations'] = 2
        #This scales with the mass of a DM particle because
        #it stops the DM scattering the BH out of the halo.
        #Newton's constant in Mpc^3 / (internal mass units)
        #cm^3 g^-1 s^-2/(H0/h) -> Mpc^3  (10^10 Msun)^-1
        GravpH = 6.672e-8/ 3.086e+24**3 * 1.989e+43 / (3.2407789e-18)**2
        #Total mass of DM in the box in internal mass units/h.
        omegatomass = self.box**3 / (8 * math.pi * GravpH)
        DMmass = (self.omega0 - self.omegab) * omegatomass / self.npart**3
        barmass = self.omegab * omegatomass / self.npart**3
        starmass = barmass/ config['Generations']
        config['SeedBHDynMass'] = DMmass * 1.5
        #This is set by the smallest observed SMBH so leave it alone.
        config['MinFoFMassForNewSeed'] = 0.5
        #This is basically "any stars" so leave it alone
        config['MinMStarForNewSeed'] = 2e-4
        #Real seed mass: no dynamical effect. Power law distributed.
        #In practice this only affects subgrid accretion so leave it alone.
        config['SeedBlackHoleMass'] = 3.0e-6
        config['MaxSeedBlackHoleMass'] = 3.0e-5
        config['SeedBlackHoleMassIndex'] = -2
        #Accretion scaling. Also affects feedback strength.
        config['BlackHoleAccretionFactor'] = 100.0
        config['BlackHoleEddingtonFactor'] = 2.1
        return self._heii_model_params(config)

    def generate_times(self):
        """Snapshot outputs for lyman alpha"""
        redshifts = np.arange(4.2, self.redend, -0.2)
        return 1./(1.+redshifts)

    def genicfile(self, camb_output):
        """Overload the genic file to also generate an HI table."""
        (genic_output, genic_param) = super().genicfile(camb_output)
        #Copy the metal cooling data
        metalcoolfile = os.path.join(self.outdir, self.metalcool)
        shutil.copytree(os.path.join(os.path.join(self.gadget_dir, "examples"), self.metalcool), metalcoolfile)
        return (genic_output, genic_param)
