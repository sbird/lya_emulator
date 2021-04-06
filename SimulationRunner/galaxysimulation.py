"""Specialization of the Simulation class to galaxy formation simulations."""

#TODO: - Seed BH masses increased to match density.
#TODO: Which parameters to vary?

import os
import os.path
import shutil
import numpy as np
from . import make_HI_reionization_table as hi
from . import simulationics
from . import lyasimulation

class GalaxySim(lyasimulation.LymanAlphaSim):
    """Specialise the Simulation class to do full physics runs with galaxy formation (stars, AGN, winds) enabled.
    Mirrors the model used in ASTERIX. No massive neutrinos.
    Extra parameters:
        bhfeedback - Amount of BH feedback."""
    __doc__ = __doc__+simulationics.SimulationICs.__doc__
    def __init__(self, *, bhfeedback = 0.05, **kwargs):
        #super generates the helium reionization table
        super().__init__(**kwargs)
        self.metalcool = "cooling_metal_UVB"
        self.bhfeedback = bhfeedback

    def _feedback_config_options(self, config, prefix=""):
        """Config options specific to the Lyman alpha forest star formation criterion"""
        #No feedback!
        _,_=config,prefix

    def _feedback_params(self, config):
        """Galaxy formation config options from ASTERIX."""
        #Stellar feedback parameters
        config['StarformationCriterion'] = 'density' #Note no h2 star formation! Different from ASTERIX.
        config['WindModel'] = 'ojft10,decouple'
        config['WindEfficiency'] = 2.0
        config['WindOn'] = 1
        config['MetalCoolFile'] = self.metalcool
        config['WindEnergyFraction'] = 1.0
        config['WindSigma0'] = 353.0 #km/s
        config['WindSpeedFactor'] = 3.7
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
        #FIXME: Make this scale with mass of a DM particle.
        config['SeedBHDynMass'] = 1e-3
        config['MinFoFMassForNewSeed'] = 0.5
        config['MinMStarForNewSeed'] = 2e-4
        #Real seed mass: no dynamical effect. Power law distributed.
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
