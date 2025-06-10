"""Specialization of the Simulation class to Lyman-alpha forest simulations."""

from . import simulationics

class NeutrinoPartICs(simulationics.SimulationICs):
    """Specialise the initial conditions for particle neutrinos."""
    __doc__ = __doc__+simulationics.SimulationICs.__doc__
    def __init__(self, *, m_nu=0.1, separate_gas=False, **kwargs):
        #Set neutrino mass
        #Note that omega0 does remains constant if we change m_nu.
        #This does mean that omegab/omegac will increase, but not by much.
        assert m_nu > 0
        super().__init__(m_nu = m_nu, separate_gas=separate_gas, **kwargs)
        self.separate_nu = True

    def _genicfile_child_options(self, config):
        """Set up particle neutrino parameters for GenIC"""
        config['NgridNu'] = self.npart
        #Degenerate neutrinos
        return config

    def _other_params(self, config):
        """Config options to make type 2 particles neutrinos."""
        #Specify neutrino masses so that omega_nu is correct
        config['MassiveNuLinRespOn'] = 0
        return config

class NeutrinoHybridICs(simulationics.SimulationICs):
    """Further specialise the NeutrinoPartICs class for semi-linear analytic massive neutrinos.
    """
    def __init__(self, *, npartnufac=0.5, vcrit=850, zz_transition = 1, **kwargs):
        self.npartnufac = npartnufac
        self.vcrit = vcrit
        self.separate_nu = True
        self.zz_transition = zz_transition
        super().__init__(**kwargs)

    def _genicfile_child_options(self, config):
        """Set up hybrid neutrino parameters for GenIC."""
        #Degenerate neutrinos
        config['NgridNu'] = int(self.npart*self.npartnufac)
        config['Max_nuvel'] = self.vcrit
        return config

    def _other_params(self, config):
        """Config options specific to kspace neutrinos. Hierarchy is off for now."""
        config['HybridNeutrinosOn'] = 1
        config['Vcrit'] = self.vcrit
        config['NuPartTime'] = 1./(1+self.zz_transition)
        return config
