"""Integration tests for the neutrinosimulation module"""

import os
import bigfile
import numpy as np
import configobj
from SimulationRunner import simulationics
from SimulationRunner import neutrinosimulation as nus

def test_neutrino_part():
    """Create a full simulation with particle neutrinos."""
    test_dir = os.path.join(os.getcwd(),"tests/test_nu/")
    Sim = nus.NeutrinoPartICs(outdir=test_dir,box = 256,npart = 128, m_nu = 0.45, redshift = 99, separate_gas=False, redend=0, nu_acc=1e-3)
    Sim.make_simulation()
    assert os.path.exists(test_dir)
    #Check these we are writing reasonable values.
    config = configobj.ConfigObj(os.path.join(test_dir,"_genic_params.ini"))
    assert config['Omega0'] == "0.288"
    assert config['OmegaLambda'] == "0.712"
    assert config['NgridNu'] == "128"
    assert config['MNue'] == '0.15'
    assert config['MNum'] == '0.15'
    assert config['MNut'] == '0.15'
    #Check that the output has neutrino particles
    f = bigfile.BigFile(os.path.join(test_dir,"ICS/256_128_99"),'r')
    assert f["Header"].attrs["TotNumPart"][2] == 128**3
    #Clean the test directory if test was successful
    #Check the mass is correct
    mcdm = f["Header"].attrs["MassTable"][1]
    mnu = f["Header"].attrs["MassTable"][2]
    #The mass ratio should be given by the ratio of omega_nu by omega_cdm
    assert np.abs(mnu/(mcdm+mnu) / ( (Sim.m_nu/93.151/Sim.hubble**2)/(Sim.omega0)) - 1) < 1e-5
    assert np.abs(f["Header"].attrs["MassTable"][1] / 61.7583 - 1) < 1e-5
    f.close()
    #shutil.rmtree("./test_nu/")

def test_neutrino_semilinear():
    """Create a full simulation with semi-linear neutrinos.
    The important thing here is to test that OmegaNu is correctly set."""
    test_dir = os.path.join(os.getcwd(),"tests/test_nu_semilin/")
    Sim = simulationics.SimulationICs(outdir=test_dir,box = 256,npart = 128, m_nu = 0.45, redshift = 99, separate_gas=False, nu_hierarchy='normal', redend=0, nu_acc=1e-3)
    Sim.make_simulation()
    assert os.path.exists(test_dir)
    #Check these files have not changed
    config = configobj.ConfigObj(os.path.join(test_dir,"_genic_params.ini"))
    assert config['Omega0'] == "0.288"
    assert config['OmegaLambda'] == "0.712"
    assert config['NgridNu'] == "0"
    assert config['MNue'] == '0.147144021962'
    assert config['MNum'] == '0.147399671639'
    assert config['MNut'] == '0.155456306399'

    config = configobj.ConfigObj(os.path.join(test_dir,"mpgadget.param"))
    assert config['MNue'] == '0.147144021962'
    assert config['MNum'] == '0.147399671639'
    assert config['MNut'] == '0.155456306399'
    assert config['MassiveNuLinRespOn'] == "1"
    assert config['LinearTransferFunction'] == "camb_linear/ics_transfer_99.dat"
    #Check that the output has no neutrino particles
    f = bigfile.BigFile(os.path.join(test_dir, "ICS/256_128_99"),'r')
    assert f["Header"].attrs["TotNumPart"][2] == 0
    #Check the mass is correct: the CDM particles should have the same mass as in the particle simulation
    assert np.abs(f["Header"].attrs["MassTable"][1] / 61.7583 - 1) < 1e-5
    f.close()
    #shutil.rmtree("./test_nu/")

def test_neutrino_mass_spec():
    """Check that our solution to the neutrino hierarchy is valid"""
    M21 = 7.53e-5 #Particle data group 2016: +- 0.18e-5 eV2
    M32n = 2.44e-3 #Particle data group: +- 0.06e-3 eV2
#     M32i = 2.51e-3
    numass = simulationics.get_neutrino_masses(0.3, 'degenerate')
    assert np.all(np.abs(numass-0.1) < 1e-6)
    numass = simulationics.get_neutrino_masses(0.3, 'normal')
    #Check the original inequalities are satisfied
    assert np.abs(numass[0]+numass[1]+numass[2] - 0.3) < 1e-4
    assert np.abs(numass[0]**2 - numass[1]**2 - M32n) < 1e-4
    assert np.abs(numass[1]**2 - numass[2]**2 - M21) < 1e-4
    numass = simulationics.get_neutrino_masses(0.08, 'normal')
    assert np.abs(numass[0]+numass[1]+numass[2] - 0.08) < 1e-4
    assert np.abs(numass[0]**2 - numass[1]**2 - M32n) < 1e-4
    assert np.abs(numass[1]**2 - numass[2]**2 - M21) < 1e-4
    numass = simulationics.get_neutrino_masses(0.11, 'inverted')
    assert np.abs(numass[0]+numass[1]+numass[2] - 0.11) < 1e-4
    assert np.abs(numass[0]**2 - numass[1]**2 + M32n) < 1e-4
    assert np.abs(numass[1]**2 - numass[2]**2 - M21) < 1e-4
    numass = simulationics.get_neutrino_masses(0.058333333, 'normal')
