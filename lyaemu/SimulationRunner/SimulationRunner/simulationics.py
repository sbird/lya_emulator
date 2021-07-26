"""Class to generate simulation ICS, separated out for clarity."""
from __future__ import print_function
import os.path
import math
import subprocess
import json
import shutil
#To do crazy munging of types for the storage format
import importlib
import numpy as np
import configobj
import classylss
import classylss.binding as CLASS
from . import utils
from . import clusters
from . import read_uvb_tab
from . import cambpower

class SimulationICs:
    """
    Class for creating the initial conditions for a simulation.
    There are a few things this class needs to do:

    - Generate CAMB input files
    - Generate MP-GenIC input files (to use CAMB output)
    - Run CAMB and MP-GenIC to generate ICs

    The class will store the parameters of the simulation.
    We also save a copy of the input and enough information to reproduce the resutls exactly in SimulationICs.json.
    Many things are left hard-coded.
    We assume flatness.

    Init parameters:
    outdir - Directory in which to save ICs
    box - Box size in comoving Mpc/h
    npart - Cube root of number of particles
    redshift - redshift at which to generate ICs
    separate_gas - if true the ICs will contain baryonic particles. If false, just DM.
    omegab - baryon density. Note that if we do not have gas particles, still set omegab, but set separate_gas = False
    omega0 - Total matter density at z=0 (includes massive neutrinos and baryons)
    hubble - Hubble parameter, h, which is H0 / (100 km/s/Mpc)
    scalar_amp - A_s at k = 0.05, comparable to the Planck value.
    ns - Scalar spectral index
    m_nu - neutrino mass
    unitary - if true, do not scatter modes, but use a unitary gaussian amplitude.
    """
    def __init__(self, *, outdir, box, npart, seed = 9281110, redshift=99, redend=0, separate_gas=True, omega0=0.288, omegab=0.0472, hubble=0.7, scalar_amp=2.427e-9, ns=0.97, rscatter=False, m_nu=0, nu_hierarchy='degenerate', uvb="pu", cluster_class=clusters.StampedeClass, nu_acc=1e-5, unitary=True, timelimit=1.5, nnode=2):
        #Check that input is reasonable and set parameters
        #In Mpc/h
        assert box < 20000
        self.box = box
        #Cube root
        assert 16000 > npart > 1
        self.npart = int(npart)
        #Physically reasonable
        assert 0 < omega0 <= 1
        self.omega0 = omega0
        assert 1 > omegab > 0
        self.omegab = omegab
        assert 1100 > redshift > 1
        self.redshift = redshift
        assert 0 <= redend < 1100
        self.redend = redend
        assert 0 < hubble < 1
        self.hubble = hubble
        assert 0 < scalar_amp < 1e-7
        self.scalar_amp = scalar_amp
        assert 2 > ns > 0
        self.ns = ns
        self.unitary = unitary
        #Neutrino accuracy for CLASS
        self.nu_acc = nu_acc
        #UVB? Only matters if gas
        self.uvb = uvb
        self.rscatter = rscatter
        outdir = os.path.realpath(os.path.expanduser(outdir))
        #Make the output directory: will fail if parent does not exist
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        else:
            if os.listdir(outdir) != []:
                print("Warning: ",outdir," is non-empty")
        #Structure seed.
        self.seed = seed
        #Baryons?
        self.separate_gas = separate_gas
        #If neutrinos are combined into the DM,
        #we want to use a different CAMB transfer when checking output power.
        self.separate_nu = False
        self.m_nu = m_nu
        self.nu_hierarchy = nu_hierarchy
        self.outdir = outdir
        self._set_default_paths()
        self._cluster = cluster_class(gadget=self.gadgetexe, param=self.gadgetparam, genic=self.genicexe, genicparam=self.genicout, nproc=nnode, timelimit=timelimit)
        #For repeatability, we store git hashes of Gadget, GenIC, CAMB and ourselves
        #at time of running.
        self.simulation_git = utils.get_git_hash(os.path.dirname(__file__))

    def _set_default_paths(self):
        """Default paths and parameter names."""
        #Default parameter file names
        self.gadgetparam = "mpgadget.param"
        self.genicout = "_genic_params.ini"
        #Executable names
        self.gadgetexe = "MP-Gadget"
        self.genicexe = "MP-GenIC"
        defaultpath = os.path.dirname(__file__)
        #Default GenIC paths
        self.genicdefault = os.path.join(defaultpath,"mpgenic.ini")
        self.gadgetconfig = "Options.mk"
        self.gadget_dir = os.path.expanduser("~/codes/MP-Gadget/")

    def cambfile(self):
        """Generate the IC power spectrum using classylss."""
        #Load high precision defaults
        pre_params = {'tol_background_integration': 1e-9, 'tol_perturb_integration' : 1.e-7, 'tol_thermo_integration':1.e-5, 'k_per_decade_for_pk': 50,'k_bao_width': 8, 'k_per_decade_for_bao':  200, 'neglect_CMB_sources_below_visibility' : 1.e-30, 'transfer_neglect_late_source': 3000., 'l_max_g' : 50, 'l_max_ur':150, 'extra metric transfer functions': 'y'}
        #Set the neutrino density and subtract it from omega0
        omeganu = self.m_nu/93.14/self.hubble**2
        omcdm = (self.omega0 - self.omegab) - omeganu
        gparams = {'h':self.hubble, 'Omega_cdm':omcdm,'Omega_b': self.omegab, 'Omega_k':0, 'n_s': self.ns, 'A_s': self.scalar_amp}
        #Lambda is computed self-consistently
        gparams['Omega_fld'] = 0
        numass = get_neutrino_masses(self.m_nu, self.nu_hierarchy)
        #Set up massive neutrinos
        if self.m_nu > 0:
            gparams['m_ncdm'] = '%.8f,%.8f,%.8f' % (numass[2], numass[1], numass[0])
            gparams['N_ncdm'] = 3
            gparams['N_ur'] = 0.00641
            #Neutrino accuracy: Default pk_ref.pre has tol_ncdm_* = 1e-10,
            #which takes 45 minutes (!) on my laptop.
            #tol_ncdm_* = 1e-8 takes 20 minutes and is machine-accurate.
            #Default parameters are fast but off by 2%.
            #I chose 1e-5, which takes 6 minutes and is accurate to 1e-5
            gparams['tol_ncdm_newtonian'] = min(self.nu_acc,1e-5)
            gparams['tol_ncdm_synchronous'] = self.nu_acc
            gparams['tol_ncdm_bg'] = 1e-10
            gparams['l_max_ncdm'] = 50
            #This disables the fluid approximations, which make P_nu not match camb on small scales.
            #We need accurate P_nu to initialise our neutrino code.
            gparams['ncdm_fluid_approximation'] = 2
            #Does nothing unless ncdm_fluid_approximation = 2
            #Spend less time on neutrino power for smaller neutrino mass
            gparams['ncdm_fluid_trigger_tau_over_tau_k'] = 30000.* (self.m_nu / 0.4)
        else:
            gparams['N_ur'] = 3.046
        #Initial cosmology
        pre_params.update(gparams)
        maxk = 2*math.pi/self.box*self.npart*8
        powerparams = {'output': 'dTk vTk mPk', 'P_k_max_h/Mpc' : maxk, "z_max_pk" : self.redshift+1}
        pre_params.update(powerparams)

        #At which redshifts should we produce CAMB output: we want the start and end redshifts of the simulation,
        #but we also want some other values for checking purposes
        camb_zz = np.concatenate([[self.redshift,], 1/self.generate_times()-1,[self.redend,]])

        cambpars = os.path.join(self.outdir, "_class_params.ini")
        classconf = configobj.ConfigObj()
        classconf.filename = cambpars
        classconf.update(pre_params)
        classconf['z_pk'] = camb_zz
        classconf.write()

        engine = CLASS.ClassEngine(pre_params)
        powspec = CLASS.Spectra(engine)
        #Save directory
        camb_output = "camb_linear/"
        camb_outdir = os.path.join(self.outdir,camb_output)
        try:
            os.mkdir(camb_outdir)
        except FileExistsError:
            pass
        #Save directory
        #Get and save the transfer functions
        for zz in camb_zz:
            trans = powspec.get_transfer(z=zz)
            #fp-roundoff
            trans['k'][-1] *= 0.9999
            transferfile = os.path.join(camb_outdir, "ics_transfer_"+self._camb_zstr(zz)+".dat")
            save_transfer(trans, transferfile)
            pk_lin = powspec.get_pklin(k=trans['k'], z=zz)
            pkfile = os.path.join(camb_outdir, "ics_matterpow_"+self._camb_zstr(zz)+".dat")
            np.savetxt(pkfile, np.vstack([trans['k'], pk_lin]).T)

        return camb_output

    def _camb_zstr(self,zz):
        """Get the formatted redshift for CAMB output files."""
        if zz > 10:
            zstr = str(int(zz))
        else:
            zstr = '%.1g' % zz
        return zstr

    def genicfile(self, camb_output):
        """Generate the GenIC parameter file"""
        config = configobj.ConfigObj(self.genicdefault)
        config.filename = os.path.join(self.outdir, self.genicout)
        config['BoxSize'] = self.box*1000
        genicout = "ICS"
        try:
            os.mkdir(os.path.join(self.outdir, genicout))
        except FileExistsError:
            pass
        config['OutputDir'] = genicout
        #Is this enough information, or should I add a short hash?
        genicfile = str(self.box)+"_"+str(self.npart)+"_"+str(self.redshift)
        config['FileBase'] = genicfile
        config['Ngrid'] = self.npart
        config['NgridNu'] = 0
        #config['MaxMemSizePerNode'] = 0.8
        config['ProduceGas'] = int(self.separate_gas)
        #Suppress Gaussian mode scattering
        config['UnitaryAmplitude'] = int(self.unitary)
        #The 2LPT correction is computed for one fluid. It is not clear
        #what to do with a second particle species, so turn it off.
        #Even for CDM alone there are corrections from radiation:
        #order: Omega_r / omega_m ~ 3 z/100 and it is likely
        #that the baryon 2LPT term is dominated by the CDM potential
        #(YAH, private communication)
        #Total matter density, not CDM matter density.
        config['Omega0'] = self.omega0
        config['OmegaLambda'] = 1- self.omega0
        config['OmegaBaryon'] = self.omegab
        config['HubbleParam'] = self.hubble
        config['Redshift'] = self.redshift
        zstr = self._camb_zstr(self.redshift)
        config['FileWithInputSpectrum'] = camb_output + "ics_matterpow_"+zstr+".dat"
        config['FileWithTransferFunction'] = camb_output + "ics_transfer_"+zstr+".dat"
        numass = get_neutrino_masses(self.m_nu, self.nu_hierarchy)
        config['MNue'] = numass[2]
        config['MNum'] = numass[1]
        config['MNut'] = numass[0]
        config['SavePrePos'] = 0
        assert config['WhichSpectrum'] == '2'
        assert config['RadiationOn'] == '1'
        assert config['DifferentTransferFunctions'] == '1'
        assert config['InputPowerRedshift'] == '-1'
        config['Seed'] = self.seed
        config = self._genicfile_child_options(config)
        config.update(self._cluster.cluster_runtime())
        config.write()
        return (os.path.join(genicout, genicfile), config.filename)

    def _alter_power(self, camb_output):
        """Function to hook if you want to change the CAMB output power spectrum.
        Should save the new power spectrum to camb_output + _matterpow_str(redshift).dat"""
        zstr = self._camb_zstr(self.redshift)
        camb_file = os.path.join(camb_output,"ics_matterpow_"+zstr+".dat")
        os.stat(camb_file)

    def _genicfile_child_options(self, config):
        """Set extra parameters in child classes"""
        return config

    def _fromarray(self):
        """Convert the data stored as lists back to what it was."""
        for arr in self._really_arrays:
            self.__dict__[arr] = np.array(self.__dict__[arr])
        self._really_arrays = []
        for arr in self._really_types:
            #Some crazy nonsense to convert the module, name
            #string tuple we stored back into a python type.
            mod = importlib.import_module(self.__dict__[arr][0])
            self.__dict__[arr] = getattr(mod, self.__dict__[arr][1])
        self._really_types = []

    def txt_description(self):
        """Generate a text file describing the parameters of the code that generated this simulation, for reproducibility."""
        #But ditch the output of make
        self.make_output = ""
        self._really_arrays = []
        self._really_types = []
        cc = self._cluster
        self._cluster = 0
        for nn, val in self.__dict__.items():
            #Convert arrays to lists
            if isinstance(val, np.ndarray):
                self.__dict__[nn] = val.tolist()
                self._really_arrays.append(nn)
            #Convert types to string tuples
            if isinstance(val, type):
                self.__dict__[nn] = (val.__module__, val.__name__)
                self._really_types.append(nn)
        with open(os.path.join(self.outdir, "SimulationICs.json"), 'w') as jsout:
            json.dump(self.__dict__,jsout)
        #Turn the changed types back.
        self._fromarray()
        self._cluster = cc

    def load_txt_description(self):
        """Load the text file describing the parameters of the code that generated a simulation."""
        cc = self._cluster
        with open(os.path.join(self.outdir, "SimulationICs.json"), 'r') as jsin:
            self.__dict__ = json.load(jsin)
        self._fromarray()
        self._cluster = cc

    def gadget3config(self, prefix="OPT += -D"):
        """Generate a config Options file for MP-Gadget.
        This code is configured via runtime options."""
        g_config_filename = os.path.join(self.outdir, self.gadgetconfig)
        with open(g_config_filename,'w') as config:
            optimize = self._cluster.cluster_optimize()
            config.write("OPTIMIZE = "+optimize+"\n")
            self._cluster.cluster_config_options(config, prefix)
            self._gadget3_child_options(config, prefix)
        return g_config_filename

    def _gadget3_child_options(self, _, __):
        """Gadget-3 compilation options for Config.sh which should be written by the child class
        This is MP-Gadget, so it is likely there are none."""

    def gadget3params(self, genicfileout):
        """MP-Gadget parameter file. This *is* a configobj.
        Note MP-Gadget supprts default arguments, so no need for a defaults file.
        Arguments:
            genicfileout - where the ICs are saved
        """
        config = configobj.ConfigObj()
        filename = os.path.join(self.outdir, self.gadgetparam)
        config.filename = filename
        config['InitCondFile'] = genicfileout
        config['OutputDir'] = "output"
        try:
            os.mkdir(os.path.join(self.outdir, "output"))
        except FileExistsError:
            pass
        config['TimeLimitCPU'] = int(60*60*self._cluster.timelimit-300)
        config['TimeMax'] = 1./(1+self.redend)
        config['Omega0'] = self.omega0
        config['OmegaLambda'] = 1- self.omega0
        #OmegaBaryon should be zero for gadget if we don't have gas particles
        config['OmegaBaryon'] = self.omegab*self.separate_gas
        config['HubbleParam'] = self.hubble
        config['RadiationOn'] = 1
        config['HydroOn'] = 1
        #Neutrinos
        if self.m_nu > 0:
            config['MassiveNuLinRespOn'] = 1
        else:
            config['MassiveNuLinRespOn'] = 0
        numass = get_neutrino_masses(self.m_nu, self.nu_hierarchy)
        config['MNue'] = numass[2]
        config['MNum'] = numass[1]
        config['MNut'] = numass[0]
        #FOF
        config['SnapshotWithFOF'] = 1
        config['FOFHaloLinkingLength'] = 0.2
        config['OutputList'] =  ','.join([str(t) for t in self.generate_times()])
        #These are only used for gas, but must be set anyway
        config['MinGasTemp'] = 100
        #In equilibrium with the CMB at early times.
        config['InitGasTemp'] = 2.7*(1+self.redshift)
        config['DensityIndependentSphOn'] = 1
        config['PartAllocFactor'] = 2
        config['WindOn'] = 0
        config['WindModel'] = 'nowind'
        config['BlackHoleOn'] = 0
        config['MetalReturnOn'] = 0
        config['OutputPotential'] = 0
        if self.separate_gas:
            config['CoolingOn'] = 1
            #Copy a TREECOOL file into the right place.
            config['TreeCoolFile'] = self._copy_uvb()
            config = self._sfr_params(config)
            config = self._feedback_params(config)
        else:
            config['CoolingOn'] = 0
            config['StarformationOn'] = 0
        #Add other config parameters
        config = self._other_params(config)
        config.update(self._cluster.cluster_runtime())
        config.write()
        return config

    def _sfr_params(self, config):
        """Config parameters for the default Springel & Hernquist star formation model"""
        config['StarformationOn'] = 1
        config['StarformationCriterion'] = 'density'
        return config

    def _feedback_params(self, config):
        """Config parameters for the feedback models"""
        return config

    def _other_params(self, config):
        """Function to override to set other config parameters"""
        return config

    def generate_times(self):
        """List of output times for a simulation. Can be overridden."""
        astart = 1./(1+self.redshift)
        aend = 1./(1+self.redend)
        times = np.array([0.02,0.1,0.2,0.25,0.3333,0.5,0.66667,0.83333])
        ii = np.where((times > astart)*(times < aend))
        assert np.size(times[ii]) > 0
        return times[ii]

    def _copy_uvb(self):
        """The UVB amplitude for Gadget is specified in a file named TREECOOL in the same directory as the gadget binary."""
        fuvb = read_uvb_tab.get_uvb_filename(self.uvb)
        baseuvb = os.path.basename(fuvb)
        shutil.copy(fuvb, os.path.join(self.outdir,baseuvb))
        return baseuvb

    def do_gadget_build(self, gadget_config):
        """Make a gadget build and check it succeeded."""
        conffile = os.path.join(self.gadget_dir, self.gadgetconfig)
        if os.path.islink(conffile):
            os.remove(conffile)
        if os.path.exists(conffile):
            os.rename(conffile, conffile+".backup")
        os.symlink(gadget_config, conffile)
        #Build gadget
        gadget_binary = os.path.join(os.path.join(self.gadget_dir, "gadget"), self.gadgetexe)
        try:
            g_mtime = os.stat(gadget_binary).st_mtime
        except FileNotFoundError:
            g_mtime = -1
        self.gadget_git = utils.get_git_hash(gadget_binary)
        try:
            self.make_output = subprocess.check_output(["make", "-j"], cwd=self.gadget_dir, universal_newlines=True, stderr=subprocess.STDOUT, shell=True)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise
        #Check that the last-changed time of the binary has actually changed..
        assert g_mtime != os.stat(gadget_binary).st_mtime
        shutil.copy(gadget_binary, os.path.join(os.path.dirname(gadget_config),self.gadgetexe))

    def generate_mpi_submit(self, genicout):
        """Generate a sample mpi_submit file.
        The prefix argument is a string at the start of each line.
        It separates queueing system directives from normal comments"""
        self._cluster.generate_mpi_submit(self.outdir)
        #Generate an mpi_submit for genic
        zstr = self._camb_zstr(self.redshift)
        check_ics = "#python3 cambpower.py "+genicout+" --czstr "+zstr+" --mnu "+str(self.m_nu)
        self._cluster.generate_mpi_submit_genic(self.outdir, extracommand=check_ics)
        #Copy the power spectrum routine
        shutil.copy(os.path.join(os.path.dirname(__file__),"cambpower.py"), os.path.join(self.outdir,"cambpower.py"))

    def make_simulation(self, pkaccuracy=0.05, do_build=False):
        """Wrapper function to make the simulation ICs."""
        #First generate the input files for CAMB
        camb_output = self.cambfile()
        #Then run CAMB
        self.camb_git = classylss.__version__
        #Change the power spectrum file on disc if we want to do that
        self._alter_power(os.path.join(self.outdir,camb_output))
        #Now generate the GenIC parameters
        (genic_output, genic_param) = self.genicfile(camb_output)
        #Save a json of ourselves.
        self.txt_description()
        #Check that the ICs have the right power spectrum
        #Generate Gadget makefile
        gadget_config = self.gadget3config()
        #Symlink the new gadget config to the source directory
        #Generate Gadget parameter file
        self.gadget3params(genic_output)
        #Generate mpi_submit file
        self.generate_mpi_submit(genic_output)
        #Run MP-GenIC
        if do_build:
            subprocess.check_call([os.path.join(os.path.join(self.gadget_dir, "genic"),self.genicexe), genic_param],cwd=self.outdir)
            zstr = self._camb_zstr(self.redshift)
            cambpower.check_ic_power_spectra(genic_output, camb_zstr=zstr, m_nu=self.m_nu, outdir=self.outdir, accuracy=pkaccuracy)
            self.do_gadget_build(gadget_config)
        return gadget_config

def save_transfer(transfer, transferfile):
    """Save a transfer function. Note we save the CLASS FORMATTED transfer functions.
    The transfer functions differ from CAMB by:
        T_CAMB(k) = -T_CLASS(k)/k^2 """
    header="""Transfer functions T_i(k) for adiabatic (AD) mode (normalized to initial curvature=1)
d_i   stands for (delta rho_i/rho_i)(k,z) with above normalization
d_tot stands for (delta rho_tot/rho_tot)(k,z) with rho_Lambda NOT included in rho_tot
(note that this differs from the transfer function output from CAMB/CMBFAST, which gives the same
 quantities divided by -k^2 with k in Mpc^-1; use format=camb to match CAMB)
t_i   stands for theta_i(k,z) with above normalization
t_tot stands for (sum_i [rho_i+p_i] theta_i)/(sum_i [rho_i+p_i]))(k,z)
1:k (h/Mpc)              2:d_g                    3:d_b                    4:d_cdm                  5:d_ur        6:d_ncdm[0]              7:d_ncdm[1]              8:d_ncdm[2]              9:d_tot                 10:phi     11:psi                   12:h                     13:h_prime               14:eta                   15:eta_prime     16:t_g                   17:t_b                   18:t_ur        19:t_ncdm[0]             20:t_ncdm[1]             21:t_ncdm[2]             22:t_tot"""
    #This format matches the default output by CLASS command line.
    np.savetxt(transferfile, transfer, header=header)

def get_neutrino_masses(total_mass, hierarchy):
    """Get the three neutrino masses, including the mass splittings.
        Hierarchy is 'inverted' (two heavy), 'normal' (two light) or degenerate."""
    #Neutrino mass splittings
    nu_M21 = 7.53e-5 #Particle data group 2016: +- 0.18e-5 eV2
    nu_M32n = 2.44e-3 #Particle data group: +- 0.06e-3 eV2
    nu_M32i = 2.51e-3 #Particle data group: +- 0.06e-3 eV2

    if hierarchy == 'normal':
        nu_M32 = nu_M32n
        #If the total mass is below that allowed by the hierarchy,
        #assign one active neutrino.
        if total_mass < np.sqrt(nu_M32n) + np.sqrt(nu_M21):
            return np.array([total_mass, 0, 0])
    elif hierarchy == 'inverted':
        nu_M32 = -nu_M32i
        if total_mass < 2*np.sqrt(nu_M32i) - np.sqrt(nu_M21):
            return np.array([total_mass/2., total_mass/2., 0])
    #Hierarchy == 0 is 3 degenerate neutrinos
    else:
        return np.ones(3)*total_mass/3.

    #DD is the summed masses of the two closest neutrinos
    DD1 = 4 * total_mass/3. - 2/3.*np.sqrt(total_mass**2 + 3*nu_M32 + 1.5*nu_M21)
    #Last term was neglected initially. This should be very well converged.
    DD = 4 * total_mass/3. - 2/3.*np.sqrt(total_mass**2 + 3*nu_M32 + 1.5*nu_M21+0.75*nu_M21**2/DD1**2)
    nu_masses = np.array([ total_mass - DD, 0.5*(DD + nu_M21/DD), 0.5*(DD - nu_M21/DD)])
    assert np.isfinite(DD)
    assert np.abs(DD1/DD -1) < 2e-2
    assert np.all(nu_masses >= 0)
    return nu_masses
