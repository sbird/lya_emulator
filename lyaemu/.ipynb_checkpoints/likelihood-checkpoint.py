"""Module for computing the likelihood function for the forest emulator."""
from datetime import datetime
import numpy as np
import scipy.interpolate
from . import coarse_grid
from . import flux_power
from . import lyman_data
from . import mean_flux as mflux
from .quadratic_emulator import QuadraticEmulator
from cobaya.likelihood import Likelihood
from cobaya.run import run as cobaya_run
from cobaya.log import LoggedError
from mpi4py import MPI
import json
import h5py
import os
from .meanT import t0_likelihood

def _siIIIcorr(kf):
    """For precomputing the shape of the SiIII correlation"""
    #Compute bin boundaries in logspace.
    kmids = np.zeros(np.size(kf)+1)
    kmids[1:-1] = np.exp((np.log(kf[1:])+np.log(kf[:-1]))/2.)
    #arbitrary final point
    kmids[-1] = 2*np.pi/2271 + kmids[-2]
    # This is the average of cos(2271k) across the k interval in the bin
    siform = np.zeros_like(kf)
    siform = (np.sin(2271*kmids[1:])-np.sin(2271*kmids[:-1]))/(kmids[1:]-kmids[:-1])/2271.
    #Correction for the zeroth bin, because the integral is oscillatory there.
    siform[0] = np.cos(2271*kf[0])
    return siform

def SiIIIcorr(fSiIII, tau_eff, kf):
    """The correction for SiIII contamination, as per McDonald."""
    assert tau_eff > 0
    aa = fSiIII/(1-np.exp(-tau_eff))
    return 1 + aa**2 + 2 * aa * _siIIIcorr(kf)

def DLAcorr(kf, z, alpha):
    """The correction for DLA contamination, from arXiv:1706.08532"""
    # fit values and pivot redshift directly from arXiv:1706.08532
    z_0 = 2
    # parameter order: LLS, Sub-DLA, Small-DLA, Large-DLA
    a_0 = np.array([2.2001, 1.5083, 1.1415, 0.8633])
    a_1 = np.array([0.0134, 0.0994, 0.0937, 0.2943])
    b_0 = np.array([36.449, 81.388, 162.95, 429.58])
    b_1 = np.array([-0.0674, -0.2287, 0.0126, -0.4964])
    # compute the z-dependent correction terms
    a_z = a_0 * ((1+z)/(1+z_0))**a_1
    b_z = b_0 * ((1+z)/(1+z_0))**b_1
    factor = np.ones(kf.size) # alpha_0 degenerate with mean flux, set to 1
    # a_lls and a_sub degenerate with each other (as are a_sdla, a_ldla), so only use two values
    factor += alpha[0] * ((1+z)/(1+z_0))**-3.55 * ((a_z[0]*np.exp(b_z[0]*kf) - 1)**-2 + (a_z[1]*np.exp(b_z[1]*kf) - 1)**-2)
    factor += alpha[1] * ((1+z)/(1+z_0))**-3.55 * ((a_z[2]*np.exp(b_z[2]*kf) - 1)**-2 + (a_z[3]*np.exp(b_z[3]*kf) - 1)**-2)
    return factor

def load_data(datadir, *, kfkms, kfmpc, zout, max_z=4.6, min_z=2.2, t0=1., tau_thresh=None, data_index=21):
    """Load and initialise a "fake data" flux power spectrum"""
    try: # first, try loading an existing file for the flux power
        zinds = np.where((min_z <= zout)*(max_z >= zout))[0]
        data_hdf5 = h5py.File(datadir+'/mf_emulator_flux_vectors_tau1000000.hdf5', 'r')
        dfp = data_hdf5['flux_vectors'][data_index].reshape(zout.size, -1)[zinds].flatten()
        params = data_hdf5['params'][data_index]
        data_hdf5.close()
        omega_m = params[7]/params[6]**2
        _, data_fluxpower = flux_power.rebin_power_to_kms(kfkms=kfkms, kfmpc=kfmpc, flux_powers=dfp, zbins=zout[zinds], omega_m=omega_m)
    except:
        #Load the data directory
        myspec = flux_power.MySpectra(max_z=max_z, min_z=min_z)
        pps = myspec.get_snapshot_list(datadir)
        #self.data_fluxpower is used in likelihood.
        data_fluxpower = pps.get_power(kf=kfkms, mean_fluxes=np.exp(-t0*mflux.obs_mean_tau(myspec.zout, amp=0)), tau_thresh=tau_thresh)
        assert np.size(data_fluxpower) % np.size(kfkms) == 0
    return data_fluxpower.flatten()

class LikelihoodClass:
    """Class to contain likelihood computations."""
    def __init__(self, basedir, mean_flux='s', max_z=4.6, min_z=2.2, emulator_class="standard", t0_training_value=1., optimise_GP=True, emulator_json_file='emulator_params.json', data_corr=True, tau_thresh=None, use_meant=False, traindir=None, HRbasedir=None, loo_errors=False):
        """Initialise the emulator by loading the flux power spectra from the simulations.
        Parameters:
        - basedir: directory to load emulator
        - mean_flux: whether to use (redshift dependent) mean flux rescaling. 's' uses rescaling, 'c' does not. Must match flux power file.
        - min_z, max_z: minimum and maximum redshift to include in the likelihood
        - emulator_class: type of emulator to use: quadratic, knot, or (default) GP
        - optimise_GP: whether to train the GPs (if sampling, set initial call to false to avoid training twice)
        - emulator_json_file: json file containing various emulator settings and parameters
        - data_corr: whether to include corrections for DLAs and SiIII
        - tau_thresh: optical depth threshold for computing flux power. Must match previously calculated flux power file, or will generate a new one.
        - use_meant: whether to include the mean temperature emulator.
        - traindir: directory where trained GPs reside. If no trained GP in that directory, train then save at that location.
        - HRbasedir: directory to load high resolution emulator, turns this into a multi-fidelity emulator, with predictions for the HR outputs.
        - loo_errors: whether to use leave-one-out errors in place of emulator errors. File with errors must be in same directory as emulator files, and be called 'loo_fps.hdf5'
        """
        # Needed for Cobaya dictionary construction
        self.basedir, self.HRbasedir = basedir, HRbasedir
        self.emulator_json_file, self.traindir = emulator_json_file, traindir
        self.mean_flux, self.data_corr, self.tau_thresh = mean_flux, data_corr, tau_thresh
        self.use_meant, self.loo_errors = use_meant, loo_errors
        self.max_z, self.min_z = max_z, min_z
        self.t0_training_value, self.emulator_class = t0_training_value, emulator_class
        myspec = flux_power.MySpectra(max_z=max_z, min_z=min_z)
        self.zout = myspec.zout
        # Default data is flux power from Chabanier 2019 (BOSS DR14),
        # pass datafile='dr9' to use data from Palanque-Delabrouille 2013
        self.sdss = lyman_data.BOSSData()
        self.kf = self.sdss.get_kf()
        # Load BOSS data vector
        zbins = np.where((self.sdss.get_redshifts() <= self.max_z)*(self.sdss.get_redshifts() >= self.min_z))
        self.BOSS_flux_power = self.sdss.pf.reshape(-1, self.kf.shape[0])[::-1][zbins]
        # Units: km / s; Size: n_z * n_k
        self.mf_slope = False
        # get leave_one_out errors
        if loo_errors: self.get_loo_errors()
        # Param limits on t0
        t0_factor = np.array([0.75, 1.25])
        if mean_flux == 'c':
            mf = mflux.ConstMeanFlux(value=t0_training_value)
        # Redshift bins are independent -- for redshift-dependent mean flux models
        # we convert the input parameters to a list of mean flux scalings in each redshift bin.
        # This parametrises the mean flux as an amplitude and slope.
        elif mean_flux == 's':
            # Add a slope to the parameter limits
            t0_slope = np.array([-0.4, 0.25])
            self.mf_slope = True
            # Get the min_z and max_z for the emulator, regardless of what is requested
            with open(basedir+"/"+emulator_json_file, "r") as emulator_json:
                loaded = json.load(emulator_json)
                nz = int(np.round((loaded["max_z"]-loaded["min_z"])/0.2, 1)) + 1
                z_mflux = np.linspace(loaded["min_z"], loaded["max_z"], nz)
            slopehigh = np.max(mflux.mean_flux_slope_to_factor(z_mflux, t0_slope[1]))
            slopelow = np.min(mflux.mean_flux_slope_to_factor(z_mflux, t0_slope[0]))
            dense_limits = np.array([np.array(t0_factor) * np.array([slopelow, slopehigh])])
            mf = mflux.MeanFluxFactor(dense_limits=dense_limits)
        else:
            mf = mflux.MeanFluxFactor()
        # Select emulator type
        if emulator_class == "standard":
            self.emulator = coarse_grid.Emulator(basedir, kf=self.kf, mf=mf, tau_thresh=tau_thresh)
        elif emulator_class == "knot":
            self.emulator = coarse_grid.KnotEmulator(basedir, kf=self.kf, mf=mf)
        elif emulator_class == "quadratic":
            self.emulator = QuadraticEmulator(basedir, kf=self.kf, mf=mf)
        else:
            raise ValueError("Emulator class not recognised")

        # Load the parameters, etc. associated with this emulator (overwrite defaults)
        self.emulator.load(dumpfile=emulator_json_file)
        self.param_limits = self.emulator.get_param_limits(include_dense=True)
        if mean_flux == 's':
            # Add a slope to the parameter limits
            self.param_limits = np.vstack([t0_slope, self.param_limits])
            # Shrink param limits t0 so that they are within the emulator range
            self.param_limits[1, :] = t0_factor

        # Set up SiIII and DLA corrections, if requested
        self.data_params = {}
        if data_corr:
            self.ndim = np.shape(self.param_limits)[0]
            # Create some useful objects for implementing the DLA and SiIII corrections
            self.dnames = [('a_lls', r'\alpha_{lls}'), ('a_dla', r'\alpha_{dla}'), ('fSiIII', 'fSiIII')]
            self.data_params = {self.dnames[i][0]:np.arange(self.ndim, self.ndim+np.shape(self.dnames)[0])[i] for i in range(np.shape(self.dnames)[0])}
            # Limits for the data correction parameters
            alpha_limits = np.array([[-1.0, 1.0], [-0.3, 0.3]])
            fSiIII_limits = np.array([-0.03, 0.03])
            self.param_limits = np.vstack([self.param_limits, alpha_limits, fSiIII_limits])
        self.ndim = np.shape(self.param_limits)[0]
        assert np.shape(self.param_limits)[1] == 2

        # Set up MPI protections (as suggested in Cobaya documentation)
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        # Generate emulator
        if optimise_GP:
            gpemu = None
            if rank == 0:
                #Build the emulator only on rank 0 and broadcast
                print('Beginning to generate emulator at', str(datetime.now()))
                if HRbasedir is None:
                    gpemu = self.emulator.get_emulator(max_z=max_z, min_z=min_z, traindir=traindir)
                else:
                    gpemu = self.emulator.get_MFemulator(HRbasedir, max_z=max_z, min_z=min_z, traindir=traindir)
                print('Finished generating emulator at', str(datetime.now()))
            self.gpemu = comm.bcast(gpemu, root = 0)
        if use_meant:
            assert self.min_z <= 3.8, "Emulator does not support temperatures outside 2.2 < z < 3.8"
            self.meant_gpemu = t0_likelihood.T0LikelihoodClass(self.basedir, max_z=np.min([3.8, self.max_z]), min_z=self.min_z, optimise_GP=optimise_GP, HRbasedir=self.HRbasedir, loo_errors=loo_errors)

    def get_loo_errors(self, savefile="loo_fps.hdf5"):
        if self.HRbasedir is None:
            filepath = os.path.join(self.basedir, savefile)
        else:
            filepath = os.path.join(self.HRbasedir, savefile)
        ff = h5py.File(filepath, 'r')
        fpp, fpt, looz = ff['flux_predict'][:], ff['flux_true'][:], ff['zout'][:]
        ff.close()
        zinds = np.where([(looz <= self.max_z)*(looz >= self.min_z)])[1]
        # after loading the absolute difference, calculate errors including BOSS data
        loo_errors = np.mean(np.abs(fpp - fpt)[:, zinds], axis=0)
        nz = np.shape(loo_errors)[0]
        self.icov_bin = []
        self.cdet = []
        for bb in range(nz):
            covar_bin = self.get_BOSS_error(bb)
            covar_bin += np.outer(loo_errors[bb], loo_errors[bb])
            self.icov_bin.append(np.linalg.inv(covar_bin))
            self.cdet.append(np.linalg.slogdet(covar_bin)[1])
        return loo_errors

    def calculate_loo_errors(self, savefile='loo_fps.hdf5'):
        """Calculate leave-one-out errors: saves predicted flux power, true flux power,
        predicted error, and parameters for each simulation in the training set."""
        # call to loo function in coarse_grid with appropriate settings
        if self.HRbasedir is not None:
            HRemu = coarse_grid.Emulator(self.HRbasedir, kf=self.kf, mf=self.emulator.mf, tau_thresh=self.tau_thresh)
            predict, std, true, params = self.emulator.generate_loo_errors(HRemu=HRemu, min_z=self.min_z, max_z=self.max_z)
            savepath = self.HRbasedir
        else:
            predict, std, true, params = self.emulator.generate_loo_errors(min_z=self.min_z, max_z=self.max_z)
            savepath = self.basedir
        # save into hdf5 file
        savefile = h5py.File(os.path.join(savepath, savefile), 'w')
        savefile['flux_predict'] = predict
        savefile['std_predict'] = std
        savefile['flux_true'] = true
        savefile['params'] = params
        savefile['zout'] = self.zout
        savefile.close()

    def get_predicted(self, params):
        """Helper function to get the predicted flux power spectrum and error, rebinned to match the desired kbins."""
        nparams = params
        if self.mf_slope:
            # tau_0_i[z] @dtau_0 / tau_0_i[z] @[dtau_0 = 0]
            # Divided by lowest redshift case
            tau0_fac = mflux.mean_flux_slope_to_factor(self.zout, params[0])
            nparams = params[1:] #Keep only t0 sampling parameter (of mean flux parameters)
        else: #Otherwise bug if choose mean_flux = 'c'
            tau0_fac = None
        # .predict should take [{list of parameters: t0; cosmo.; thermal},]
        # Here: emulating @ cosmo.; thermal; sampled t0 * [tau0_fac from above]
        predicted_nat, std_nat = self.gpemu.predict(np.array(nparams).reshape(1, -1), tau0_factors=tau0_fac)
        ndense = len(self.emulator.mf.dense_param_names)
        hindex = ndense + self.emulator.param_names["hub"]
        omegamh2_index = ndense + self.emulator.param_names["omegamh2"]
        assert 0.5 < nparams[hindex] < 1
        omega_m = nparams[omegamh2_index]/nparams[hindex]**2
        okf, predicted = flux_power.rebin_power_to_kms(kfkms=self.kf, kfmpc=self.gpemu.kf, flux_powers=predicted_nat[0], zbins=self.zout, omega_m=omega_m)
        _, std = flux_power.rebin_power_to_kms(kfkms=self.kf, kfmpc=self.gpemu.kf, flux_powers=std_nat[0], zbins=self.zout, omega_m=omega_m)
        return okf, predicted, std

    def get_BOSS_error(self, zbin):
        """Get the BOSS covariance matrix error."""
        #Redshifts
        sdssz = self.sdss.get_redshifts()
        #Fix maximum redshift bug
        sdssz = sdssz[(sdssz <= self.max_z)*(sdssz >= self.min_z)]
        #Important assertion
        np.testing.assert_allclose(sdssz, self.zout, atol=1.e-16)
        if zbin < 0:
            # Returns the covariance matrix in block format for all redshifts up to max_z (sorted low to high redshift)
            covar_bin = self.sdss.get_covar()[:sdssz.shape[0]*self.kf.shape[0], :sdssz.shape[0]*self.kf.shape[0]]
        else:
            covar_bin = self.sdss.get_covar(sdssz[zbin])
        return covar_bin

    def get_data_correction(self, okf, params, redshift):
        """Get the DLA and SiIII flux power corrections."""
        # First, get the DLA correction
        dla_corr = DLAcorr(okf, redshift, params[self.data_params['a_lls']:self.data_params['a_dla']+1])
        # Then the SiIII correction
        tau_eff = 0.0046*(1+redshift)**3.3 # model from Palanque-Delabrouille 2013, arXiv:1306.5896
        siIII_corr = SiIIIcorr(params[self.data_params['fSiIII']], tau_eff, okf)
        return dla_corr*siIII_corr

    def hubble_prior(self, params, source='none'):
        """Return a prior on little h (either Planck or SH0ES)"""
        if source == 'none': return 0
        hh = self.emulator.param_names['hub']
        if self.mf_slope:
            hh = hh + 2
        if source == 'shoes':
            shoes_mean, shoes_sigma = 0.7253, 0.0099 # SH0ES arxiv: 2112.04510
            return -((params[hh]-shoes_mean)/shoes_sigma)**2
        if source == 'planck':
            planck_mean, planck_sigma = 0.674, 0.005 # Planck arxiv: 1807.06209
            return -((params[hh]-planck_mean)/planck_sigma)**2
        else: return 0

    def omega_prior(self, params):
        """Return a prior on Omega_m h^2 (Planck 2018)"""
        # values from Planck: arxiv 1807.06209
        oo = self.emulator.param_names['omegamh2']
        if self.mf_slope:
            oo = oo + 2
        planck_mean, planck_sigma = 0.1424, 0.001
        return -((params[oo]-planck_mean)/planck_sigma)**2

    def bhfeedback_prior(self, params):
        """Return a prior on black hole feedback (prior away)"""
        # value range is [0.03, 0.07]
        bh = self.emulator.param_names['bhfeedback']
        if self.mf_slope:
            bh = bh + 2
        bh_mean, bh_sigma = 0.05, 0.01
        return -((params[bh]-bh_mean)/bh_sigma)**2

    def likelihood(self, params, include_emu=True, data_power=None, hprior='none', oprior=False, bhprior=False, use_meant=None, meant_fac=9.1):
        """A simple likelihood function for the Lyman-alpha forest.
        Assumes data is quadratic with a covariance matrix.
        The covariance for the emulator points is assumed to be
        completely correlated with each z bin, as the emulator
        parameters are estimated once per z bin."""
        # Default data to use is BOSS data
        if data_power is None:
            data_power = np.copy(self.BOSS_flux_power)
        # Set parameter limits as the hull of the original emulator.
        if np.any(params >= self.param_limits[:, 1]) or np.any(params <= self.param_limits[:, 0]):
            return -np.inf

        okf, predicted, std = self.get_predicted(params[:self.ndim-len(self.data_params)])
        nkf = int(np.size(self.kf))
        nz = np.shape(predicted)[0]
        assert nz == int(np.size(data_power)/nkf)
        # Likelihood using full covariance matrix
        chi2 = 0
        for bb in range(nz):
            idp = np.where(self.kf >= okf[bb][0])
            if len(self.data_params) != 0:
                # Get and apply the DLA and SiIII corrections to the prediction
                predicted[bb] = predicted[bb]*self.get_data_correction(okf[bb], params, self.zout[bb])
            diff_bin = predicted[bb] - data_power[bb][idp]
            diff_bin = diff_bin
            std_bin = std[bb]
            bindx = np.min(idp)
            covar_bin = self.get_BOSS_error(bb)[bindx:, bindx:]
            assert np.shape(np.outer(std_bin, std_bin)) == np.shape(covar_bin)
            if include_emu:
                if self.loo_errors:
                    icov_bin = self.icov_bin[bb]
                    cdet = self.cdet[bb]
                else:
                    # Assume completely correlated emulator errors within this bin
                    covar_emu = np.outer(std_bin, std_bin)
                    covar_bin += covar_emu
                    icov_bin = np.linalg.inv(covar_bin)
                    cdet = np.linalg.slogdet(covar_bin)[1]
            else:
                icov_bin = np.linalg.inv(covar_bin)
                cdet = np.linalg.slogdet(covar_bin)[1]
            dcd = - np.dot(diff_bin, np.dot(icov_bin, diff_bin),)/2.
            chi2 += dcd -0.5 * cdet
            assert 0 > chi2 > -2**31
            assert not np.isnan(chi2)
        if use_meant or (use_meant is None and self.use_meant):
            # omit mean flux and flux power data correction parameters
            indi = 0
            if self.mf_slope: indi = 2
            chi2 += self.meant_gpemu.likelihood(params[indi:self.ndim-len(self.data_params)], include_emu=include_emu, data_meanT=self.sim_meant)*meant_fac
        chi2 += self.hubble_prior(params, source=hprior)
        if oprior: chi2 += self.omega_prior(params)
        if bhprior: chi2 += self.bhfeedback_prior(params)
        return chi2

    def get_pnames(self):
        """Get a list of the parameter names"""
        pnames = self.emulator.print_pnames()
        # Add mean flux parameters
        if self.mf_slope:
            pnames = [('dtau0', r'd\tau_0'),]+pnames
        # Add DLA, SiIII correction parameters
        if len(self.data_params) != 0:
            pnames = pnames + self.dnames
        return pnames

    def make_cobaya_dict(self, *, data_power=None, burnin=1e4, nsamples=3e4, use_meant=None, meant_fac=9.1, pscale=80, emu_error=True, hprior='none', oprior=False, bhprior=False):
        """Return a dictionary that can be used to run Cobaya MCMC sampling."""
        # Parameter names
        pnames = self.get_pnames()
        # Get parameter ranges for use as a rough estimate of proposal pdf width
        prange = (self.param_limits[:, 1]-self.param_limits[:, 0])
        # Build the dictionary
        info = {}
        info["likelihood"] = {__name__+".CobayaLikelihoodClass": {"basedir": self.basedir, "HRbasedir": self.HRbasedir, "mean_flux": self.mean_flux, "max_z": self.max_z, "min_z": self.min_z, "emulator_class": self.emulator_class, "t0_training_value": self.t0_training_value, "optimise_GP": True, "emulator_json_file": self.emulator_json_file, "data_corr": self.data_corr, "traindir": self.traindir, "loo_errors": self.loo_errors, "hprior": hprior, "oprior": oprior, "bhprior": bhprior, "tau_thresh": self.tau_thresh, "sim_meant": self.sim_meant, "use_meant": use_meant, "meant_fac": meant_fac, "include_emu": emu_error, "data_power": data_power}}
        # Each of the parameters has a prior with limits and a proposal width (the proposal covariance matrix
        # is learned, so the value given needs to be small enough for the sampler to get started)
        info["params"] = {pnames[i][0]: {'prior': {'min': self.param_limits[i, 0], 'max': self.param_limits[i, 1]}, 'proposal': prange[i]/pscale, 'latex': pnames[i][1]} for i in range(self.ndim)}
        # Set up the mcmc sampler options (to do seed runs, add the option 'seed': integer between 0 and 2**32 - 1)
        info["sampler"] = {"mcmc": {"burn_in": burnin, "max_samples": nsamples, "Rminus1_stop": 0.01, "output_every": '60s', "learn_proposal": True, "learn_proposal_Rminus1_max": 20, "learn_proposal_Rminus1_max_early": 30}}
        return info

    def do_sampling(self, savefile=None, datadir=None, burnin=3e4, nsamples=3e5, pscale=80, include_emu_error=True, use_meant=None, meant_fac=9.1, hprior='none', oprior=False, bhprior=False):
        """Run MCMC using Cobaya. Cobaya supports MPI, with a separate chain for each process (for HPCC, 4-6 chains recommended).
        burnin and nsamples are per chain. If savefile is None, the chain will not be saved."""
        # if use_meant not specificed, default to setting from initialization
        if use_meant is None: use_meant = self.use_meant

        # If datadir is None, default is to use the flux power data from BOSS (dr14 or dr9)
        data_power = None
        self.sim_meant = None
        if datadir is not None:
            _, kfmpc, _ = self.emulator.get_flux_vectors(max_z=self.max_z, min_z=self.min_z, kfunits="mpc")
            # Load the data directory (i.e. use a simulation flux power as data)
            data_power = load_data(datadir, kfkms=self.kf, kfmpc=kfmpc, t0=self.t0_training_value, zout=self.zout, max_z=self.max_z, min_z=self.min_z, tau_thresh=self.tau_thresh)
            # get the appropriate simulation data for temperature as well
            if use_meant:
                self.sim_meant = t0_likelihood.load_data(datadir+'/emulator_meanT.hdf5', 0, max_z=3.8, min_z=2.2)

        # Construct the "info" dictionary used by Cobaya
        info = self.make_cobaya_dict(data_power=data_power, emu_error=include_emu_error, pscale=pscale, burnin=burnin, nsamples=nsamples, use_meant=use_meant, meant_fac=meant_fac, hprior=hprior, oprior=oprior, bhprior=bhprior)

        if savefile is not None:
            info["output"] = savefile

        # Set up MPI protections (as suggested in Cobaya documentation)
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        success = False
        try:
            # Run the sampler, Cobaya MCMC -- resume will only work if a savefile is given (so it can load previous chain)
            updated_info, sampler = cobaya_run(info, resume=True)
            success = True
        except LoggedError as err:
            pass
        success = all(comm.allgather(success))
        if not success and rank == 0:
            print("Sampling failed!")
        else:
            all_chains = comm.gather(sampler.products()["sample"], root=0)
            return sampler, all_chains

    def get_covar_det(self, params, include_emu):
        """Get the determinant of the covariance matrix.for certain parameters"""
        if np.any(params >= self.param_limits[:, 1]) or np.any(params <= self.param_limits[:, 0]):
            return -np.inf
        sdssz = self.sdss.get_redshifts()
        #Fix maximum redshift bug
        sdssz = sdssz[sdssz <= self.max_z]
        nz = sdssz.size
        if include_emu:
            okf, _, std = self.get_predicted(params)
        detc = 1
        for bb in range(nz):
            covar_bin = self.sdss.get_covar(sdssz[bb])
            if include_emu:
                idp = np.where(self.kf >= okf[bb][0])
                std_bin = std[bb]
                #Assume completely correlated emulator errors within this bin
                covar_emu = np.outer(std_bin, std_bin)
                covar_bin[idp, idp] += covar_emu
            _, det_bin = np.linalg.slogdet(covar_bin)
            #We have a block diagonal covariance
            detc *= det_bin
        return detc

    def refine_metric(self, params):
        """This evaluates the 'refinement metric':
           the extent to which the emulator error dominates the covariance.
           The idea is that when it is > 1, refinement is necessary"""
        detnoemu = self.get_covar_det(params, False)
        detemu = self.get_covar_det(params, True)
        return detemu/detnoemu

    def make_err_grid(self, i, j, samples=30000):
        """Make an error grid"""
        ndim = np.size(self.param_limits[:, 0])
        rr = lambda x: np.random.rand(ndim)*(self.param_limits[:, 1]-self.param_limits[:, 0]) + self.param_limits[:, 0]
        rsamples = np.array([rr(i) for i in range(samples)])
        randscores = [self.refine_metric(rr) for rr in rsamples]
        grid_x, grid_y = np.mgrid[0:1:200j, 0:1:200j]
        grid_x = grid_x * (self.param_limits[i, 1] - self.param_limits[i, 0]) + self.param_limits[i, 0]
        grid_y = grid_y * (self.param_limits[j, 1] - self.param_limits[j, 0]) + self.param_limits[j, 0]
        grid = scipy.interpolate.griddata(rsamples[:, (i, j)], randscores, (grid_x, grid_y), fill_value=0)
        return grid


class CobayaLikelihoodClass(Likelihood, LikelihoodClass):
    """Class inheriting Cobaya functionality. Strictly for use with Cobaya sampling."""
    # Cobaya automatically recognizes (and sets as default) inputs with the following names
    basedir: str
    HRbasedir: str = None
    mean_flux: str = 's'
    max_z: float = 4.6
    min_z: float = 2.2
    emulator_class: str = "standard"
    t0_training_value: float = 1.
    optimise_GP: bool = True
    emulator_json_file: str = 'emulator_params.json'
    data_corr: bool = True
    tau_thresh: int = None
    use_meant: bool = False
    sim_meant: float = None
    meant_fac: float = 9.1
    traindir: str = None
    data_power: float = None
    include_emu: bool = True
    loo_errors: bool = False
    hprior: str = 'none'
    oprior: bool = False
    bhprior: bool = False
    # Required for Cobaya to correctly parse which parameters are for input
    input_params_prefix: str = ""

    def initialize(self):
        """Initialization of Cobaya likelihood using LikelihoodClass init.
        Gets the emulator by loading the flux power spectra from the simulations."""
        LikelihoodClass.__init__(self, self.basedir, HRbasedir=self.HRbasedir, mean_flux=self.mean_flux, max_z=self.max_z, min_z=self.min_z, emulator_class=self.emulator_class, t0_training_value=self.t0_training_value, optimise_GP=self.optimise_GP, emulator_json_file=self.emulator_json_file, data_corr=self.data_corr, tau_thresh=self.tau_thresh, use_meant=self.use_meant, traindir=self.traindir, loo_errors=self.loo_errors)

    def logp(self, **params_values):
        """Cobaya-compatible call to the base class likelihood function.
        Must be called logp."""
        # self.input_params is specially recognized by Cobaya (will be the "params" section
        # of the Cobaya dictionary passed to it)
        params = np.array([params_values[p] for p in self.input_params])
        return self.likelihood(params, include_emu=self.include_emu, data_power=self.data_power, use_meant=self.use_meant, meant_fac=self.meant_fac, hprior=self.hprior, oprior=self.oprior, bhprior=self.bhprior)
