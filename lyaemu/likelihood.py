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
    for i in range(4):
        factor += alpha[i] * ((1+z)/(1+z_0))**-3.55 * (a_z[i]*np.exp(b_z[i]*kf) - 1)**-2
    return factor

def load_data(datadir, *, kf, max_z=4.6, min_z=2.2, t0=1., tau_thresh=None):
    """Load and initialise a "fake data" flux power spectrum"""
    #Load the data directory
    myspec = flux_power.MySpectra(max_z=max_z, min_z=min_z)
    pps = myspec.get_snapshot_list(datadir)
    #self.data_fluxpower is used in likelihood.
    data_fluxpower = pps.get_power(kf=kf, mean_fluxes=np.exp(-t0*mflux.obs_mean_tau(myspec.zout, amp=0)), tau_thresh=tau_thresh)
    assert np.size(data_fluxpower) % np.size(kf) == 0
    return data_fluxpower

class LikelihoodClass:
    """Class to contain likelihood computations."""
    def __init__(self, basedir, mean_flux='s', max_z=4.6, min_z=2.2, emulator_class="standard", t0_training_value=1., optimise_GP=True, emulator_json_file='emulator_params.json', data_corr=True, tau_thresh=None):
        """Initialise the emulator by loading the flux power spectra from the simulations."""
        # Needed for Cobaya dictionary construction
        self.basedir = basedir
        self.mean_flux = mean_flux
        self.emulator_class = emulator_class
        self.emulator_json_file = emulator_json_file
        self.data_corr = data_corr
        self.tau_thresh = tau_thresh

        self.max_z = max_z
        self.min_z = min_z
        self.t0_training_value = t0_training_value
        myspec = flux_power.MySpectra(max_z=max_z, min_z=min_z)
        self.zout = myspec.zout
        # Default data is flux power from Chabanier 2019 (BOSS DR14),
        # pass datafile='dr9' to use data from Palanque-Delabrouille 2013
        self.sdss = lyman_data.BOSSData()
        self.kf = self.sdss.get_kf()
        # Load BOSS data vector
        self.BOSS_flux_power = self.sdss.pf.reshape(-1, self.kf.shape[0])[:self.zout.shape[0]][::-1]
        # Units: km / s; Size: n_z * n_k

        self.mf_slope = False
        # Param limits on t0
        t0_factor = np.array([0.75, 1.25])
        if mean_flux == 'c':
            mf = mflux.ConstMeanFlux(value=t0_training_value)
        # Redshift bins are independent -- for redshift-dependent mean flux models
        # we convert the input parameters to a list of mean flux scalings in each redshift bin.
        # This parametrises the mean flux as an amplitude and slope.
        elif mean_flux == 's':
            # Add a slope to the parameter limits
            t0_slope = np.array([-0.25, 0.25])
            self.mf_slope = True
            slopehigh = np.max(mflux.mean_flux_slope_to_factor(self.zout[::-1], 0.25))
            slopelow = np.min(mflux.mean_flux_slope_to_factor(self.zout[::-1], -0.25))
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
            # Shrink param limits t0 so that even with
            # a slope they are within the emulator range
            self.param_limits[1, :] = t0_factor

        # Set up SiIII and DLA corrections, if requested
        self.data_params = {}
        self.dla_data_corr = data_corr
        if data_corr:
            self.ndim = np.shape(self.param_limits)[0]
            # Create some useful objects for implementing the DLA and SiIII corrections
            self.dnames = [('a_lls', r'\alpha_{lls}'), ('a_sub', r'\alpha_{sub}'), ('a_sdla', r'\alpha_{sdla}'), ('a_ldla', r'\alpha_{ldla}'), ('fSiIII', 'fSiIII')]
            self.data_params = {self.dnames[i][0]:np.arange(self.ndim, self.ndim+np.shape(self.dnames)[0])[i] for i in range(np.shape(self.dnames)[0])}
            # Limits for the data correction parameters
            alpha_limits = np.repeat(np.array([[-1., 1.]]), 4, axis=0)
            fSiIII_limits = np.array([-0.03, 0.03])
            self.param_limits = np.vstack([self.param_limits, alpha_limits, fSiIII_limits])
        self.ndim = np.shape(self.param_limits)[0]
        assert np.shape(self.param_limits)[1] == 2

        # Generate emulator
        if optimise_GP:
            print('Beginning to generate emulator at', str(datetime.now()))
            self.gpemu = self.emulator.get_emulator(max_z=max_z, min_z=min_z)
            print('Finished generating emulator at', str(datetime.now()))


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
        sdssz = sdssz[sdssz <= self.max_z]
        #Important assertion
        np.testing.assert_allclose(sdssz, self.zout, atol=1.e-16)
        #print('SDSS redshifts are', sdssz)
        if zbin < 0:
            # Returns the covariance matrix in block format for all redshifts up to max_z (sorted low to high redshift)
            covar_bin = self.sdss.get_covar()[:sdssz.shape[0]*self.kf.shape[0], :sdssz.shape[0]*self.kf.shape[0]]
        else:
            covar_bin = self.sdss.get_covar(sdssz[zbin])
        return covar_bin

    def get_data_correction(self, okf, params, redshift):
        # First, apply the DLA correction to the prediction
        dla_corr = DLAcorr(okf, redshift, params[self.data_params['a_lls']:self.data_params['a_ldla']+1])
        # Then apply the SiIII correction
        tau_eff = 0.0046*(1+redshift)**3.3 # model from Palanque-Delabrouille 2013, arXiv:1306.5896
        siIII_corr = SiIIIcorr(params[self.data_params['fSiIII']], tau_eff, okf)
        return dla_corr*siIII_corr

    def data_correction_prior(self, params):
        sigma_dla = 0.2 # somewhat arbitrary values for the prior widths
        dla = -np.sum((params[self.data_params['a_lls']:self.data_params['a_ldla']+1]/sigma_dla)**2)
        sigma_siIII = 1e-2
        siIII = -(params[self.data_params['fSiIII']]/sigma_siIII)**2
        return dla + siIII

    def likelihood(self, params, include_emu=True, data_power=None):
        """A simple likelihood function for the Lyman-alpha forest.
        Assumes data is quadratic with a covariance matrix.
        The covariance for the emulator points is assumed to be
        completely correlated with each z bin, as the emulator
        parameters are estimated once per z bin."""
        # Default data to use is BOSS data
        if data_power is None:
            data_power = np.copy(self.BOSS_flux_power.flatten())
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
                # Add a prior for the DLA and SiIII correction parameters (zero-centered normal)
                chi2 += self.data_correction_prior(params)
                # Get and apply the DLA and SiIII corrections to the prediction
                predicted[bb] = predicted[bb]*self.get_data_correction(okf[bb], params, self.zout[bb])
            diff_bin = predicted[bb] - data_power[nkf*bb:nkf*(bb+1)][idp]
            std_bin = std[bb]
            bindx = np.min(idp)
            covar_bin = self.get_BOSS_error(bb)[bindx:, bindx:]
            assert np.shape(np.outer(std_bin, std_bin)) == np.shape(covar_bin)
            if include_emu:
                # Assume completely correlated emulator errors within this bin
                covar_emu = np.outer(std_bin, std_bin)
                covar_bin += covar_emu
            icov_bin = np.linalg.inv(covar_bin)
            (_, cdet) = np.linalg.slogdet(covar_bin)
            dcd = - np.dot(diff_bin, np.dot(icov_bin, diff_bin),)/2.
            chi2 += dcd -0.5 * cdet
            assert 0 > chi2 > -2**31
            assert not np.isnan(chi2)
        return chi2

    def make_cobaya_dict(self, *, data_power, burnin, nsamples, pscale=50, emu_error=True):
        pnames = self.emulator.print_pnames()
        #Set up mean flux
        if self.mf_slope:
            pnames = [('dtau0', r'd\tau_0'),]+pnames
        # Add DLA, SiIII correction parameters
        if len(self.data_params) != 0:
            pnames = pnames + self.dnames
        # get parameter ranges for rough estimate of proposal pdf width for mcmc sampler
        prange = (self.param_limits[:, 1]-self.param_limits[:, 0])
        info = {}
        info["likelihood"] = {__name__+".CobayaLikelihoodClass": {"basedir": self.basedir, "mean_flux": self.mean_flux, "max_z": self.max_z, "min_z": self.min_z,
                                                                    "emulator_class": self.emulator_class, "t0_training_value": self.t0_training_value,
                                                                    "optimise_GP": True, "emulator_json_file": self.emulator_json_file, "data_corr": self.data_corr,
                                                                    "tau_thresh": self.tau_thresh, "include_emu": emu_error, "data_power": data_power}}
        # each of the parameters (name should match those in input_params) has prior with limits and proposal width
        # (the proposal covariance matrix is learned, so the value given needs only be small enough to accept steps)
        info["params"] = {pnames[i][0]: {'prior': {'min': self.param_limits[i, 0], 'max': self.param_limits[i, 1]}, 'proposal': prange[i]/pscale,
                                         'latex': pnames[i][1]} for i in range(self.ndim)}
        # set up the mcmc sampler options (to do seed runs, add below the option seed: integer between 0 and 2**32 - 1)
        # default for computing Gelman-Rubin is to split chain into 4; to change, add option Rminus1_single_split: integer
        info["sampler"] = {"mcmc": {"burn_in": burnin, "max_samples": nsamples, "Rminus1_stop": 0.01, "output_every": '60s', "learn_proposal": True,
                                    "learn_proposal_Rminus1_max": 30, "learn_proposal_Rminus1_max_early": 30}}
        return info

    def do_acceptance_check(self, info, steps=100):
        print("-----------------------------------------------------")
        print("Test run to check acceptance rate")
        info_test = info.copy()
        # don't do a burn-in, limit the sample to steps, and increase the max_tries to ensure an acceptance rate
        info_test.update({"sampler": {"mcmc": {"burn_in": 0, "max_samples": steps, "max_tries": '1000d'}}})
        updated_info, sampler = cobaya_run(info_test)
        print('Acceptance rate:', sampler.get_acceptance_rate())
        print("----------------------------------------------------- \n")
        assert sampler.get_acceptance_rate() > 0.01, "Acceptance rate very low. Consider decreasing the proposal width by increasing the pscale parameter"

    def do_sampling(self, savefile=None, datadir=None, burnin=3000, nsamples=50000, pscale=50, include_emu_error=True, test_accept=True):
        """Initialise and run MCMC using Cobaya."""
        # If datadir is None, default is to use the flux power data from BOSS (dr14 or dr9)
        data_power = None
        if datadir is not None:
            # Load the data directory (i.e. use a simulation flux power as data)
            data_power = load_data(datadir, kf=self.kf, t0=self.t0_training_value, max_z=self.max_z, min_z=self.min_z, tau_thresh=self.tau_thresh)

        # Construct the "info" dictionary used by Cobaya
        info = self.make_cobaya_dict(data_power=data_power, emu_error=include_emu_error, pscale=pscale, burnin=burnin, nsamples=nsamples)

        # Test run a fraction of the full chain to check acceptance rate before running full chain
        if test_accept is True:
            self.do_acceptance_check(info, steps=100)

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
        # Now recombine chains into a single combined file and save
        all_chains = comm.gather(sampler.products()["sample"], root=0)
        if rank == 0:
            self.combine_chains(all_chains)

        return sampler

    def combine_chains(self, all_chains):
        full_chain = all_chains[0]
        for chain in all_chains[1:]:
            full_chain.append(chain)
        # Set up the save the same way as Cobaya
        n_float = 8
        width_col = lambda col: max(7 + n_float, len(col))
        numpy_fmts = ["%{}.{}".format(width_col(col), n_float) + "g" for col in full_chain.data.columns]
        header_formatter = [eval('lambda s, w=width_col(col): ''("{:>" + "{}".format(w) + "s}").format(s)', {'width_col': width_col, 'col': col}) for col in full_chain.data.columns]
        with open(full_chain.root_file_name[2:]+".combined.txt", "w", encoding="utf-8") as out:
            out.write("#" + " ".join(f(col) for f, col in zip(header_formatter, full_chain.data.columns))[1:] + "\n")
        with open(full_chain.root_file_name[2:]+".combined.txt", "a", encoding="utf-8") as out:
            np.savetxt(out, full_chain.data.to_numpy(), fmt=numpy_fmts)

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
    """Class to contain likelihood computations."""

    basedir: str
    mean_flux: str = 's'
    max_z: float = 4.6
    min_z: float = 2.2
    emulator_class: str = "standard"
    t0_training_value: float = 1.
    optimise_GP: bool = True
    emulator_json_file: str = 'emulator_params.json'
    data_corr: bool = True
    tau_thresh: int = None
    data_power: float = None
    include_emu: bool = True
    input_params_prefix: str = ""

    def initialize(self):
        """Initialise the emulator by loading the flux power spectra from the simulations."""
        LikelihoodClass.__init__(self, self.basedir, mean_flux=self.mean_flux, max_z=self.max_z, min_z=self.min_z,
                         emulator_class=self.emulator_class, t0_training_value=self.t0_training_value,
                         optimise_GP=self.optimise_GP, emulator_json_file=self.emulator_json_file,
                         data_corr=self.data_corr, tau_thresh=self.tau_thresh)

    def logp(self, **params_values):
        """Cobaya-compatible call to the base class likelihood function."""
        params = np.array([params_values[p] for p in self.input_params])
        return self.likelihood(params, include_emu=self.include_emu, data_power=self.data_power)
