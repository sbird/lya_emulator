"""Module for computing the reduced likelihood function for the Lyman-alpha forest.
This is a 2D Gaussian likelihood, which is an approximation to the full likelihood used in likelihood.py.
Be warned that marginalisation effects means that the contours are not fully Gaussian, so this will not be entirely accurate.

Input parameters are the amplitude and slope of the linear matter power spectrum at z_P=3 and k_P = 0.009 s/km = 1 Mpc^-1.

In reality this Lyman alpha forest dataset is sensitive to redshifts z=4.6 - 2.6 and scales k = 10^-3 - 2x10^-2 s/km.
In Mpc units these translate to k = 0.1 - 2.6 Mpc^-1 (redshift dependent! 0.13 - 2.6 Mpc^-1 at z=4.6 and (0.1 - 2.1) Mpc^-1 at z=2.6."""
from cobaya.likelihood import Likelihood
from cobaya.run import run as cobaya_run
from cobaya.log import LoggedError
from mpi4py import MPI

class ReducedLymanAlpha(Likelihood):
    """Class inheriting Cobaya functionality, designed to sample the reduced Lyman alpha likelihood at z=3, k = 1 / Mpc via a (inaccurate) 2D Gaussian fit.
    The chain fit to is eBOSS FPS + T0 z >= 2.6.

    Input parameters are the amplitude and slope of the linear matter power spectrum at z_P=3 and k_P = 0.009 s/km = 1 Mpc^-1.

    Sensitive to redshifts z=4.6 - 2.6 and scales k = 10^-3 - 2x10^-2 s/km, or k = 0.1 - 2.6 Mpc^-1."""
    deltal: float
    sigmadeltal: float
    neff: float
    sigmaneff: float
    correlation: float
    chabanier: bool
    # Required for Cobaya to correctly parse which parameters are for input
    input_params_prefix: str = ""

    def initialize(self):
        """Initialization of Cobaya likelihood using LikelihoodClass init. Sets parameter values using Table 3 of 2306.05471."""
        self.deltal = 0.267
        #Avg of upper and lower limit...
        self.sigmadeltal = 0.02
        self.neff = -2.288
        self.sigmaneff = 0.02
        self.correlation = 0.4
        #Results from the fit to Chabanier (2303.00746, Table 1)
        if self.chabanier:
            self.deltal = 0.31
            self.sigmadeltal = 0.02
            self.neff = -2.34
            self.sigmaneff = 0.006
            self.correlation = 0.512

    def logp(self, **params_values):
        """Cobaya-compatible call to the base class likelihood function.
        Must be called logp."""
        deltal = params_values["deltal"]
        neff = params_values["neff"]
        deltadl = (deltal - self.deltal)/self.sigmadeltal
        deltaneff = (neff - self.neff)/self.sigmaneff
        #2D gaussian log likelihood
        logl = -1 * (deltadl**2 - 2 * self.correlation * deltadl * deltaneff + deltaneff**2) / (2 * (1 - self.correlation**2))
        return logl

    def do_sampling(self, savefile=None, burnin=3e2, nsamples=3e3, pscale=80, chabanier = False):
        """Run MCMC using Cobaya. Cobaya supports MPI, with a separate chain for each process (for HPCC, 4-6 chains recommended).
        burnin and nsamples are per chain. If savefile is None, the chain will not be saved."""
        # Construct the "info" dictionary used by Cobaya
        info = {}
        info["likelihood"] = {__name__+".ReducedLymanAlpha": {"chabanier": chabanier}}
        # Each of the parameters has a prior with limits and a proposal width (the proposal covariance matrix
        # is learned, so the value given needs to be small enough for the sampler to get started)
        info["params"] = {"deltal": {'prior': {'min': 0.2, 'max': 0.7}, 'proposal': (0.7-0.2)/pscale, 'latex': r"$\Delta^2_L$"},
                          "neff": {'prior': {'min': -3.0, 'max': -2.0}, 'proposal': 1/pscale, 'latex': r"$n_\mathrm{eff}$"}
                          }
        # Set up the mcmc sampler options (to do seed runs, add the option 'seed': integer between 0 and 2**32 - 1)
        info["sampler"] = {"mcmc": {"burn_in": burnin, "max_samples": nsamples, "Rminus1_stop": 0.01, "output_every": '60s', "learn_proposal": True, "learn_proposal_Rminus1_max": 20, "learn_proposal_Rminus1_max_early": 30}}
        if savefile is not None:
            info["output"] = savefile

        # Set up MPI protections (as suggested in Cobaya documentation)
        comm = MPI.COMM_WORLD
        success = False
        try:
            # Run the sampler, Cobaya MCMC -- resume will only work if a savefile is given (so it can load previous chain)
            _, sampler = cobaya_run(info, resume=True)
            success = True
        except LoggedError as err:
            print(err)
        success = all(comm.allgather(success))
        if not success:
            raise LoggedError("Sampling failed!")
        all_chains = comm.gather(sampler.products()["sample"], root=0)
        return sampler, all_chains
