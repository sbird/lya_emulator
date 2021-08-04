"""Separate file for doing Bayesian optimisation on a likelihood.
The main routine is BayesianOpt.find_new_trials

To perform Bayesian optimisation, do:

bayes.BayesianOpt(emulatordir, datadir)
new_sims = bayes.find_new_trials(1-3)
bayes.gen_new_simulations(new_sims)

Then run the simulations

To regenerate the emulator once the new simulations have run you can do:
        emulator.reconstruct()
        emulator.dump()
"""
import math
import mpmath as mmh
import numpy as np
import scipy.optimize as spo
from .latin_hypercube import map_to_unit_cube, map_from_unit_cube
from . import likelihood
from . import gpemulator

class BayesianOpt:
    """Class for doing Bayesian optimisation with the likelihood."""
    def __init__(self, emudir, datadir):
        self.like = likelihood.LikelihoodClass(emudir, mean_flux='s', data_corr=False)
        self.param_limits = self.like.param_limits
        #This will be replaced with real data (set equal to None to use default BOSS data)
        self.data_fluxpower = likelihood.load_data(datadir, kf=self.like.kf, t0=self.like.t0_training_value)
        #Parameters to calculate the exploration weight. In practice exploration is usually subdominant so these are not very important.
        self.delta = 0.5
        self.nu = 1

    def find_new_trials(self, nsamples, iteration_number=1, marginalise_mean_flux=True):
        """Main driver of Bayesian optimisation.
        Optimises the acquisition function multiple times to find new simulations to run. This is the batch mode of Bayesian optimisation."""
        rng = np.random.default_rng()
        #Pick a starting point for the optimisation in the middle of the parameter range
        starting_params = (self.param_limits[2:,0] + self.param_limits[2:,1])/2.
        new_points = np.zeros((nsamples,)+np.shape(starting_params))
        for i in range(nsamples):
            offset = rng.uniform(0.2, 0.8, size=np.shape(starting_params))
            starting_params = offset * self.param_limits[2:,0] + (1-offset)*self.param_limits[2:,1]
            #Generate a new optimum of the Bayesian optimisation function
            new_points[i,:] = self.optimise_acquisition_function(starting_params, marginalise_mean_flux=marginalise_mean_flux,
                                                           iteration_number = iteration_number+i)
            #Build a new GP emulator adding the *prediction* of this new optimum from the old GP emulator.
            #This will shrink the error bars and thus change the next Bayesian point.
            new_params_with_mf, new_flux_with_mf = self.get_new_predicted_power(new_points[i,:], self.like.gpemu, self.like.emulator.mf)
            newgpemu = self.rebuild_emulator_extra(new_params_with_mf, new_flux_with_mf)
            self.like.gpemu = newgpemu
        return new_points

    def rebuild_emulator_extra(self, new_params, new_flux):
        """Build a new emulator adding extra flux vectors, assumed to include mean flux samples. Used for batch mode Bayesian optimisation."""
        aparams, kf, flux_vectors = self.like.gpemu.get_training_data()
        params = np.concatenate([new_params, aparams])
        flux = np.concatenate([new_flux, flux_vectors])
        par_lim = self.like.emulator.get_param_limits(include_dense=True)
        gp = gpemulator.MultiBinGP(params=params, kf=kf, powers = flux, param_limits = par_lim)
        return gp

    def get_new_predicted_power(self, new_point, gpemu, mf):
        """This routine generates a new sample - including with new mean flux values - from the emulator.
        This is needed for the batch mode of Bayesian optimisation: the new predictions of the emulator will
        be added as training data."""
        #Make sure we have the right number of params
        assert np.shape(new_point)[0]+2 == np.shape(self.param_limits)[0]
        #Note this gets tau_0 as a linear scale factor from the observed power law
        dpvals = mf.get_params()
        if dpvals is not None:
            aparams = np.array([np.concatenate([dp,new_point]) for dp in dpvals])
        # .predict should take [{list of parameters: t0; cosmo.; thermal},]
        flux_vectors = np.array([gpemu.predict(np.array(aa).reshape(1, -1), tau0_factors=None)[0][0] for aa in aparams])
        return aparams, flux_vectors

    def gen_new_simulations(self, new_samples):
        """Generate simulations for the newly found points."""
        if len(np.shape(new_samples)) == 1:
            new_samples = [new_samples,]
        [self.like.emulator.do_ic_generation(ev) for ev in new_samples]

    def loglike_marginalised_mean_flux(self, params, include_emu=True, integration_bounds='default', integration_options='gauss-legendre', verbose=False, integration_method='Quadrature'):
        """Evaluate (Gaussian) likelihood marginalised over mean flux parameter axes: (dtau0, tau0)"""
        #assert len(marginalised_axes) == 2
        #marginalised_axes=(0, 1)
        if integration_bounds == 'default':
            integration_bounds = [list(self.param_limits[0]), list(self.param_limits[1])]
        likelihood_function = lambda dtau0, tau0: mmh.exp(self.like.likelihood(np.concatenate(([dtau0, tau0], params)), include_emu=include_emu, data_power=self.data_fluxpower))
        if integration_method == 'Quadrature':
            integration_output = mmh.quad(likelihood_function, integration_bounds[0], integration_bounds[1], method=integration_options, error=True, verbose=verbose)
        elif integration_method == 'Monte-Carlo':
            integration_output = (self._do_Monte_Carlo_marginalisation(likelihood_function, self.param_limits, n_samples=integration_options),)
        return float(mmh.log(integration_output[0]))

    def _do_Monte_Carlo_marginalisation(self, function, param_limits, n_samples=6000):
        """Marginalise likelihood by Monte-Carlo integration"""
        random_samples = param_limits[:2, 0, np.newaxis] + (param_limits[:2, 1, np.newaxis] - param_limits[:2, 0, np.newaxis]) * np.random.rand(2, n_samples)
        function_sum = 0.
        for i in range(n_samples):
            print('Likelihood function evaluation number =', i + 1)
            function_sum += function(random_samples[0, i], random_samples[1, i])
        volume_factor = (param_limits[0, 1] - param_limits[0, 0]) * (param_limits[1, 1] - param_limits[1, 0])
        return volume_factor * function_sum / n_samples

    def get_GP_UCB_exploration_term(self, params, iteration_number=1, marginalise_mean_flux = True):
        """Evaluate the exploration term of the GP-UCB acquisition function"""
        assert iteration_number >= 1.
        assert 0. < self.delta < 1.
        param_limits_mf = self.param_limits[:2, :]
        #if self._inverse_BOSS_covariance_full is None:
            #self._inverse_BOSS_covariance_full = invert_block_diagonal_covariance(self.get_BOSS_error(-1), self.zout.shape[0])
        exploration_weight = math.sqrt(self.nu * 2. * math.log((iteration_number**((np.shape(params)[0] / 2.) + 2.)) * (math.pi**2) / 3. / self.delta))
        #Exploration term: least accurate part of the emulator
        if not marginalise_mean_flux:
            okf, _, std = self.like.get_predicted(params)
        else:
            #Compute the error averaged over the mean flux
            dtau0 = np.mean([param_limits_mf[0, 0], param_limits_mf[0, 1]])
            tau0 = np.mean([param_limits_mf[1, 0], param_limits_mf[1, 1]])
            okf, _, std = self.like.get_predicted(np.concatenate([[dtau0, tau0], params]))
            n_samples = 0
            #We don't really need to do this, because the interpolation error should always be dominated
            #by the position in simulation parameter space: we have always used multiple mean flux points.
            #So just use the error at the average mean flux value
            if n_samples > 0:
                for dtau0 in np.linspace(param_limits_mf[0, 0], param_limits_mf[0, 1], num=n_samples):
                    for tau0 in np.linspace(param_limits_mf[1, 0], param_limits_mf[1, 1], num=n_samples):
                        _, _, std_loc = self.like.get_predicted(np.concatenate([[dtau0, tau0], params]))
                        for ii, ss in enumerate(std_loc):
                            std[ii] += ss
                for ss in std:
                    ss/=(n_samples**2+1)
        #Do the summation of sigma_emu^T \Sigma^{-1}_{BOSS} sigma_emu (ie, emulator error convolved with data covariance)
        posterior_estimated_error = 0
        nz = np.shape(std)[0]
        #Likelihood using full covariance matrix
        for bb in range(nz):
            idp = np.where(self.like.kf >= okf[bb][0])
            std_bin = std[bb]
            bindx = np.min(idp)
            covar_bin = self.like.get_BOSS_error(bb)[bindx:, bindx:]
            assert np.shape(np.outer(std_bin, std_bin)) == np.shape(covar_bin)
            icov_bin = np.linalg.inv(covar_bin)
            dcd = - np.dot(std_bin, np.dot(icov_bin, std_bin),)/2.
            posterior_estimated_error += dcd
            assert 0 > posterior_estimated_error > -2**31
            assert not np.isnan(posterior_estimated_error)
        return exploration_weight * posterior_estimated_error

    def acquisition_function_GP_UCB(self, params, iteration_number=1, exploitation_weight=1., marginalise_mean_flux = True):
        """Evaluate the GP-UCB at given parameter vector. This is an acquisition function for determining where to run
        new training simulations"""
        #Exploration term: least accurate part of the emulator
        exploration = self.get_GP_UCB_exploration_term(params, marginalise_mean_flux=marginalise_mean_flux, iteration_number=iteration_number)
        #Exploitation term: how good is the likelihood at this point
        exploitation = 0
        if exploitation_weight is not None:
            if marginalise_mean_flux:
                loglike = self.loglike_marginalised_mean_flux(params)
            else:
                loglike = self.like.likelihood(params, data_power = self.data_fluxpower)
            exploitation = loglike * exploitation_weight
        print("acquis: %g explor: %g exploit:%g params:" % (exploitation+exploration,exploration,exploitation), params)
        return exploration + exploitation

    def optimise_acquisition_function(self, starting_params, opt_bound = 0.05, iteration_number=1, exploitation_weight=1., marginalise_mean_flux=True):
        """Find parameter vector (marginalised over mean flux parameters) at maximum of (GP-UCB) acquisition function"""
        #We marginalise the mean flux parameters so they should not be mapped
        if marginalise_mean_flux:
            param_limits_no_mf = self.param_limits[2:,:]
        mapped = lambda parameter_vector: map_from_unit_cube(parameter_vector, param_limits_no_mf)
        optimisation_function = lambda parameter_vector: -1.*self.acquisition_function_GP_UCB(mapped(parameter_vector),
                                                                iteration_number=iteration_number, exploitation_weight=exploitation_weight,
                                                                marginalise_mean_flux=marginalise_mean_flux)
        #Shrink optimisation bounds to interior 95% of emulator to avoid edge effects
        optimisation_bounds = [(opt_bound, 1. - opt_bound) for i in range(starting_params.shape[0])]
        min_result = spo.minimize(optimisation_function, map_to_unit_cube(starting_params, param_limits_no_mf), bounds=optimisation_bounds)
        if not min_result.success:
            print(min_result)
            raise ValueError(min_result.message)
        return map_from_unit_cube(min_result.x, param_limits_no_mf)
