"""Separate file for doing Bayesian optimisation on a likelihood."""
import math
import mpmath as mmh
import numpy as np
import scipy.optimize as spo
from .latin_hypercube import map_to_unit_cube, map_from_unit_cube
from . import likelihood

def invert_block_diagonal_covariance(full_covariance_matrix, n_blocks):
    """Efficiently invert block diagonal covariance matrix"""
    inverse_covariance_matrix = np.zeros_like(full_covariance_matrix)
    nz = n_blocks
    nk = int(full_covariance_matrix.shape[0] / nz)
    for i, z in zip(range(nz), reversed(range(nz))): #Loop over blocks by redshift
        start_index = nk * z
        end_index = nk * (z + 1)
        inverse_covariance_block = np.linalg.inv(full_covariance_matrix[start_index:end_index, start_index:end_index])
        # Reverse the order of the full matrix so it runs from high to low redshift
        start_index = nk * i
        end_index = nk * (i + 1)
        inverse_covariance_matrix[start_index:end_index, start_index:end_index] = inverse_covariance_block
    return inverse_covariance_matrix

class BayesianOpt:
    """Class for doing Bayesian optimisation with the likelihood."""
    def __init__(self, emudir, datadir):
        self.like = likelihood.LikelihoodClass(emudir, mean_flux='s', data_corr=False)
        self.data_fluxpower = likelihood.load_data(datadir, kf=self.like.kf, t0=self.like.t0_training_value)
        self.optimise_acquisition_function(np.array([0.875, 2.58e-9, 4.24, 3.17, 1.6, 0.748, 0.146, 8.47, 0.04]))

    def loglike_marginalised_mean_flux(self, params, include_emu=True, integration_bounds='default', integration_options='gauss-legendre', verbose=False, integration_method='Quadrature'):
        """Evaluate (Gaussian) likelihood marginalised over mean flux parameter axes: (dtau0, tau0)"""
        #assert len(marginalised_axes) == 2
        #marginalised_axes=(0, 1)
        if integration_bounds == 'default':
            integration_bounds = [list(self.like.param_limits[0]), list(self.like.param_limits[1])]
        likelihood_function = lambda dtau0, tau0: mmh.exp(self.like.likelihood(np.concatenate(([dtau0, tau0], params)), include_emu=include_emu, data_power=self.data_fluxpower))
        if integration_method == 'Quadrature':
            integration_output = mmh.quad(likelihood_function, integration_bounds[0], integration_bounds[1], method=integration_options, error=True, verbose=verbose)
        elif integration_method == 'Monte-Carlo':
            integration_output = (self._do_Monte_Carlo_marginalisation(likelihood_function, self.like.param_limits, n_samples=integration_options),)
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

    def get_GP_UCB_exploration_term(self, params, iteration_number=1, delta=0.5, nu=1.,
                                    marginalise_mean_flux = True, use_updated_training_set=False):
        """Evaluate the exploration term of the GP-UCB acquisition function"""
        assert iteration_number >= 1.
        assert 0. < delta < 1.
        param_limits_mf = self.like.param_limits[:2, :]
        #if self._inverse_BOSS_covariance_full is None:
            #self._inverse_BOSS_covariance_full = invert_block_diagonal_covariance(self.get_BOSS_error(-1), self.zout.shape[0])
        exploration_weight = math.sqrt(nu * 2. * math.log((iteration_number**((np.shape(params)[0] / 2.) + 2.)) * (math.pi**2) / 3. / delta))
        #Exploration term: least accurate part of the emulator
        if not marginalise_mean_flux:
            okf, _, std = self.like.get_predicted(params, use_updated_training_set=use_updated_training_set)
        else:
            #Compute the error averaged over the mean flux
            dtau0 = np.mean([param_limits_mf[0, 0], param_limits_mf[0, 1]])
            tau0 = np.mean([param_limits_mf[1, 0], param_limits_mf[1, 1]])
            okf, _, std = self.like.get_predicted(np.concatenate([[dtau0, tau0], params]), use_updated_training_set=use_updated_training_set)
            n_samples = 0
            #We don't really need to do this, because the interpolation error should always be dominated
            #by the position in simulation parameter space: we have always used multiple mean flux points.
            #So just use the error at the average mean flux value
            if n_samples > 0:
                for dtau0 in np.linspace(param_limits_mf[0, 0], param_limits_mf[0, 1], num=n_samples):
                    for tau0 in np.linspace(param_limits_mf[1, 0], param_limits_mf[1, 1], num=n_samples):
                        _, _, std_loc = self.like.get_predicted(np.concatenate([[dtau0, tau0], params]), use_updated_training_set=use_updated_training_set)
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

    def acquisition_function_GP_UCB(self, params, iteration_number=1, delta=0.5, nu=1., exploitation_weight=1., marginalise_mean_flux = True, use_updated_training_set=False):
        """Evaluate the GP-UCB at given parameter vector. This is an acquisition function for determining where to run
        new training simulations"""
        #Exploration term: least accurate part of the emulator
        exploration = self.get_GP_UCB_exploration_term(params, marginalise_mean_flux=marginalise_mean_flux, iteration_number=iteration_number, delta=delta, nu=nu, use_updated_training_set=use_updated_training_set)
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

    def optimise_acquisition_function(self, starting_params, optimisation_bounds='default', optimisation_method=None, iteration_number=1, delta=0.5, nu=1., exploitation_weight=1., marginalise_mean_flux=True):
        """Find parameter vector (marginalised over mean flux parameters) at maximum of (GP-UCB) acquisition function"""
        #We marginalise the mean flux parameters so they should not be mapped
        if marginalise_mean_flux:
            param_limits_no_mf = self.like.param_limits[2:,:]
        if optimisation_bounds == 'default': #Default to prior bounds
            #optimisation_bounds = [tuple(self.param_limits[2 + i]) for i in range(starting_params.shape[0])]
            optimisation_bounds = [(1.e-7, 1. - 1.e-7) for i in range(starting_params.shape[0])] #Might get away with 1.e-7

        mapped = lambda parameter_vector: map_from_unit_cube(parameter_vector, param_limits_no_mf)
        optimisation_function = lambda parameter_vector: -1.*self.acquisition_function_GP_UCB(mapped(parameter_vector),
                                                                                                iteration_number=iteration_number, delta=delta, nu=nu,
                                                                                                exploitation_weight=exploitation_weight,
                                                                                                marginalise_mean_flux=marginalise_mean_flux)
        min_result = spo.minimize(optimisation_function, map_to_unit_cube(starting_params, param_limits_no_mf), method=optimisation_method, bounds=optimisation_bounds)
        if not min_result.success:
            print(min_result)
            raise ValueError(min_result.message)
        return map_from_unit_cube(min_result.x, param_limits_no_mf)

    def refinement(self, nsamples):
        """Do the refinement step."""
        new_samples = self.like.emulator.build_params(nsamples=nsamples)
        assert np.shape(new_samples)[0] == nsamples
        self.like.emulator.gen_simulations(nsamples=nsamples, samples=new_samples)
