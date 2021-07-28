"""Separate file for doing Bayesian optimisation on a likelihood."""
import math
import mpmath as mmh
import numpy as np
import numpy.random as npr
import scipy.optimize as spo
import scipy.interpolate
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

class BayesianOpt(likelihood.LikelihoodClass):
    """Class to add Bayesian optimisation methods to likelihood."""
    def loglike_marginalised_mean_flux(self, params, include_emu=True, integration_bounds='default', integration_options='gauss-legendre', verbose=False, integration_method='Quadrature'): #marginalised_axes=(0, 1)
        """Evaluate (Gaussian) likelihood marginalised over mean flux parameter axes: (dtau0, tau0)"""
        #assert len(marginalised_axes) == 2
        assert self.mf_slope
        if integration_bounds == 'default':
            integration_bounds = [list(self.param_limits[0]), list(self.param_limits[1])]
        likelihood_function = lambda dtau0, tau0: mmh.exp(self.likelihood(np.concatenate(([dtau0, tau0], params)), include_emu=include_emu))
        if integration_method == 'Quadrature':
            integration_output = mmh.quad(likelihood_function, integration_bounds[0], integration_bounds[1], method=integration_options, error=True, verbose=verbose)
        elif integration_method == 'Monte-Carlo':
            integration_output = (self._do_Monte_Carlo_marginalisation(likelihood_function, n_samples=integration_options),)
        return float(mmh.log(integration_output[0]))

    def _do_Monte_Carlo_marginalisation(self, function, n_samples=6000):
        """Marginalise likelihood by Monte-Carlo integration"""
        random_samples = self.param_limits[:2, 0, np.newaxis] + (self.param_limits[:2, 1, np.newaxis] - self.param_limits[:2, 0, np.newaxis]) * npr.rand(2, n_samples)
        function_sum = 0.
        for i in range(n_samples):
            print('Likelihood function evaluation number =', i + 1)
            function_sum += function(random_samples[0, i], random_samples[1, i])
        volume_factor = (self.param_limits[0, 1] - self.param_limits[0, 0]) * (self.param_limits[1, 1] - self.param_limits[1, 0])
        return volume_factor * function_sum / n_samples

    def get_GP_UCB_exploration_term(self, params, iteration_number=1, delta=0.5, nu=1.,
                                    marginalise_mean_flux = True, use_updated_training_set=False):
        """Evaluate the exploration term of the GP-UCB acquisition function"""
        assert iteration_number >= 1.
        assert 0. < delta < 1.
        #if self._inverse_BOSS_covariance_full is None:
            #self._inverse_BOSS_covariance_full = invert_block_diagonal_covariance(self.get_BOSS_error(-1), self.zout.shape[0])
        exploration_weight = math.sqrt(nu * 2. * math.log((iteration_number**((np.shape(params)[0] / 2.) + 2.)) * (math.pi**2) / 3. / delta))
        #Exploration term: least accurate part of the emulator
        if not marginalise_mean_flux:
            okf, _, std = self.get_predicted(params, use_updated_training_set=use_updated_training_set)
        else:
            #Compute the error averaged over the mean flux
            dtau0 = np.mean([self.param_limits[0, 0], self.param_limits[0, 1]])
            tau0 = np.mean([self.param_limits[1, 0], self.param_limits[1, 1]])
            okf, _, std = self.get_predicted(np.concatenate([[dtau0, tau0], params]), use_updated_training_set=use_updated_training_set)
            n_samples = 0
            #We don't really need to do this, because the interpolation error should always be dominated
            #by the position in simulation parameter space: we have always used multiple mean flux points.
            #So just use the error at the average mean flux value
            if n_samples > 0:
                for dtau0 in np.linspace(self.param_limits[0, 0], self.param_limits[0, 1], num=n_samples):
                    for tau0 in np.linspace(self.param_limits[1, 0], self.param_limits[1, 1], num=n_samples):
                        _, _, std_loc = self.get_predicted(np.concatenate([[dtau0, tau0], params]), use_updated_training_set=use_updated_training_set)
                        for ii, ss in enumerate(std_loc):
                            std[ii] += ss
                for ss in std:
                    ss/=(n_samples**2+1)
        #Do the summation of sigma_emu^T \Sigma^{-1}_{BOSS} sigma_emu (ie, emulator error convolved with data covariance)
        posterior_estimated_error = 0
        nz = np.shape(std)[0]
        #Likelihood using full covariance matrix
        for bb in range(nz):
            idp = np.where(self.kf >= okf[bb][0])
            std_bin = std[bb]
            bindx = np.min(idp)
            covar_bin = self.get_BOSS_error(bb)[bindx:, bindx:]
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
                loglike = self.likelihood(params)
            exploitation = loglike * exploitation_weight
        print("acquis: %g explor: %g exploit:%g params:" % (exploitation+exploration,exploration,exploitation), params)
        return exploration + exploitation

    def optimise_acquisition_function(self, starting_params, datadir=None, optimisation_bounds='default', optimisation_method=None, iteration_number=1, delta=0.5, nu=1., exploitation_weight=1., marginalise_mean_flux=True):
        """Find parameter vector (marginalised over mean flux parameters) at maximum of (GP-UCB) acquisition function"""
        #We do not want the DLA model corrections enabled here
        assert not self.dla_data_corr
        #We marginalise the mean flux parameters so they should not be mapped
        if marginalise_mean_flux:
            param_limits_no_mf = self.param_limits[2:,:]
            assert self.mf_slope
        if datadir is not None:
            self.data_fluxpower = likelihood.load_data(datadir, kf=self.kf, t0=self.t0_training_value)
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

    def check_for_refinement(self, conf=0.95, thresh=1.05):
        """Crude check for refinement: check whether the likelihood is dominated by
           emulator error at the 1 sigma contours."""
        limits = self.new_parameter_limits(confidence=conf, include_dense=True)
        while True:
            #Do the check
            uref = self.refine_metric(limits[:, 0])
            lref = self.refine_metric(limits[:, 1])
            #This should be close to 1.
            print("up =", uref, " low=", lref)
            if (uref < thresh) and (lref < thresh):
                break
            #Iterate by moving each limit 40% outwards.
            midpt = np.mean(limits, axis=1)
            limits[:, 0] = 1.4*(limits[:, 0] - midpt) + midpt
            limits[:, 0] = np.max([limits[:, 0], self.param_limits[:, 0]], axis=0)
            limits[:, 1] = 1.4*(limits[:, 1] - midpt) + midpt
            limits[:, 1] = np.min([limits[:, 1], self.param_limits[:, 1]], axis=0)
            if np.all(limits == self.param_limits):
                break
        return limits

    def refinement(self, nsamples, confidence=0.99):
        """Do the refinement step."""
        new_limits = self.new_parameter_limits(confidence=confidence)
        new_samples = self.emulator.build_params(nsamples=nsamples, limits=new_limits)
        assert np.shape(new_samples)[0] == nsamples
        self.emulator.gen_simulations(nsamples=nsamples, samples=new_samples)

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
