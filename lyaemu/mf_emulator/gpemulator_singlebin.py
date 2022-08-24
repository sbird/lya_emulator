"""
Building multi-Fidelity emulator using many single-output GP.

1. SingleBinGP: the single-fidelity emulator in the paper.
2. SingleBinLinearGP: the linear multi-fidelity emulator (AR1).
3. SingleBinNonLinearGP: the non-linear multi-fidelity emulator (NARGP).
4. SingleBinDeepGP: the deep GP for multi-fidelity (MF-DGP). This one is not
    mentioned in the paper due to we haven't found a way to fine-tune the
    hyperparameters.

Most of the model constructions are similar to Emukit's examples, with some
modifications on the choice of hyperparameters and modelling each output as
an independent GP (many single-output GP).
"""
from typing import Tuple, List, Optional, Dict, Type

import logging
import numpy as np

import GPy
from emukit.model_wrappers import GPyMultiOutputWrapper

from emukit.multi_fidelity.kernels import LinearMultiFidelityKernel
from emukit.multi_fidelity.convert_lists_to_array import (
    convert_y_list_to_array,
    convert_xy_lists_to_arrays,
    convert_x_list_to_array,
)

# we made modifications on not using the ARD for high-fidelity, but we can
# just have the modifications in make_non_linear_kernels
from emukit.multi_fidelity.models.non_linear_multi_fidelity_model import NonLinearMultiFidelityModel

_log = logging.getLogger(__name__)

def make_non_linear_kernels(base_kernel_class: Type[GPy.kern.Kern],
                            n_fidelities: int, n_input_dims: int, ARD: bool=False,
                            n_output_dim: int = 1,
                            turn_off_bias: bool = False,
                            ARD_last_fidelity: bool = False) -> List:
    """
    This function takes a base kernel class and constructs the structured multi-fidelity kernels

    At the first level the kernel is simply:
    .. math
        k_{base}(x, x')

    At subsequent levels the kernels are of the form
    .. math
        k_{base}(x, x')k_{base}(y_{i-1}, y{i-1}') + k_{base}(x, x')

    If turn off bias (added by jibancat):
    .. math
        k_{base}(x, x')k_{base}(y_{i-1}, y{i-1}')

    :param base_kernel_class: GPy class definition of the kernel type to construct the kernels at
    :param n_fidelities: Number of fidelities in the model. A kernel will be returned for each fidelity
    :param n_input_dims: The dimensionality of the input.
    :param ARD: If True, uses different lengthscales for different dimensions. Otherwise the same lengthscale is used
                for all dimensions. Default False.
    :param turn_off_bias: do not apply bias kernel for high-fidelity.
    :param ARD_last_fidelity: whether to apply ARD for the last (highest) fidelity.
        Default, False.

    :return: A list of kernels with one entry for each fidelity starting from lowest to highest fidelity.
    """

    base_dims_list = list(range(n_input_dims))
    out_dims_list = list(range(n_input_dims, n_input_dims + n_output_dim))
    kernels = [
        base_kernel_class(n_input_dims, active_dims=base_dims_list, ARD=ARD, name='kern_fidelity_1')
        # TODO: check added Linear
        + GPy.kern.Linear(n_input_dims, active_dims=base_dims_list, ARD=ARD, name='kern_fidelity_1_linear')
    ]
    for i in range(1, n_fidelities):
        fidelity_name = 'fidelity' + str(i + 1)
        interaction_kernel = base_kernel_class(n_input_dims, active_dims=base_dims_list, ARD=ARD_last_fidelity,
                                               name='scale_kernel_no_ARD_' + fidelity_name)
        scale_kernel = base_kernel_class(
            n_output_dim, active_dims=out_dims_list, name='previous_fidelity_' + fidelity_name
        ) + GPy.kern.Linear(n_output_dim, active_dims=out_dims_list, name='previous_fidelity_' + fidelity_name + "_linear") # TODO: check added Linear
        bias_kernel = base_kernel_class(n_input_dims, active_dims=base_dims_list,
                                        ARD=ARD_last_fidelity, name='bias_kernel_no_ARD_' + fidelity_name)
        if turn_off_bias:
            kernels.append(interaction_kernel * scale_kernel)
        else:
            kernels.append(interaction_kernel * scale_kernel + bias_kernel)
    return kernels


class SingleBinGP:
    """
    A GPRegression models GP on each k bin of powerspecs
    
    :param X_hf: (n_points, n_dims) input parameters
    :param Y_hf: (n_points, k modes) power spectrum
    """
    def __init__(self, X_hf: np.ndarray, Y_hf: np.ndarray):
        # a list of GP emulators
        gpy_models: List = []

        # make a GP on each P(k) bin
        for i in range(Y_hf.shape[1]):
            # Standard squared-exponential kernel with a different length scale for
            # each parameter, as they may have very different physical properties.
            nparams = np.shape(X_hf)[1]

            kernel = GPy.kern.RBF(nparams, ARD=True)            

            gp = GPy.models.GPRegression(X_hf, Y_hf[:, [i]], kernel)
            gpy_models.append(gp)

        self.gpy_models = gpy_models

        self.name = "single_fidelity"

    def optimize_restarts(self, n_optimization_restarts: int) -> None:
        """
        Optimize GP on each bin of the power spectrum.
        """
        models = []

        _log.info("\n --- Optimization: ---\n".format(self.name))
        for i,gp in enumerate(self.gpy_models):
            gp.optimize_restarts(n_optimization_restarts)
            models.append(gp)

        self.models = models

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts mean and variance for fidelity specified by last column of X.
        Note that we predict from gp from each k bin.

        :param X: point(s) at which to predict
        :return: predicted P(all k bins) (mean, variance) at X
        """
        means = np.full((X.shape[0], len(self.models)), fill_value=np.nan)
        variances = np.full((X.shape[0], len(self.models)), fill_value=np.nan)

        for i,model in enumerate(self.models):
            mean, variance = model.predict(X)

            means[:, i] = mean[:, 0]
            variances[:, i] = variance[:, 0]

        return means, variances

    def to_dict(self) -> Dict:
        """
        Save hyperparameters into a dict
        """
        param_dict = {}

        for i,model in enumerate(self.models):        

            param_dict["bin_{}".format(i)] = model.to_dict()

        return param_dict

class SingleBinLinearGP:
    """
    A thin wrapper around GPy.core.GP that does some input checking and provides
    a default likelihood. Also model each k bin as an independent GP.

    :param X_train:  (n_fidelities, n_points, n_dims) list of parameter vectors.
    :param Y_train:  (n_fidelities, n_points, k modes) list of matter power spectra.
    :param n_fidelities: number of fidelities stored in the list.
    :param ARD_last_fidelity: whether to apply ARD for the last (highest) fidelity.
        Default, False.
    """

    def __init__(
        self,
        X_train: List[np.ndarray],
        Y_train: List[np.ndarray],
        kernel_list: Optional[List],
        n_fidelities: int,
        likelihood: GPy.likelihoods.Likelihood = None,
        ARD_last_fidelity: bool = False,
    ):
        # a list of GP emulators
        gpy_models: List = []

        self.n_fidelities = len(X_train)

        # convert into X,Y for MultiOutputGP
        # not normalize due to no improvements
        X, Y = convert_xy_lists_to_arrays(X_train, Y_train)

        # linear multi-fidelity setup
        if X.ndim != 2:
            raise ValueError("X should be 2d")

        if Y.ndim != 2:
            raise ValueError("Y should be 2d")

        if np.any(X[:, -1] >= n_fidelities):
            raise ValueError(
                "One or more points has a higher fidelity index than number of fidelities"
            )

        # make a GP on each P(k) bin
        for i in range(Y.shape[1]):
            y_metadata = {"output_index": X[:, -1].astype(int)}

            # Make default likelihood as different noise for each fidelity
            likelihood = GPy.likelihoods.mixed_noise.MixedNoise(
                [GPy.likelihoods.Gaussian(variance=1.0) for _ in range(n_fidelities)]
            )

            # Standard squared-exponential kernel with a different length scale for
            # each parameter, as they may have very different physical properties.
            kernel_list = []
            for j in range(n_fidelities):
                nparams = np.shape(X_train[j])[1]

                kernel = GPy.kern.Linear(nparams, ARD=True)
                # kernel = GPy.kern.RatQuad(nparams, ARD=True)
                kernel += GPy.kern.RBF(nparams, ARD=True)
                
                # final fidelity not ARD due to lack of training data
                if j == n_fidelities - 1:
                    kernel = GPy.kern.Linear(nparams, ARD=ARD_last_fidelity)
                    kernel += GPy.kern.RBF(nparams, ARD=ARD_last_fidelity)

                kernel_list.append(kernel)

            # make multi-fidelity kernels
            kernel = LinearMultiFidelityKernel(kernel_list)

            gp = GPy.core.GP(X, Y[:, [i]], kernel, likelihood, Y_metadata=y_metadata)
            gpy_models.append(gp)

        self.gpy_models = gpy_models

        self.name = "ar1"

    def optimize(self, n_optimization_restarts: int) -> None:
        """
        Optimize GP on each bin of the power spectrum.
        """
        models = []

        _log.info("\n--- Optimization ---\n".format(self.name))

        for i,gp in enumerate(self.gpy_models):
            # fix noise and optimize
            getattr(gp.mixed_noise, "Gaussian_noise").fix(1e-6)
            for j in range(1, self.n_fidelities):
                getattr(
                    gp.mixed_noise, "Gaussian_noise_{}".format(j)
                ).fix(1e-6)

            model = GPyMultiOutputWrapper(gp, n_outputs=self.n_fidelities, n_optimization_restarts=n_optimization_restarts)

            _log.info("\n[Info] Optimizing {} bin ...\n".format(i))

            # first step optimization with fixed noise
            model.gpy_model.optimize_restarts(
                n_optimization_restarts,
                verbose=model.verbose_optimization,
                robust=True,
                parallel=False,
            )

            # unfix noise and re-optimize
            getattr(model.gpy_model.mixed_noise, "Gaussian_noise").unfix()
            for j in range(1, self.n_fidelities):
                getattr(
                    model.gpy_model.mixed_noise, "Gaussian_noise_{}".format(j)
                ).unfix()

            # first step optimization with fixed noise
            model.gpy_model.optimize_restarts(
                n_optimization_restarts,
                verbose=model.verbose_optimization,
                robust=True,
                parallel=False,
            )

            models.append(model)

        self.models = models

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts mean and variance for fidelity specified by last column of X.
        Note that we predict from gp from each k bin.

        :param X: point(s) at which to predict
        :return: predicted P(all k bins) (mean, variance) at X
        """
        means = np.full((X.shape[0], len(self.models)), fill_value=np.nan)
        variances = np.full((X.shape[0], len(self.models)), fill_value=np.nan)

        for i,model in enumerate(self.models):
            mean, variance = model.predict(X)

            means[:, i] = mean[:, 0]
            variances[:, i] = variance[:, 0]

        return means, variances

    def to_dict(self) -> Dict:
        """
        Save hyperparameters into a dict
        """
        param_dict = {}

        for i,model in enumerate(self.models):

            this_param_dict = {}
        
            # a constant scaling value
            this_param_dict["scale"] = model.gpy_model.multifidelity.scale.values.tolist()
            # append dict from each key
            for j, kern in enumerate(model.gpy_model.multifidelity.kernels):
                this_param_dict["kern_{}".format(j)] = kern.to_dict()

            param_dict["bin_{}".format(i)] = this_param_dict

        return param_dict


class SingleBinNonLinearGP:
    """
    A thin wrapper around NonLinearMultiFidelityModel. It models each k input as
    an independent GP.

    :param X_train:  (n_fidelities, n_points, n_dims) list of parameter vectors.
    :param Y_train:  (n_fidelities, n_points, k modes) list of matter power spectra.
    :param n_fidelities: number of fidelities stored in the list.
    :param n_samples: Number of samples to use to do quasi-Monte-Carlo integration at each fidelity.
    :param optimization_restarts: number of optimization restarts you want in GPy.
    :param ARD_last_fidelity: whether to apply ARD for the last (highest) fidelity.
        Default, False.
    """

    def __init__(
        self,
        X_train: List[np.ndarray],
        Y_train: List[np.ndarray],
        n_fidelities: int,
        n_samples: int = 500,
        optimization_restarts: int = 30,
        turn_off_bias: bool = False,
        ARD_last_fidelity: bool = False
    ):
        # a list of GP emulators
        models: List = []

        self.n_fidelities = len(X_train)

        # convert into X,Y for MultiOutputGP
        # not normalize due to no improvements
        X, Y = convert_xy_lists_to_arrays(X_train, Y_train)

        # linear multi-fidelity setup
        if X.ndim != 2:
            raise ValueError("X should be 2d")

        if Y.ndim != 2:
            raise ValueError("Y should be 2d")

        if np.any(X[:, -1] >= n_fidelities):
            raise ValueError(
                "One or more points has a higher fidelity index than number of fidelities"
            )

        # make a GP on each P(k) bin
        for i in range(Y.shape[1]):
            # make GP non linear kernel
            base_kernel_1 = GPy.kern.RBF
            kernels = make_non_linear_kernels(
                base_kernel_1, n_fidelities, X.shape[1] - 1, ARD=True, n_output_dim=1,
                turn_off_bias=turn_off_bias, ARD_last_fidelity=ARD_last_fidelity,
            )  # -1 for the multi-fidelity labels

            model = NonLinearMultiFidelityModel(
                X,
                Y[:, [i]],
                n_fidelities,
                kernels=kernels,
                verbose=True,
                n_samples=n_samples,
                optimization_restarts=optimization_restarts,
            )

            models.append(model)

        self.models = models

        self.name = "nargp"

    def optimize(self) -> None:
        """
        Optimize GP on each bin of the power spectrum.
        """

        _log.info("\n--- Optimization: ---\n".format(self.name))

        for i,gp in enumerate(self.models):
            _log.info("\n [Info] Optimizing {} bin ... \n".format(i))

            for m in gp.models:
                m.Gaussian_noise.variance.fix(1e-6)
            
            gp.optimize()

            for m in gp.models:
                m.Gaussian_noise.variance.unfix()
            
            gp.optimize()


    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts mean and variance for fidelity specified by last column of X.
        Note that we predict from gp from each k bin.

        :param X: point(s) at which to predict
        :return: predicted P(all k bins) (mean, variance) at X
        """
        means = np.full((X.shape[0], len(self.models)), fill_value=np.nan)
        variances = np.full((X.shape[0], len(self.models)), fill_value=np.nan)

        for i,model in enumerate(self.models):
            mean, variance = model.predict(X)

            means[:, i] = mean[:, 0]
            variances[:, i] = variance[:, 0]

        return means, variances

    def to_dict(self) -> Dict:
        """
        Save hyperparameters into a dict
        """
        param_dict = {}

        for i,model in enumerate(self.models):
            this_param_dict = {}
            # append a list of kernel paramaters
            for j, m in enumerate(model.models):
                this_param_dict["fidelity_{}".format(j)] = m.kern.to_dict()

            param_dict["bin_{}".format(i)] = this_param_dict

        return param_dict

class SingleBinDeepGP:
    """
    A thin wrapper around MultiFidelityDeepGP. Help to handle inputs.
    
    To run this model, you need additional packages:
    - tensorflow==1.8
    - gpflow==1.3 (Note: it said 1.1.1 on the website, but the code actually only works
        for 1.3 version)
    - pip install git+https://github.com/ICL-SML/Doubly-Stochastic-DGP.git

    Warning: this deepGP code hasn't fully tested on the matter power spectrum we have
        here. Be aware you might need more HR samples for train it.

    :param X_train:  (n_fidelities, n_points, n_dims) list of parameter vectors.
    :param Y_train:  (n_fidelities, n_points, k modes) list of matter power spectra.
    :param n_fidelities: number of fidelities stored in the list.
    """

    def __init__(
        self,
        X_train: List[np.ndarray],
        Y_train: List[np.ndarray],
        n_fidelities: int,
    ):
        # DGP model
        from emukit.examples.multi_fidelity_dgp.multi_fidelity_deep_gp import MultiFidelityDeepGP

        # a list of GP emulators
        models: List = []

        self.n_fidelities = len(X_train)

        # make a GP on each P(k) bin
        for i in range(Y_train[0].shape[1]):

            model = MultiFidelityDeepGP(X_train, [power[:, [i]] for power in Y_train])

            models.append(model)

        self.models = models

        self.name = "dgp"

    def optimize(self) -> None:
        """
        Optimize GP on each bin of the power spectrum.
        """

        _log.info("\n--- Optimization ---\n".format(self.name))

        for i,gp in enumerate(self.models):
            _log.info("\n [Info] Optimizing {} bin ... \n".format(i))
            gp.optimize()

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts mean and variance for fidelity specified by last column of X.
        Note that we predict from gp from each k bin.

        :param X: point(s) at which to predict
        :return: predicted P(all k bins) (mean, variance) at X
        """
        means = np.full((X.shape[0], len(self.models)), fill_value=np.nan)
        variances = np.full((X.shape[0], len(self.models)), fill_value=np.nan)

        for i,model in enumerate(self.models):
            mean, variance = model.predict(X)

            means[:, i] = mean[:, 0]
            variances[:, i] = variance[:, 0]

        return means, variances

    def to_dict(self) -> Dict:
        """
        Save hyperparameters into a dict
        """
        NotImplementedError
