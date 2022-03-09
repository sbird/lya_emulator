"""
A class to handle optimizing the choice of high-res simulations
using low-res only emulators.
"""

import numpy as np
import GPy

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray, y_pred_variance=None) -> float:
    """
    Mean squared error

        MSE = 1/N \Sum (y_true - y_pred)^2
    """
    return np.mean((y_true - y_pred)**2)


class TrainSetOptimize:
    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.X = X
        self.Y = Y

    def loss(self, ind: np.ndarray, loss_fn=None, n_optimization_restarts:int = 5):
        """
        Train a GP conditioned on (X[ind], Y[ind]),
        return the loss average over (X[~ind], Y[~ind])

        Parameters:
        ----
        ind: boolean array, indicator function for the training data.
        loss_fn: the loss function we used. If not specified, mean squared errors.
        """
        assert ind.dtype == np.bool

        # train a GP across all k bins
        _, nparams = self.X.shape

        # here I hard coded the kernel
        kernel = GPy.kern.RBF(nparams, ARD=True)
        gp = GPy.models.GPRegression(self.X[ind], self.Y[ind], kernel)

        gp.optimize_restarts(n_optimization_restarts)

        # predicting on the rest of X
        mean, variance = gp.predict(self.X[~ind])

        if not loss_fn:
            loss_fn = mean_squared_error

        loss = loss_fn(self.Y[~ind], mean, variance)

        return loss

    def optimize(self, prev_ind: np.ndarray, loss_fn=None, n_optimization_restarts:int = 5) -> None:
        """
        Find the optimal index in the X space for the next run,
        via optimizing the acquisition function (or loss function).

        Parameters:
        ----
        selected_ind: the index

        Return:
        ----
        (optimal index, loss values)
        """
        assert prev_ind.dtype == np.bool

        all_loss = []

        n_samples, _ = self.X.shape

        rest_index = np.arange(n_samples)[~prev_ind]

        for i in rest_index:
            ind = np.zeros(n_samples, dtype=np.bool)

            # set previous index and the additional index
            ind[prev_ind] = True
            ind[i] = True

            assert np.sum(ind) == (np.sum(prev_ind) + 1)

            loss = self.loss(ind, loss_fn=loss_fn, n_optimization_restarts=n_optimization_restarts)

            all_loss.append(loss)

        # get the index for minimizing loss
        I = np.argmin(all_loss)

        return (rest_index[I], all_loss)
