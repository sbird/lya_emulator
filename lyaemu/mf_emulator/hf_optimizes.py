"""
Procedures to optimize the hf training set,

- Optimize condition on individual redshifts
- Optimize across redshifts
- Optimize directly
- Search for the optimal next sample
"""
import json
# for getting all combinations of the highres choice
from itertools import combinations

import numpy as np
import h5py

from .dataloader import input_normalize
from .mf_trainset_optimize import TrainSetOptimize

# set a random number seed for reproducibility
np.random.seed(0)


class FluxVectorLowFidelity:

    """
    A class for optimizing the HF samples from LF.
    Should be able to load single-fidelity data.
    """

    def __init__(
        self,
        filename: str = "data/emu_30mpc_lores/cc_emulator_flux_vectors_tau1000000.hdf5",
        json_name: str = "data/emu_30mpc_lores/emulator_params.json",
    ) -> None:
        
        # read param file and the flux power spectra
        with open(json_name, "r") as param:
            param = json.load(param)

        f = h5py.File(filename, "r")

        print("All keys", f.keys())

        print("Shape of redshfits", f["zout"].shape)
        print("Shape of params", f["params"].shape)
        print("Shape of kfkms", f["kfkms"].shape)
        print("Shape of kfmpc", f["kfmpc"].shape)
        print("Shape of flux vectors", f["flux_vectors"].shape)

        # use kfmpc so all redshifts use the same k bins
        kfmpc = f["kfmpc"][()]

        zout = f["zout"][()]

        # flux power spectra, all redshifts
        flux_vectors = f["flux_vectors"][()]


        # for input normalization
        param_limits = np.array(param["param_limits"])
        X_train = f["params"][()]

        # No need to normalize since only single fidelity
        # but it's better to map param to a unit cube to avoid
        # ARD hyperparameter optimization searching in a wide
        # dynamical range
        X = input_normalize(X_train, param_limits=param_limits)
        print("[Info] Input parameters are normalized to a unit cube.")

        num_samples, _ = X.shape

        # Assign class attributes
        self.kfmpc = kfmpc
        self.zout = zout
        self.flux_vectors = flux_vectors

        self.X_train = X_train
        self.X = X # normalized
        self.num_samples = num_samples

        # some tests
        last_flux_vector = self.get_flux_vector_at_z(len(zout) - 1)
        assert len(last_flux_vector) == f["params"].shape[0]
        assert last_flux_vector.shape[1] == len(kfmpc)

        first_flux_vector = self.get_flux_vector_at_z(0)
        assert len(first_flux_vector) == f["params"].shape[0]
        assert first_flux_vector.shape[1] == len(kfmpc)


    def get_flux_vector_at_z(self, i: int) -> np.ndarray:
        """
        read flux power spectrum at ith redshift
        flux power spectrum | z = ?

        Parameters:
        ----
        i : index for the redshift, high->low.
        """
        return self.flux_vectors[:, i * len(self.kfmpc) : (i + 1) * len(self.kfmpc)]


def direct_search(
        data: FluxVectorLowFidelity,
        num_selected: int = 3,
        n_optimization_restarts: int = 10,
    ):
    """
    Direct search the optimal N HF from LF simulations.

    Low-fidelity only emulator -

        f ~ GP(X_train, Y_train)

        X_train = input parameters in a unit cube
        Y_train = log10(flux power spectra)
        
        loss = sum_{zi}^{zf} (MSE(z))
    """


    # looking for all possible combinations
    all_combinations = list(combinations(range(data.num_samples), num_selected))

    all_z_loss = []

    # loop over all redshifts
    for i, z in enumerate(data.zout):

        # if not log scale, some selections cannot be trained, which might
        # indicate log scale is a better normalization for GP
        Y = np.log10(data.get_flux_vector_at_z(i))
        print("[Info] Getting flux vector at z = {:.3g} ...".format(z))
        print("[Info] Turn the flux power into log scale.")

        train_opt = TrainSetOptimize(X=data.X, Y=Y)

        # loop over all combinations
        all_this_loss = []
        for j,selected_index in enumerate(all_combinations):

            # need to convert to boolean array
            ind = np.zeros(data.num_samples, dtype=np.bool)
            ind[np.array(selected_index)] = True

            loss = train_opt.loss(ind, n_optimization_restarts=n_optimization_restarts)

            print("iteration: {} out of {}".format((i, j), (len(data.zout), len(all_combinations))))

            all_this_loss.append(loss)

        all_z_loss.append(all_this_loss)

    # [sum over all redshifts] To account for all available redshift,
    # we sum the MSE loss from each z. The optimal one should have the
    # lowest average MSE loss across all z. 
    loss_sum_z = np.array(all_z_loss).sum(axis=0)


    # find the best samples to minimize the loss
    all_z_selected_index = [] # this is for individual z

    # need to print for all redshifts
    for i,z in enumerate(data.zout):

        all_loss = all_z_loss[i]

        # the optimal one is the one has lowest MSE loss
        selected_index = np.array(all_combinations[np.argmin(all_loss)])

        print("Optimal selection for z = {:.3g}".format(z))
        print(selected_index)

        all_z_selected_index.append(selected_index)

    # [sum over all redshifts]
    selected_index_sum_z = np.array(all_combinations[np.argmin(loss_sum_z)])
    print("Optimal selection (summing over all redshifts)", selected_index_sum_z)

    return all_z_loss, loss_sum_z, all_z_selected_index, selected_index_sum_z

def search_next(
        data: FluxVectorLowFidelity,
        all_z_selected_index: np.ndarray,
        selected_index_sum_z: np.ndarray,
        n_optimization_restarts: int = 10,
    ) -> np.ndarray:
    """
    Condition on the selected index, search for next.

    Parameters:
    ----
    data: FluxVectorLowFidelity, includes multiple redshifts.
    all_z_selected_index: Optimal selection for individual z, from high->low z.
                          shape=(num of zs, num of selected points)
    selected_index_sum_z: Optimal selection across zs.
                          shape=(num of selected points, )
    """
    ## First Part: individual z
    all_z_next_loss = []
    all_z_next_selected_index = []

    # Loop over all redshifts.
    # You still need to loop over (X, Y) for train_opt.
    for i, selected_index in enumerate(all_z_selected_index):

        # if not log scale, some selections cannot be trained, which might
        # indicate log scale is a better normalization for GP
        Y = np.log10(data.get_flux_vector_at_z(i))
        print("[Info] Getting flux vector at z = {:.3g} ...".format(data.zout[i]))
        print("[Info] Turn the flux power into log scale.")

        # you need to instantiate the train_open per loop, since
        # Y is different for different zs.
        train_opt = TrainSetOptimize(X=data.X, Y=Y)

        # turn integers into a boolean array
        prev_ind = np.zeros(data.num_samples, dtype=np.bool)
        prev_ind[np.array(selected_index)] = True

        assert np.sum(prev_ind) == len(selected_index)
        
        # this method loop over all index \in {1..N} - {prev_ind}
        # and compute the loss
        next_index, next_loss = train_opt.optimize(prev_ind, n_optimization_restarts=n_optimization_restarts)

        # optimal next selection indices
        selected_index = np.append(selected_index, next_index)

        assert np.where(~prev_ind)[0][np.argmin(next_loss)] == next_index

        all_z_next_selected_index.append(selected_index)
        all_z_next_loss.append(next_loss)


    ## Second Part: Summing over all z
    all_z_next_loss_sum_z = []

    # don't have to loop this since you only have one selection across z
    prev_ind = np.zeros(data.num_samples, dtype=np.bool)
    prev_ind[np.array(selected_index_sum_z)] = True
    assert np.sum(prev_ind) == len(selected_index_sum_z)

    for i, z in enumerate(data.zout):

        Y = np.log10(data.get_flux_vector_at_z(i))
        print("[Info] Getting flux vector at z = {:.3g} ...".format(z))
        print("[Info] Turn the flux power into log scale.")

        train_opt = TrainSetOptimize(X=data.X, Y=Y)

        # this method loop over all index \in {1..N} - {prev_ind}
        # and compute the loss
        next_index, next_loss = train_opt.optimize(prev_ind, n_optimization_restarts=n_optimization_restarts)

        all_z_next_loss_sum_z.append(next_loss)

        assert np.where(~prev_ind)[0][np.argmin(next_loss)] == next_index


    # [sum over all redshifts] To account for all available redshift,
    loss_sum_z = np.array(all_z_next_loss_sum_z).sum(axis=0)

    next_index_sum_z = np.where(~prev_ind)[0][np.argmin(loss_sum_z)]

    # optimal next selection indices
    selected_index_sum_z = np.append(selected_index_sum_z, next_index_sum_z)


    # print all
    for i,z in enumerate(data.zout):

        selected_index = all_z_next_selected_index[i]

        print("Optimal three for z = {:.3g}".format(z), selected_index)

    print("Optimal three (summing over all redshifts)", selected_index_sum_z)

    return all_z_next_loss, loss_sum_z, all_z_next_selected_index, selected_index_sum_z

