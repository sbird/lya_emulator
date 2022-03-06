from typing import List

import numpy as np
from mpi4py import MPI

from itertools import combinations

from .hf_optimizes import FluxVectorLowFidelity
from .mf_trainset_optimize import TrainSetOptimize

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

    # MPI World:
    # ----
    comm = MPI.COMM_WORLD     # shared by all computers
    my_rank = comm.Get_rank() # id for this computer

    p = comm.Get_size()       # number of processors

    # MPI settings
    # ----
    n = len(all_combinations) # total number of loops used MPI
    dest = 0                  # receiver destination processor
    local_n = int(n / p)      # number of loops per processors;
    assert (local_n - n/p) < 1e-5 # make sure n/p is integer

    # Each MPI rank runs trough all zs
    all_z_loss = np.full((len(data.zout), n), fill_value=np.nan, dtype=np.float)

    all_z_local_loss = np.full((len(data.zout), local_n), fill_value=np.nan, dtype=np.float)

    # loop over all redshifts, but separate the loop of combinations into MPI ranks
    for i, z in enumerate(data.zout):

        print("[Info] Getting flux vector at z = {:.3g} ...".format(z))

        # if not log scale, some selections cannot be trained, which might
        # indicate log scale is a better normalization for GP
        Y = np.log10(data.get_flux_vector_at_z(i))

        # Start MPI loops
        # ----
        local_i = my_rank * local_n                 # start point to run on this processor
        local_f = local_i + local_n                 # end   point to run on this processor

        local_loss = compute_local_loss(
            local_i, local_f, all_combinations, data.X, Y, data.num_samples, n_optimization_restarts
        )
        local_loss = np.array(local_loss)

        all_z_local_loss[i, :] = local_loss


    # the receiver rank
    if my_rank == 0:
        all_z_loss[i, local_i:local_f] = all_z_local_loss

        # receive the results for each mpi rank
        for source in range(1, p):
            all_z_local_loss_i = np.full(all_z_local_loss.shape, fill_value=np.nan, dtype=np.float)
            comm.Recv([all_z_local_loss_i, MPI.FLOAT], source=source)
            print("Passing", my_rank, "<-", source)

            local_i = source * local_n
            local_f = local_i + local_n

            all_z_loss[i, local_i:local_f] = all_z_local_loss_i

    # send message to the receiver rank
    else:
        comm.Send([all_z_local_loss, MPI.FLOAT], dest=dest)

    MPI.Finalize

    # [sum over all redshifts] To account for all available redshift,
    # we sum the MSE loss from each z. The optimal one should have the
    # lowest average MSE loss across all z. 
    loss_sum_z = np.nansum(all_z_loss, axis=0)

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


def compute_local_loss(
        local_i: int,
        local_f: int,
        all_combinations: List, 
        X: np.ndarray,
        Y: np.ndarray,
        num_samples: int,
        n_optimization_restarts: int
    ) -> List:

    # make the object here instead of making copies
    train_opt = TrainSetOptimize(X=X, Y=Y)

    # loop over all combinations
    local_loss = []

    for selected_index in all_combinations[local_i:local_f]:

        # need to convert to boolean array
        ind = np.zeros(num_samples, dtype=np.bool)
        ind[np.array(selected_index)] = True

        loss = train_opt.loss(ind, n_optimization_restarts=n_optimization_restarts)

        local_loss.append(loss)

    return local_loss
