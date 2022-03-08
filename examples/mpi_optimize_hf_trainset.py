"""
Functions to select the optimal HF samples, and validate the selection.
"""
from typing import List, Optional

import os, json
import numpy as np
from mpi4py import MPI

from lyaemu.mf_emulator.hf_optimizes import *
from lyaemu.mf_emulator.mpi_hf_optimizes import direct_search as mpi_direct_search

# MPI World:
# ----
comm = MPI.COMM_WORLD     # shared by all computers
my_rank = comm.Get_rank() # id for this computer

p = comm.Get_size()       # number of processors


def hf_optimize(
        num_selected: int,
        filename: str = "data/emu_30mpc_lores/cc_emulator_flux_vectors_tau1000000.hdf5",
        json_name: str = "data/emu_30mpc_lores/emulator_params.json",
        selected_index: Optional[List] = None,
        n_optimization_restarts: int = 10,
        outname: str = "", ## added filename
    ) -> None:
    """
    Optimize the HF choice and validate the choice using direct search.
    """

    # fix the random seed
    np.random.seed(0)

    flux_lf = FluxVectorLowFidelity(filename, json_name)

    if selected_index is not None:
        # your previously selected index is shorted than num_selected
        assert len(selected_index) + 1 == num_selected

        selected_index = np.array(selected_index)

    # if you previously selected index, direct search the num_selected - 1, and
    # search the (num_selected)th one.
    if selected_index is None:

        # direct search optimal (num_selected-1)
        # only get the return from rank 0
        if my_rank == 0:
            all_z_loss, loss_sum_z, all_z_selected_index, selected_index = mpi_direct_search(
                flux_lf, num_selected=num_selected - 1, n_optimization_restarts=n_optimization_restarts,
            )
            # search for the optimal (num_selected)th
            all_z_next_loss, loss_sum_z, all_z_next_selected_index, selected_index_sum_z = search_next(
                flux_lf, all_z_selected_index, selected_index, n_optimization_restarts=n_optimization_restarts,
            )

        else:
            mpi_direct_search(
                flux_lf, num_selected=num_selected - 1, n_optimization_restarts=n_optimization_restarts,
            )

    else:
        # search_next currently not parallelized
        if my_rank == 0:
            # search for the optimal (num_selected)th
            loss_sum_z, selected_index_sum_z = search_next(
                flux_lf, None, selected_index, n_optimization_restarts=n_optimization_restarts,
            )


    # [Direct search] for validation.
    if my_rank == 0:
        all_z_loss3, loss_sum_z3, all_z_selected_index3, selected_index_sum_z3 = mpi_direct_search(
            flux_lf, num_selected=num_selected, n_optimization_restarts=n_optimization_restarts,
        )
    else:
        mpi_direct_search(
            flux_lf, num_selected=num_selected, n_optimization_restarts=n_optimization_restarts,
        )

    # result section only runs in rank 0
    if my_rank == 0:
        # Summarize the results
        print("Search next for {} optimals:".format(num_selected), selected_index_sum_z)
        print("Direct search for {} optimals:".format(num_selected), selected_index_sum_z3)

        # validate the select next optimal
        print("Performance from searching next for the 3rd optimal:")
        print("----")
        better_than(
            num_selected   = num_selected,
            num_samples    = flux_lf.num_samples,
            selected_index = selected_index_sum_z, # here changed
            all_z_loss     = all_z_loss3,
            loss_sum_z     = loss_sum_z3,
            zout           = flux_lf.zout,
        )
        print("")

        print("Performance from directly selecting the optimal three:")
        print("----")
        better_than(
            num_selected   = num_selected,
            num_samples    = flux_lf.num_samples,
            selected_index = selected_index_sum_z3, # here changed
            all_z_loss     = all_z_loss3,
            loss_sum_z     = loss_sum_z3,
            zout           = flux_lf.zout,
        )

        # print the loaded filenames
        print("Used files:", filename, json_name)
        print("Previous selected index", selected_index)
        print("N optimization restarts", n_optimization_restarts)

        # to avoid running again
        folder_name = "hf_{}_{}_{}".format(num_selected, "-".join(map(str, selected_index)), n_optimization_restarts)
        outdir = os.path.join("output", "hf_optimals", folder_name + outname)
        os.makedirs(outdir, exist_ok=True)

        basic_info = {
            "filename" : filename,
            "json_name" : json_name,
            "num_selected" : num_selected,
            "selected_index" : selected_index.tolist(),
            "selected_index_sum_z": selected_index_sum_z.tolist(),
            "selected_index_sum_z3": selected_index_sum_z3.tolist(),
            "n_optimization_restarts" : n_optimization_restarts,
        }

        np.savetxt(os.path.join(outdir, "all_z_loss3"), all_z_loss3)
        np.savetxt(os.path.join(outdir, "loss_sum_z3"), loss_sum_z3)
        np.savetxt(os.path.join(outdir, "selected_index_sum_z"), selected_index_sum_z)
        np.savetxt(os.path.join(outdir, "selected_index_sum_z3"), selected_index_sum_z3)

        with open(os.path.join(outdir, "basic_info.json"), "w") as f:
            json.dump(basic_info, f, indent=4)
