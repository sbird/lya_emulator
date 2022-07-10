"""
Run the direct search for selecting n optimal points for
high-fidelity simulations.
"""
from typing import List

import os, json

import numpy as np
from mpi4py import MPI


from lyaemu.mf_emulator.hf_optimizes import FluxVectorLowFidelity
from lyaemu.mf_emulator.mpi_hf_optimizes import direct_search as mpi_direct_search


# MPI World:
# ----
comm = MPI.COMM_WORLD     # shared by all computers
my_rank = comm.Get_rank() # id for this computer

p = comm.Get_size()       # number of processors


def direct_search(
        num_selected: int,
        filename: str = "data/emu_30mpc_lores/cc_emulator_flux_vectors_tau1000000.hdf5",
        json_name: str = "data/emu_30mpc_lores/emulator_params.json",
        n_optimization_restarts: int = 10,
        num_divisions: int = 1,
        nth_division: int = 0, # 1 2 3 4 5 6 7 8 9
        outname: str = "", ## added filename
        ) -> None:
    """
    Direct search for n optimal training points, i.e.,
    running for (N)!/(N - n)!n! loops.

    Note:
    ----
    Default save path = "output/hf_optimals/hf_{num_selected}_{n_optimization_restarts}/direct_search/
    """

    # fix the random seed
    np.random.seed(0)

    flux_lf = FluxVectorLowFidelity(filename, json_name)

    print("[Info] {} rank, running {}th division out of {} of divisions".format(
        my_rank, nth_division, num_divisions,
    ))

    # [Direct search] for validation.
    if my_rank != 0:
        mpi_direct_search(
            flux_lf, num_selected=num_selected, n_optimization_restarts=n_optimization_restarts,
            num_divisions=num_divisions, nth_division=nth_division,
        )
    else:
        all_z_loss3, loss_sum_z3, all_z_selected_index3, selected_index_sum_z3, all_combinations = mpi_direct_search(
            flux_lf, num_selected=num_selected, n_optimization_restarts=n_optimization_restarts,
            num_divisions=num_divisions, nth_division=nth_division,
        )

        print("Direct search for {} optimals:".format(num_selected), selected_index_sum_z3)

        # print the loaded filenames
        print("Used files:", filename, json_name)
        print("N optimization restarts", n_optimization_restarts)

        # to avoid running again
        folder_name = "hf_{}_{}".format(num_selected, n_optimization_restarts)
        division_name = "nth_{}_num_{}".format(nth_division, num_divisions)
        outdir = os.path.join("output", "hf_optimals", folder_name + outname, "direct_search", division_name)
        os.makedirs(outdir, exist_ok=True)

        basic_info = {
            "filename" : filename,
            "json_name" : json_name,
            "num_selected" : num_selected,
            "selected_index_sum_z3": selected_index_sum_z3.tolist(),
            "all_z_selected_index3" : [idx.tolist() for idx in all_z_selected_index3],
            "n_optimization_restarts" : n_optimization_restarts,
            "nth_division" : nth_division,
            "num_divisions" : num_divisions,
            "all_combinations" : all_combinations,
        }

        np.savetxt(os.path.join(outdir, "all_z_loss3"), all_z_loss3)
        np.savetxt(os.path.join(outdir, "loss_sum_z3"), loss_sum_z3)
        np.savetxt(os.path.join(outdir, "selected_index_sum_z3"), selected_index_sum_z3)

        with open(os.path.join(outdir, "basic_info.json"), "w") as f:
            json.dump(basic_info, f, indent=4)
