"""
Search the next optimal training point given previous selected points.
"""
from typing import Optional, List

import os, json

import numpy as np

from lyaemu.mf_emulator.hf_optimizes import FluxVectorLowFidelity, search_next


def search_next_hf_trainset(
        num_selected: int,
        filename: str = "data/emu_30mpc_lores/cc_emulator_flux_vectors_tau1000000.hdf5",
        json_name: str = "data/emu_30mpc_lores/emulator_params.json",
        selected_index: List = [1, 17, 18],
        n_optimization_restarts: int = 10,
        outname: str = "", ## added filename
    ) -> None:
    """
    Parameters:
    ----
    num_selected: Number of samples to be selected.
    filename: The h5 file contains the low-fidelity power spectra.
    selected_index: The previously selected indices. The length is num_selected - 1.
    n_optimization_restarts: Number of optimization restarts of GP training.
    outname: Suffix for the output filename.

    Note:
    ----
    Default save path = "output/hf_optimals/hf_{num_selected}_{selected_index}_{n_optimization_restarts}/search_next/
    """

    # fix the random seed
    np.random.seed(0)

    flux_lf = FluxVectorLowFidelity(filename, json_name)

    # your previously selected index is shorted than num_selected
    assert len(selected_index) + 1 == num_selected

    selected_index = np.array(selected_index)

    # search for the optimal (num_selected)th
    loss_sum_z, selected_index_sum_z = search_next(
        flux_lf, None, selected_index, n_optimization_restarts=n_optimization_restarts,
    )

    # Summarize the results
    print("Search next for {} optimals:".format(num_selected), selected_index_sum_z)

    # print the loaded filenames
    print("Used files:", filename, json_name)
    print("Previous selected index", selected_index)
    print("N optimization restarts", n_optimization_restarts)

    # to avoid running again
    folder_name = "hf_{}_{}_{}".format(num_selected, "-".join(map(str, selected_index)), n_optimization_restarts)
    outdir = os.path.join("output", "hf_optimals", folder_name + outname, "search_next")
    os.makedirs(outdir, exist_ok=True)


    basic_info = {
        "filename" : filename,
        "json_name" : json_name,
        "num_selected" : num_selected,
        "selected_index" : selected_index.tolist(),
        "selected_index_sum_z": selected_index_sum_z.tolist(),
        "n_optimization_restarts" : n_optimization_restarts,
    }

    np.savetxt(os.path.join(outdir, "selected_index_sum_z"), selected_index_sum_z)
    np.savetxt(os.path.join(outdir, "loss_sum_z"), loss_sum_z)


    with open(os.path.join(outdir, "basic_info.json"), "w") as f:
        json.dump(basic_info, f, indent=4)
