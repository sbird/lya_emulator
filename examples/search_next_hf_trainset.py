"""
Search the next optimal training point given previous selected points.
"""
from typing import Optional, List

import os, json
from itertools import combinations

import numpy as np

from lyaemu.mf_emulator.hf_optimizes import FluxVectorLowFidelity, search_next, better_than


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


def better_than_validation(
        filename: str = "data/emu_30mpc_lores/cc_emulator_flux_vectors_tau1000000.hdf5",
        json_name: str = "data/emu_30mpc_lores/emulator_params.json",
        optimal_index: List = [1, 17, 18],
        direct_search_folder: str = "output/hf_optimals/hf_3_10mpi/direct_search/",
        num_divisions: int = 145,
    ) -> None:

    """
    Validate the selected points are better than ?% of the direct search
    results. Need to load results from direct search.
    """

    # fix the random seed
    np.random.seed(0)

    flux_lf = FluxVectorLowFidelity(filename, json_name)

    num_samples = flux_lf.num_samples
    num_selected = len(optimal_index)
    print("[Info] Selecting {} samples out of {} samples.".format(num_selected, num_samples))

    # looking for all possible combinations
    all_combinations = list(combinations(range(num_samples), num_selected))


    # the optimization losses are save into different folders, now we need to
    # combine all of them into as single array.

    # Init a NaN array, and check the number of NaNs in the end.
    all_loss = np.full(
        (len(flux_lf.zout), len(all_combinations)),
        fill_value=np.nan,
    )
    for nth_division in range(num_divisions):
        # copy paste from validate_hf_trainset
        division_name = "nth_{}_num_{}".format(nth_division, num_divisions)
        loaddir = os.path.join(direct_search_folder, division_name)
        
        # file: (number of redshifts, number of combinations)
        # We need to append axis=1
        filename = "all_z_loss3"

        loss = np.loadtxt(os.path.join(loaddir, filename))

        # ref: mpi_hf_optimizes.direct_search
        all_loss[:, nth_division::num_divisions] = loss[:, :]

    # check number of NaNs
    num_nans = np.isnan(all_loss).sum()
    assert num_nans < 4 # I think we shouldn't have more that 3 optimization failures
                        # if it's more than 3, there might be some bugs.

    # Optimization loss summing over redshifts
    loss_sum_z = np.nansum(all_loss, axis=0)

    # Find the optimal selected points from direct search
    idx = np.argmin(loss_sum_z)
    selected_index_sum_z_direct_search = all_combinations[idx]


    # Summarize the results
    print(
        "Search next for {} optimals:".format(num_selected),
        optimal_index,
        "(Your Input)",
    )
    print(
        "Direct search for {} optimals:".format(num_selected),
        selected_index_sum_z_direct_search,
        "(from Direct Search)",
    )

    # validate the select next optimal
    print("Performance from searching next for the 3rd optimal:")
    print("----")
    better_than(
        num_selected   = num_selected,
        num_samples    = flux_lf.num_samples,
        selected_index = optimal_index, # here changed
        all_z_loss     = all_loss,
        loss_sum_z     = loss_sum_z,
        zout           = flux_lf.zout,
    )
    print("")

    print("Performance from directly selecting the optimal three:")
    print("----")
    better_than(
        num_selected   = num_selected,
        num_samples    = flux_lf.num_samples,
        selected_index = selected_index_sum_z_direct_search, # here changed
        all_z_loss     = all_loss,
        loss_sum_z     = loss_sum_z,
        zout           = flux_lf.zout,
    )

    # print the loaded filenames
    print("Used files:", filename, json_name)
