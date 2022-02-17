"""
Functions to select the optimal HF samples, and validate the selection.
"""
from typing import List, Optional

import numpy as np

from lyaemu.mf_emulator.hf_optimizes import *


def hf_optimize(
        num_selected: int,
        filename: str = "data/emu_30mpc_lores/cc_emulator_flux_vectors_tau1000000.hdf5",
        json_name: str = "data/emu_30mpc_lores/emulator_params.json",
        selected_index: Optional[List] = None,
        n_optimization_restarts: int = 10,
    ) -> None:
    """
    Optimize the HF choice and validate the choice using direct search.
    """

    flux_lf = FluxVectorLowFidelity(filename, json_name)

    # if you previously selected index, direct search the num_selected - 1, and
    # search the (num_selected)th one.
    if selected_index is None:

        # direct search optimal (num_selected-1)
        all_z_loss, loss_sum_z, all_z_selected_index, selected_index = direct_search(
            flux_lf, num_selected=num_selected - 1, n_optimization_restarts=n_optimization_restarts,
        )

    # search for the optimal (num_selected)th
    all_z_next_loss, loss_sum_z, all_z_next_selected_index, selected_index_sum_z = search_next(
        flux_lf, all_z_selected_index, selected_index, n_optimization_restarts=n_optimization_restarts,
    )

    # [Direct search] for validation.
    all_z_loss3, loss_sum_z3, all_z_selected_index3, selected_index_sum_z3 = direct_search(
        flux_lf, num_selected=num_selected, n_optimization_restarts=n_optimization_restarts,
    )

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
