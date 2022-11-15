import json
from itertools import combinations
from typing import List, Optional
import numpy as np
import h5py
from .. import gpemulator as gpemu
from . import t0_gpemulator as t0gpemu

# set a random number seed for reproducibility
np.random.seed(0)

def direct_search(fps_file, t0_file, json_file, num_selected=2, max_z=5.4, min_z=2.0):
    load = h5py.File(fps_file, 'r')
    zout = np.round(load['zout'][:], 1)
    params = load['params'][:]
    kfmpc = load['kfmpc'][:]
    flux_power = load['flux_vectors'][:].reshape(-1, zout.size, kfmpc.size)
    load.close()

    load = h5py.File(t0_file, 'r')
    meant = load['meanT'][:]
    load.close()

    # remove unwanted redshifts for flux_power and meant
    z_rng = (zout <= max_z)*(zout >= min_z)
    nz = np.sum(z_rng)
    meant = meant[:, z_rng]
    flux_power = flux_power[:, z_rng].reshape(-1, nz*kfmpc.size)

    with open(json_file, 'r') as jsin:
        param_limits = np.array(json.load(jsin)['param_limits'])

    # loop over all combinations
    all_combinations = list(combinations(range(params.shape[0]), num_selected))
    all_fps_loss = []
    all_t0_loss = []
    for j, selind in enumerate(all_combinations):
        # get the two emulators (trained using the selind simulations
        # to predict the ~selind simulations)
        fps_emu = gpemu.MultiBinGP(params=params[selind, :], kf=kfmpc, powers=flux_power[selind, :], param_limits=param_limits)
        t0_emu = t0gpemu.T0MultiBinGP(params=params[selind, :], temps=meant[selind, :], param_limits=param_limits)

        # make predictions for the rest of the simulations
        unselind = np.setdiff1d(np.arange(params.shape[0]), selind)
        fps_preds = np.array([fps_emu.predict(params[unselind[i]].reshape(1, -1)) for i in range(unselind.size)]).reshape(unselind.size, 2, kfmpc.size*nz)
        t0_preds = np.array([t0_emu.predict(params[unselind[i]].reshape(1, -1)) for i in range(unselind.size)])

        # compare to true values and compute loss
        fps_loss = np.mean((fps_preds[:, 0] - flux_power[unselind])**2/flux_power[unselind]**2)
        t0_loss = np.mean((t0_preds[:, 0] - meant[unselind])**2/meant[unselind]**2)

        # save all losses
        all_fps_loss.append(fps_loss)
        all_t0_loss.append(t0_loss)

    return all_fps_loss, all_t0_loss, all_combinations


def search_next(fps_file, t0_file, json_file, prev_ind, max_z=5.4, min_z=2.0):
    load = h5py.File(fps_file, 'r')
    zout = np.round(load['zout'][:], 1)
    params = load['params'][:]
    kfmpc = load['kfmpc'][:]
    flux_power = load['flux_vectors'][:].reshape(-1, zout.size, kfmpc.size)
    load.close()

    load = h5py.File(t0_file, 'r')
    meant = load['meanT'][:]
    load.close()

    # remove unwanted redshifts for flux_power and meant
    z_rng = (zout <= max_z)*(zout >= min_z)
    nz = np.sum(z_rng)
    meant = meant[:, z_rng]
    flux_power = flux_power[:, z_rng].reshape(-1, nz*kfmpc.size)

    with open(json_file, 'r') as jsin:
        param_limits = np.array(json.load(jsin)['param_limits'])

    # loop over all combinations
    ind_rng = np.setdiff1d(np.arange(params.shape[0]), prev_ind)
    all_combinations = list((*prev_ind, ind_rng[i]) for i in range(ind_rng.size))
    all_fps_loss = []
    all_t0_loss = []
    for j, selind in enumerate(all_combinations):
        # get the two emulators (trained using the selind simulations
        # to predict the ~selind simulations)
        fps_emu = gpemu.MultiBinGP(params=params[selind, :], kf=kfmpc, powers=flux_power[selind, :], param_limits=param_limits, zout=zout[z_rng])
        t0_emu = t0gpemu.T0MultiBinGP(params=params[selind, :], temps=meant[selind, :], param_limits=param_limits)

        # make predictions for the rest of the simulations
        unselind = np.setdiff1d(np.arange(params.shape[0]), selind)
        fps_preds = np.array([fps_emu.predict(params[unselind[i]].reshape(1, -1)) for i in range(unselind.size)]).reshape(unselind.size, 2, kfmpc.size*nz)
        t0_preds = np.array([t0_emu.predict(params[unselind[i]].reshape(1, -1)) for i in range(unselind.size)])

        # compare to true values and compute loss
        fps_loss = np.mean((fps_preds[:,0] - flux_power[unselind])**2/flux_power[unselind]**2)
        t0_loss = np.mean((t0_preds[:,0] - meant[unselind])**2/meant[unselind]**2)

        # save all losses
        all_fps_loss.append(fps_loss)
        all_t0_loss.append(t0_loss)

    return all_fps_loss, all_t0_loss, all_combinations
