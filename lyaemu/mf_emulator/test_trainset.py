"""
Functions to modify the training/testing sets to
see changes on the emulation results.
"""

import json, os
from typing import List

import h5py
import numpy as np


# put the folder name on top, easier to find and modify
def folder_name(ind_test_to_train: List, num1: int, res1: int, box1: int, num2: int, res2:int, box2: int, z: float):
    return "Test2Train{}_{}_res{}box{}_{}_res{}box{}_{}".format(
        "-".join(map(str, ind_test_to_train)),  num1, res1, box1, num2, res2, box2, "{:.2g}".format(z).replace(".", "_")
    )


def convert_h5_to_txt_testset_to_trainset(
        lf_filename: str = "data/emu_smallbox_lowres/cc_emulator_flux_vectors_tau1000000.hdf5",
        hf_filename: str = "data/emu_smallbox_highres/cc_emulator_flux_vectors_tau1000000.hdf5",
        test_filename: str = "data/emu_testset/cc_emulator_flux_vectors_tau1000000.hdf5",
        lf_json: str = "data/emu_smallbox_lowres/emulator_params.json",
        hf_json: str = "data/emu_smallbox_highres/emulator_params.json",
        test_json: str = "data/emu_testset/emulator_params.json",
        ind_test_to_train: List = [0,],
    ) -> None:
    """
    Convert the h5 files Martin gave me to txt files to be read by the dataloader,
    and also add some test HF simulations as HF training to test the convergence of NARGP.
    """

    f_lf = h5py.File(lf_filename, "r")
    f_hf = h5py.File(hf_filename, "r")
    f_test = h5py.File(test_filename, "r")

    with open(lf_json, "r") as param:
        param_lf = json.load(param)
    with open(hf_json, "r") as param:
        param_hf = json.load(param)
    with open(test_json, "r") as param:
        param_test = json.load(param)

    param_limits = np.array(param_lf["param_limits"])
    assert np.all(param_limits == np.array(param_hf["param_limits"]))
    assert np.all(param_limits == np.array(param_test["param_limits"]))

    # make sure all keys are in the file
    # TODO: also save kfkms for plotting purpose
    keys = ['flux_vectors', 'kfkms', 'kfmpc', 'params', 'zout']
    for key in keys:
        assert key in f_lf.keys()
        assert key in f_hf.keys()
        assert key in f_test.keys()

    print("Low-fidelity file:")
    print("----")
    print("Resolution:", param_lf["npart"])
    print("Box (Mpc/h):", param_lf["box"])
    print("Shape of redshfits", f_lf["zout"].shape)
    print("Shape of params", f_lf["params"].shape)
    print("Shape of kfkms", f_lf["kfkms"].shape)
    print("Shape of kfmpc", f_lf["kfmpc"].shape)
    print("Shape of flux vectors", f_lf["flux_vectors"].shape)
    print("\n")

    # use kfmpc so all redshifts use the same k bins
    kfmpc = f_lf["kfmpc"][()]
    assert np.all(np.abs(kfmpc - f_hf["kfmpc"][()]) < 1e-10)

    zout = f_lf["zout"][()]
    assert np.all(zout == f_hf["zout"][()])

    # flux power spectra, all redshifts
    flux_vectors_lf = f_lf["flux_vectors"][()]

    # input parameters
    x_train_lf = f_lf["params"][()]

    # read flux power spectrum at ith redshift
    # flux power spectrum | z = ?
    def get_flux_vector_at_z(i: int, flux: np.ndarray) -> np.ndarray:
        return flux[:, i * len(kfmpc) : (i + 1) * len(kfmpc)]

    # some checking
    last_flux_vector = get_flux_vector_at_z(len(zout) - 1, flux_vectors_lf)
    assert len(last_flux_vector) == f_lf["params"].shape[0]
    assert last_flux_vector.shape[1] == len(kfmpc)

    first_flux_vector = get_flux_vector_at_z(0, flux_vectors_lf)
    assert len(first_flux_vector) == f_lf["params"].shape[0]
    assert first_flux_vector.shape[1] == len(kfmpc)


    print("High-fidelity file:")
    print("----")
    print("Resolution:", param_hf["npart"])
    print("Box (Mpc/h):", param_hf["box"])
    print("Shape of redshfits", f_hf["zout"].shape)
    print("Shape of params", f_hf["params"].shape)
    print("Shape of kfkms", f_hf["kfkms"].shape)
    print("Shape of kfmpc", f_hf["kfmpc"].shape)
    print("Shape of flux vectors", f_hf["flux_vectors"].shape)
    print("\n")

    # flux power spectra, all redshifts
    flux_vectors_hf = f_hf["flux_vectors"][()]

    # input parameters
    x_train_hf = f_hf["params"][()]

    # some checking
    last_flux_vector = get_flux_vector_at_z(len(zout) - 1, flux_vectors_hf)
    assert len(last_flux_vector) == f_hf["params"].shape[0]
    assert last_flux_vector.shape[1] == len(kfmpc)

    first_flux_vector = get_flux_vector_at_z(0, flux_vectors_hf)
    assert len(first_flux_vector) == f_hf["params"].shape[0]
    assert first_flux_vector.shape[1] == len(kfmpc)

    # test files: same resolution as high-fidelity
    print("Test file:")
    print("----")
    print("Resolution:", param_test["npart"])
    print("Box (Mpc/h):", param_test["box"])
    print("Shape of redshfits", f_test["zout"].shape)
    print("Shape of params", f_test["params"].shape)
    print("Shape of kfkms", f_test["kfkms"].shape)
    print("Shape of kfmpc", f_test["kfmpc"].shape)
    print("Shape of flux vectors", f_test["flux_vectors"].shape)
    print("\n")

    # flux power spectra, all redshifts
    flux_vectors_test = f_test["flux_vectors"][()]

    # input parameters
    x_train_test = f_test["params"][()]

    # some checking
    last_flux_vector = get_flux_vector_at_z(len(zout) - 1, flux_vectors_test)
    assert len(last_flux_vector) == f_test["params"].shape[0]
    assert last_flux_vector.shape[1] == len(kfmpc)

    first_flux_vector = get_flux_vector_at_z(0, flux_vectors_test)
    assert len(first_flux_vector) == f_test["params"].shape[0]
    assert first_flux_vector.shape[1] == len(kfmpc)

    assert len(ind_test_to_train) < x_train_test.shape[0]
    assert np.max(ind_test_to_train) < x_train_test.shape[0]

    # [test2train] Steps for moving some testing files to trainset
    ind = np.full((x_train_test.shape[0], ), fill_value=False)
    ind[np.array(ind_test_to_train)] = True

    # add test simulations to train
    x_train_hf = np.concatenate([x_train_hf, x_train_test[ind]])
    # remove the added test simulations
    x_train_test = x_train_test[~ind]

    # output training files, one redshift per folder
    for i,z in enumerate(zout):
        print("Preparing training files in {:.3g}".format(z))        

        flux_vector_lf = get_flux_vector_at_z(i, flux_vectors_lf)
        flux_vector_hf = get_flux_vector_at_z(i, flux_vectors_hf)
        flux_vector_test = get_flux_vector_at_z(i, flux_vectors_test)

        # add test simulations to train
        flux_vector_hf = np.concatenate([flux_vector_hf, flux_vector_test[ind]])
        # remove the added test simulations
        flux_vector_test = flux_vector_test[~ind]

        outdir = folder_name(
            ind_test_to_train,
            len(x_train_lf),
            param_lf["npart"],
            param_lf["box"],
            len(x_train_hf),
            param_hf["npart"],
            param_lf["box"],
            z,
        )

        this_outdir = os.path.join(
                "data",
                "processed",
                outdir,
        )
        os.makedirs(
            this_outdir,
            exist_ok=True,
        )

        # only flux power needs a loop
        np.savetxt(os.path.join(this_outdir, "train_output_fidelity_0.txt"), np.log10(flux_vector_lf))
        np.savetxt(os.path.join(this_outdir, "train_output_fidelity_1.txt"), np.log10(flux_vector_hf))
        np.savetxt(os.path.join(this_outdir, "test_output.txt"), np.log10(flux_vector_test))

        np.savetxt(os.path.join(this_outdir, "train_input_fidelity_0.txt"), x_train_lf)
        np.savetxt(os.path.join(this_outdir, "train_input_fidelity_1.txt"), x_train_hf)
        np.savetxt(os.path.join(this_outdir, "test_input.txt"), x_train_test)

        np.savetxt(os.path.join(this_outdir, "input_limits.txt"), param_limits)
        np.savetxt(os.path.join(this_outdir, "kf.txt"), np.log10(kfmpc))

