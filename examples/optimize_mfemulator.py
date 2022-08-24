"""
Build multi-fidelity emulators and optimize them.

From h5 and json files to generate training folders,
from training folders to build emulators,
from emulators to output testing  results.
"""
from typing import List

import os, json

import numpy as np
import h5py
import matplotlib
from matplotlib import pyplot as plt

from lyaemu.mf_emulator.dataloader import convert_h5_to_txt, FluxVectors, folder_name
from lyaemu.mf_emulator.gpemulator_singlebin import (
    SingleBinLinearGP,
    SingleBinNonLinearGP,
    SingleBinGP,
)

# set a random number seed to reproducibility
np.random.seed(0)

matplotlib.use("pdf")

save_figure = lambda filename: plt.savefig(
    "{}.pdf".format(filename), format="pdf", dpi=300
)


def optimize_mfemulator(
    lf_filename: str = "data/emu_30mpc_lores/cc_emulator_flux_vectors_tau1000000.hdf5",
    hf_filename: str = "data/emu_30mpc_hires/cc_emulator_flux_vectors_tau1000000.hdf5",
    test_filename: str = "data/emu_30mpc_test/cc_emulator_flux_vectors_tau1000000.hdf5",
    lf_json: str = "data/emu_30mpc_lores/emulator_params.json",
    hf_json: str = "data/emu_30mpc_hires/emulator_params.json",
    test_json: str = "data/emu_30mpc_test/emulator_params.json",
    n_optimization_restarts: int = 20,
    n_fidelities: int = 2,  # NOTE: only support 2 fidelities now.
    turn_off_bias_nargp: bool = False,
    ARD_last_fidelity: bool = False,
    max_z: float = 5.4,
    min_z: float = 2.0,
):
    """
    A function runs through all necessary procedures for training/testing the MF emulator:
    1. Convert h5,json files into individual folders contain training/testing data for
        each redshift.
    2. Train on emulators on separate redshifts.
    3. Output the training and testing results into separate folders, naming based on the
        redshifts. 

    Parameters:
    ----
    n_optimization_restarts: number of optimization you want to repeat. The GPy will
        choose the best hyperparameters among those repetitions. More is better.
    n_fidelities: only supports 2 now. You may try a larger number but some tweaks might
        be needed.
    turn_off_bias_nargp: not adding bias kernel for NARGP in high-fidelity. In case you
        find the optimization result is not stable, try turning off bias kernel. Some time
        the training data at high-fidelity is not enough to train the bias kernel and
        induce some unstable predictions.
    ARD_last_fidelity: whether to apply ARD for the last (highest) fidelity.
        Default, False.
    """

    # Convert h5,json files Martin gave me into training folders for MFEmu
    # NOTE: Nameing of the training folder follows lyaemu.mf_emulator.dataloader.folder_name
    convert_h5_to_txt(
        lf_filename=lf_filename,
        hf_filename=hf_filename,
        test_filename=test_filename,
        lf_json=lf_json,
        hf_json=hf_json,
        test_json=test_json,
    )

    # extract the boxsize and res
    with open(lf_json, "r") as param:
        param_lf = json.load(param)
    with open(hf_json, "r") as param:
        param_hf = json.load(param)
    # extract available zs
    f_lf = h5py.File(lf_filename, "r")
    zout = f_lf["zout"][()]

    # The training data follows this naming (redshift separated)
    # Now we just read them separately.
    this_folder_fn = lambda z: folder_name(
        len(param_lf["sample_params"]),
        param_lf["npart"],
        param_lf["box"],
        len(param_hf["sample_params"]),
        param_hf["npart"],
        param_hf["box"],
        z,
    )
    train_folder_fn = lambda z: os.path.join("data", "processed", this_folder_fn(z),)
    output_folder_fn = lambda z: os.path.join("data", "output", this_folder_fn(z),)

    # run through the desired redshift bins
    ind = (min_z <= zout) & (zout <= max_z)
    print("Running through z = ", zout[ind])

    for z in zout[ind]:
        print("Processing z = ", z, "...")
        do_validations(
            folder=train_folder_fn(z),
            output_folder=output_folder_fn(z),
            n_optimization_restarts=n_optimization_restarts,
            n_fidelities=n_fidelities,
            turn_off_bias_nargp=turn_off_bias_nargp,
            ARD_last_fidelity=ARD_last_fidelity,
        )
        print("... done.")


def do_validations(
    folder: str = "data/processed/Lyaflux_30_res256box30_3_res512box30_2",
    output_folder: str = "data/output/Lyaflux_30_res256box30_3_res512box30_2",
    n_optimization_restarts: int = 20,
    n_fidelities: int = 2,
    turn_off_bias_nargp: bool = False,
    ARD_last_fidelity: bool = False,
):
    """
    Train and test models (for separate redshift), and plot
    1. predicted / exact power spectrum
    2. absolute error plot
    3. parameter plots

    Only support 2 fidelities now.

    Parameters:
    ----
    folder: the folder contains the the training and testing data. See data/50_LR_3_HR
        for example.
    n_optimization_restarts: number of optimization you want to repeat. The GPy will
        choose the best hyperparameters among those repetitions. More is better.
    n_fidelities: only supports 2 now. You may try a larger number but some tweaks might
        be needed.
    turn_off_bias_nargp: not adding bias kernel for NARGP in high-fidelity. In case you
        find the optimization result is not stable, try turning off bias kernel. Some time
        the training data at high-fidelity is not enough to train the bias kernel and
        induce some unstable predictions.
    ARD_last_fidelity: whether to apply ARD for the last (highest) fidelity.
        Default, False.
    """
    # create output folder, recursively
    os.makedirs(output_folder, exist_ok=True)
    old_dir = os.getcwd()
    print("Current path:", old_dir)

    # get training and testing data. Normalization included.
    data = FluxVectors(n_fidelities=n_fidelities,)
    data.read_from_txt(folder=folder)

    # change path for saving figures
    os.chdir(output_folder)
    print(">> ", os.getcwd())

    # Plot the parameters
    plot_parameters(X_train=data.X_train, X_test=data.X_test, )

    # Multi-fidelity
    # linear multi-fidelity
    ar1 = SingleBinLinearGP(
        data.X_train_norm,
        data.Y_train_norm,
        kernel_list=None,
        n_fidelities=n_fidelities,
        ARD_last_fidelity=ARD_last_fidelity,
    )
    # non-linear multi-fidelity
    nargp = SingleBinNonLinearGP(
        data.X_train_norm,
        data.Y_train_norm,
        n_fidelities=n_fidelities,
        n_samples=500,
        optimization_restarts=n_optimization_restarts,
        turn_off_bias=turn_off_bias_nargp,
        ARD_last_fidelity=ARD_last_fidelity,
    )

    # Single-fidelity
    # high-fidelity only emulator
    hf_only = SingleBinGP(data.X_train_norm[-1], data.Y_train_norm[-1])
    lf_only = SingleBinGP(data.X_train_norm[0], data.Y_train_norm[0])

    # optimize each model
    ar1.optimize(n_optimization_restarts=n_optimization_restarts)
    nargp.optimize()
    hf_only.optimize_restarts(n_optimization_restarts=n_optimization_restarts)
    lf_only.optimize_restarts(n_optimization_restarts=n_optimization_restarts)

    # testing set
    means_ar1, vars_ar1, pred_exacts_ar1 = validate_mf(data, model=ar1)
    means_nargp, vars_nargp, pred_exacts_nargp = validate_mf(data, model=nargp)
    means_hfonly, vars_hfonly, pred_exacts_hfonly = validate_sf(data, model=hf_only)
    means_lfonly, vars_lfonly, pred_exacts_lfonly = validate_sf(data, model=lf_only)

    # versus HF
    do_emulator_error_plots(
        data,
        means_ar1,
        means_hfonly,
        pred_exacts_ar1,
        pred_exacts_hfonly,
        label_mf="AR1",
        label_sf="HF only",
        figure_name="ar1",
    )
    do_emulator_error_plots(
        data,
        means_nargp,
        means_hfonly,
        pred_exacts_nargp,
        pred_exacts_hfonly,
        label_mf="NARGP",
        label_sf="HF only",
        figure_name="nargp",
    )
    # versus LF
    do_emulator_error_plots(
        data,
        means_ar1,
        means_lfonly,
        pred_exacts_ar1,
        pred_exacts_lfonly,
        label_mf="AR1",
        label_sf="LF only",
        figure_name="ar1_lf",
    )
    do_emulator_error_plots(
        data,
        means_nargp,
        means_lfonly,
        pred_exacts_nargp,
        pred_exacts_lfonly,
        label_mf="NARGP",
        label_sf="LF only",
        figure_name="nargp_lf",
    )

    # pred/exact plot
    do_pred_exact(data, means_ar1, pred_exacts_ar1, label_mf="AR1", figure_name="ar1")
    do_pred_exact(
        data, means_nargp, pred_exacts_nargp, label_mf="NARGP", figure_name="nargp"
    )

    # saving hyperparameters
    with open("ar1.json", "w") as f:
        json.dump(ar1.to_dict(), f, indent=2)

    with open("nargp.json", "w") as f:
        json.dump(nargp.to_dict(), f, indent=2)

    with open("hf_only.json", "w") as f:
        json.dump(hf_only.to_dict(), f, indent=2)

    with open("lf_only.json", "w") as f:
        json.dump(lf_only.to_dict(), f, indent=2)

    # saving AR1
    os.makedirs("AR1/", exist_ok=True)

    np.savetxt(os.path.join("AR1", "all_gp_mean"), np.array(means_ar1))
    np.savetxt(os.path.join("AR1", "all_gp_var"), np.array(vars_ar1))
    np.savetxt(os.path.join("AR1", "pred_exacts"), np.array(pred_exacts_ar1))
    np.savetxt(os.path.join("AR1", "all_true"), np.array(data.Y_test[0]))
    np.savetxt(os.path.join("AR1", "kf"), np.array(data.kf))
    # [HF] also save the predictions from hf-only
    np.savetxt(os.path.join("AR1", "all_hf_gp_mean"), np.array(means_hfonly))
    np.savetxt(os.path.join("AR1", "all_hf_gp_var"), np.array(vars_hfonly))
    np.savetxt(os.path.join("AR1", "pred_exacts_hf"), np.array(pred_exacts_hfonly))
    # [LF] also save the predictions from lf-only
    np.savetxt(os.path.join("AR1", "all_lf_gp_mean"), np.array(means_lfonly))
    np.savetxt(os.path.join("AR1", "all_lf_gp_var"), np.array(vars_lfonly))
    np.savetxt(os.path.join("AR1", "pred_exacts_lf"), np.array(pred_exacts_lfonly))

    # saving NARGP
    os.makedirs("NARGP/", exist_ok=True)

    np.savetxt(os.path.join("NARGP", "all_gp_mean"), np.array(means_nargp))
    np.savetxt(os.path.join("NARGP", "all_gp_var"), np.array(vars_nargp))
    np.savetxt(os.path.join("NARGP", "pred_exacts"), np.array(pred_exacts_nargp))
    np.savetxt(os.path.join("NARGP", "all_true"), np.array(data.Y_test[0]))
    np.savetxt(os.path.join("NARGP", "kf"), np.array(data.kf))
    # [HF] also save the predictions from hf-only
    np.savetxt(os.path.join("NARGP", "all_hf_gp_mean"), np.array(means_hfonly))
    np.savetxt(os.path.join("NARGP", "all_hf_gp_var"), np.array(vars_hfonly))
    np.savetxt(os.path.join("NARGP", "pred_exacts_hf"), np.array(pred_exacts_hfonly))
    # [LF] also save the predictions from lf-only
    np.savetxt(os.path.join("NARGP", "all_lf_gp_mean"), np.array(means_lfonly))
    np.savetxt(os.path.join("NARGP", "all_lf_gp_var"), np.array(vars_lfonly))
    np.savetxt(os.path.join("NARGP", "pred_exacts_lf"), np.array(pred_exacts_lfonly))

    # back to root folder
    os.chdir(old_dir)


def validate_mf(data: FluxVectors, model: SingleBinNonLinearGP, fidelity: int = 1):
    """
    Validate the trained MFEmulators
    ----

    Parameters:
    ----
    fidelity: the output fidelity. Default fidelity=1, the second fidelity.

    Returns:
    ----
    all_means: predictied means from the GP (in linear scale).
    all_vars: predictied variance from the GP (in linear scale).
    all_pred_exacts: Predicted/Exact (in linear scale).
    """
    all_means = []
    all_vars = []
    all_pred_exacts = []
    for n_validations, (x_test, y_test) in enumerate(
        zip(data.X_test_norm[0], data.Y_test[0])
    ):
        # the last column is the indicator for the output fidelity.
        x_test_index = np.concatenate(
            (x_test[None, :], np.ones((1, 1)) * fidelity), axis=1
        )
        flux_predict, var = model.predict(x_test_index)

        flux_predict = flux_predict[0]
        var = var[0]

        mean = (flux_predict + 1) * data.scalefactors
        std = np.sqrt(var) * data.scalefactors

        # save variance
        var = std**2

        all_means.append(mean)
        all_vars.append(var)

        # predicted/exact
        all_pred_exacts.append(mean / y_test)

    return all_means, all_vars, all_pred_exacts


def validate_sf(data: FluxVectors, model: SingleBinGP):
    """
    Validate the trained single-fidelity emulator

    Returns:
    ----
    all_means: predictied means from the GP (in linear scale).
    all_vars: predictied variance from the GP (in linear scale).
    all_pred_exacts: Predicted/Exact (in linear scale).
    """
    all_means = []
    all_vars = []
    all_pred_exacts = []
    for n_validations, (x_test, y_test) in enumerate(
        zip(data.X_test_norm[0], data.Y_test[0])
    ):
        flux_predict, var = model.predict(x_test[None, :])

        flux_predict = flux_predict[0]
        var = var[0]

        mean = (flux_predict + 1) * data.scalefactors
        std = np.sqrt(var) * data.scalefactors

        # save variance
        var = std**2

        all_means.append(mean)
        all_vars.append(var)

        # predicted/exact
        all_pred_exacts.append(mean / y_test)

    return all_means, all_vars, all_pred_exacts


def do_emulator_error_plots(
    data: FluxVectors,
    means_mf: List[np.ndarray],
    means_sf: List[np.ndarray],
    pred_exacts_mf: List[np.ndarray],
    pred_exacts_sf: List[np.ndarray],
    label_mf: str = "NARGP",
    label_sf: str = "HF only",
    figure_name: str = "",
):
    """
    1. predicted / exact power spectrum
    2. absolute error plot
    """

    # mean emulation error
    emulator_errors = np.abs(np.array(pred_exacts_mf) - 1)
    plt.loglog(
        10 ** data.kf, np.mean(emulator_errors, axis=0), label=label_mf, color="C0"
    )
    plt.fill_between(
        10 ** data.kf,
        y1=np.min(emulator_errors, axis=0),
        y2=np.max(emulator_errors, axis=0),
        color="C0",
        alpha=0.3,
    )

    emulator_errors = np.abs(np.array(pred_exacts_sf) - 1)
    plt.loglog(
        10 ** data.kf, np.mean(emulator_errors, axis=0), label=label_sf, color="C1"
    )
    plt.fill_between(
        10 ** data.kf,
        y1=np.min(emulator_errors, axis=0),
        y2=np.max(emulator_errors, axis=0),
        color="C1",
        alpha=0.3,
    )
    plt.legend()
    plt.ylabel(r"$| P_\mathrm{predicted}(k) / P_\mathrm{true}(k) - 1|$")
    plt.xlabel(r"$k (h/\mathrm{Mpc})$")
    save_figure("absolute_errors_" + figure_name)
    plt.close()
    plt.clf()


def do_pred_exact(
    data: FluxVectors,
    means_mf: List[np.ndarray],
    pred_exacts_mf: List[np.ndarray],
    label_mf: str = "NARGP",
    figure_name: str = "",
):
    """
    Pred/Exact plot
    """
    for i, pred_exact_mf in enumerate(pred_exacts_mf):
        if i == 0:
            plt.semilogx(
                10 ** data.kf, pred_exact_mf, label=label_mf, color="C{}".format(i)
            )
        else:
            plt.semilogx(10 ** data.kf, pred_exact_mf, color="C{}".format(i))

    plt.legend()
    plt.ylim(0.96, 1.06)
    plt.xlabel(r"$k (h/\mathrm{Mpc})$")
    plt.ylabel(r"$\mathrm{Predicted/Exact}$")
    save_figure("predict_exact_" + figure_name)
    plt.close()
    plt.clf()


def plot_parameters(
    X_train: List[np.ndarray],
    X_test: List[np.ndarray],
    parameter_names: List[str] = [
        r"$n_s$",
        r"$A_p$",
        r"$\mathrm{He}_{re,i}$",
        r"$\mathrm{HE}_{re,f}$",
        r"$\alpha_q$",
        r"$h$",
        r"$\Omega_{m} h^2$",
        r"$\mathrm{H}_{re,z}$",
        r"BH Feeback",
    ],
):
    """
    Plot the selected samples with all other samples in the input data.
    This would enable us to investigate locations of the selected training samples.
    """
    n_parameters = X_train[0].shape[1]

    for i in range(n_parameters):
        for j in range(i + 1, n_parameters):
            plt.scatter(
                X_train[0][:, i],
                X_train[0][:, j],
                marker="o",
                label="LowRes training data",
                color="C0",
                s=100,
            )
            plt.scatter(
                X_train[1][:, i],
                X_train[1][:, j],
                marker="o",
                label="HighRes training data",
                color="C1",
                s=40,
            )
            plt.scatter(
                X_test[0][:, i],
                X_test[0][:, j],
                marker="x",
                label="Test spectra",
                color="C2",
                s=100,
            )
            plt.legend()
            plt.xlabel(parameter_names[i])
            plt.ylabel(parameter_names[j])

            save_figure("nested_" + parameter_names[i] + parameter_names[j])
            plt.close()
            plt.clf()
