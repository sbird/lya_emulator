# Multi-fidelity emulator sub-module

Files related to multi-fidelity emulators are written as a sub-module in `lyaemu/mf_emulator/` folder.

* Emulator main classes
  - `lyaemu.mf_emulator.gpemulator_singlebin`: Files moved from jibanCat:matter_multi_fidelity_emu
    - Class `SingleBinGP`: A GPRegression models GP on each k bin of powerspecs
    - Class `SingleBinLinearGP`, AR1: A thin wrapper around `GPy.core.GP` that does some input checking and provides
      a default likelihood. Also, model each k bin as an independent GP.
    - Class `SingleBinNonLinearGP`, NARGP:  NARGP model based on kernels from `make_non_linear_kernels`. It models each k input as an independent GP.
    - Class `SingleBinDeepGP`: A thin wrapper around MultiFidelityDeepGP.
  - `lyaemu.mf_emulator.mfemulator.MFEmulator`: A class handles multi-fidelity emulators. A thin wrapper around `lyaemu.coarse_grid.Emulator`.
- Helper functions and classes
  - `lyaemu.mf_emulator.dataloader`
    - Function `convert_h5_to_txt`: Convert the h5 files Martin gave me to txt files to be read by the dataloader.
    - Class `FluxVectors`: A data loader to load multi-fidelity training and test data.
- Procedures to optimize high-fidelity training points given low-fidelity data
  - `lyaemu.mf_emulator.hf_optimizes`: Procedures to optimize the hf training set.
  - `lyaemu.mf_emulator.mf_trainset_optimize`: A class to train low-fidelity only emulators and test on low-fidelity simulations.
  - `lyaemu.mf_emulator.test_trainset`: A function to add testing simulations from the h5 file (provided by Martin) to the high-fidelity training set and store all relevant training/testing files to a folder.

## Example scripts

- `examples.optimize_mfemulator`: Function `optimize_mfemulator` outlines the procedures to train multi-fidelity emulators (both AR1 and NARGP) based on the h5 and json files Martin provided. All the relevant results will be saved to `data/output/` folder.
- `examples.optimize_hf_trainset`: Functions to select the optimal HF samples, and validate the selection.

## How to use it

### Optimize multi-fidelity emulators

Run two-fidelity emulator for 2 <= z <= 4:

```bash
python -c 'from examples.optimize_mfemulator import *; optimize_mfemulator(\
    lf_filename = "data/emu_30mpc_lores/cc_emulator_flux_vectors_tau1000000.hdf5",\
    hf_filename = "data/emu_30mpc_hires/cc_emulator_flux_vectors_tau1000000.hdf5",\
    test_filename = "data/emu_30mpc_test/cc_emulator_flux_vectors_tau1000000.hdf5",\
    lf_json = "data/emu_30mpc_lores/emulator_params.json",\
    hf_json = "data/emu_30mpc_hires/emulator_params.json",\
    test_json = "data/emu_30mpc_test/emulator_params.json",\
    n_optimization_restarts = 20,\
    n_fidelities = 2,\
    max_z = 5.4,\
    min_z = 2.0,\
)'
```

The above code will take low-fidelity data from `data/emu_30mpc_lores/` and high-fidelity data from  `data/emu_30mpc_hires/` to generate folders containing training data in `.txt` formats. Each folder contains training data for separate redshifts. These folders will be saved at `data/processed/`.

And then, this code will train emulators using `lyaemu.mf_emulator.gpemulator_singlebin` module. The following emulators will be trained:
1. AR1 (`SingleBinLinearGP`).
2. NARGP (`SingleBinNonLinearGP`).
3. Single-fidelity emulator (`SingleBinGP`).

The trained emulators be used to predict the test data in `data/emu_30mpc_test/` and calculate the emulation accuracy.

The optimized hyperparameters and prediction plots will be saved to `data/output/` folder. Each folder contains the emulation result for a given redshift.


### Optimize high-fidelity training set given only low-fidelity data

* To direct search 3 optimal points based on given low-fidelity data,

```bash
python -c "from examples.optimize_hf_trainset import *;\
hf_optimize(\
num_selected=3,\
filename='data/emu_30mpc_lores/cc_emulator_flux_vectors_tau1000000.hdf5',\
json_name='data/emu_30mpc_lores/emulator_params.json',\
selected_index=None,\
n_optimization_restarts=10)"
```

Output files will be saved to `output/hf_optimals/` with corresponding folder names.

* To condition on previously selected 3 optimal points and search for the 4th point,


```bash
python -c "from examples.optimize_hf_trainset import *;\
hf_optimize(\
num_selected=4,\
filename='data/emu_30mpc_lores/cc_emulator_flux_vectors_tau1000000.hdf5',\
json_name='data/emu_30mpc_lores/emulator_params.json',\
selected_index=[1,17,18],\
n_optimization_restarts=10)"
```

In the above case, the previously selected optimal 3 points are [1, 17, 18].

Note that the selected index might differ for different computers due to the random seed. However, we can always validate whether this selection is optimal by comparing the results with the direct search. For example, we can compare the low-fidelity emulation accuracy of the optimal choice with all possible combinations. The above example includes validating the selection on all possible combinations.

## Requirements

- `python3+`
- `numpy`
- `scipy`
- `matplotlib`
- `GPy`
- `emukit`
