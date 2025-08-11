from distutils.core import setup

setup(
    name="lyaemu",
    version='1.0.0',
    author="Simeon Bird, Martin Fernandez and Keir Rogers",
    author_email="spb@ucr.edu",
    license = "mit",
    url="http://github.com/sbird/lya_emulator",
    description="Module for easily generating emulators for the lyman alpha forest from simulations",
    long_description="Module for easily generating emulators for the lyman alpha forest from simulations",
    long_description_content_type = "text/plain",
    packages = ['lyaemu', 'lyaemu.tests', 'lyaemu.SimulationRunner.SimulationRunner', 'lyaemu.meanT'],
    requires=['numpy', 'pandas', 'fake_spectra','scipy', "GPy", "cobaya", "h5py"],
    package_data = {
            'lyaemu': ['data/boss_dr*_data/*.dat','data/desi*/*.txt', 'data/kodiaq_squad/*.txt','data/xq100/*.csv','data/*', 'data/Gaikwad/Gaikwad_2020b_T0_Evolution_All_Statistics.txt'],
           },
    classifiers = ["Development Status :: 4 - Beta",
                   "Intended Audience :: Developers",
                   "Intended Audience :: Science/Research",
                   "Programming Language :: Python :: 3",
                   "Topic :: Scientific/Engineering :: Astronomy",
                   "Topic :: Scientific/Engineering :: Visualization"]
)
