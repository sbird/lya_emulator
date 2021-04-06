"""Setup a python module for SimulationRunner"""
from distutils.core import setup

setup(
    name="SimulationRunner",
    version='1.0',
    author="Simeon Bird",
    author_email="spb@ias.edu",
    #Use the subclass which adds openmp flags as appropriate
    url="http://github.com/sbird/SimulationRunner",
    description="Python script for generating Gadget simulation parameter files",
    packages = ['SimulationRunner'],
    requires=['numpy', 'h5py','scipy', 'nbodykit', 'camb'],
    package_data = {'SimulationRunner': ['*.ini','*.param'],},
    classifiers = ["Development Status :: 4 - Beta",
                   "Intended Audience :: Developers",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved :: MIT License",
                   "Programming Language :: Python :: 3",
                   "Topic :: Scientific/Engineering :: Astronomy",
                   "Topic :: Scientific/Engineering :: Visualization"]
)
