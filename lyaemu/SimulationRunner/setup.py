"""Setup a python module for SimulationRunner"""
from setuptools import setup

setup(
    name="SimulationRunner",
    version='1.0.2',
    author="Simeon Bird",
    author_email="sbird@ucr.edu",
    #Use the subclass which adds openmp flags as appropriate
    url="http://github.com/sbird/SimulationRunner",
    description="Python script for generating Gadget simulation parameter files",
    packages = ['SimulationRunner'],
    requires=['numpy', 'h5py','scipy'],
    license='MIT',
    package_data = {'SimulationRunner': ['*.ini','*.param', '*.dat'],},
    classifiers = ["Development Status :: 4 - Beta",
                   "Intended Audience :: Developers",
                   "Intended Audience :: Science/Research",
                   "Programming Language :: Python :: 3",
                   "Topic :: Scientific/Engineering :: Astronomy",
                   "Topic :: Scientific/Engineering :: Visualization"]
)
