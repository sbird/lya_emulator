#!/bin/bash
#Script to fetch and build dependencies for the tests, for travis.

wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b -p $HOME/miniconda
export PATH=$HOME/miniconda/bin:$PATH
conda update --yes conda
conda create --yes -n test python=3.6
source $HOME/miniconda/bin/activate test
conda install --yes -c bccp nbodykit matplotlib numpy bigfile
conda install --yes nose configobj scipy
conda install --yes gsl gcc_linux-64
cd $HOME/miniconda/envs/test/bin/
cd -
#Clone stuff.
mkdir tests

mkdir $HOME/codes/
cd $HOME/codes/
#Get and make Gadget.
git clone https://github.com/sbird/MP-Gadget
cd -
cd $HOME/codes/MP-Gadget
./bootstrap.sh
cp Options.mk.example Options.mk
make -j
cd -
