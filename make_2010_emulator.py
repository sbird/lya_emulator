#!/usr/bin/env python
# vim: set fileencoding=UTF-8 :

"""
Routines specific to the simulations run in 2010.
Create the emulator, and plot it's errors
"""
import quadratic_emulator as qe
import matplotlib.pyplot as plt
import os.path
import numpy as np

def plot_err(simdir, simparams, emulator, om, box, H0):
    """Plot the percent error for a variety of redshift bins"""
    errs = qe.get_err(simdir, simparams, emulator, om, box, H0)
    for ee in errs.values():
        plt.semilogx(emulator[0].sdsskbins, ee, '-')
    plt.xlabel("k (s/km)")
    plt.xlim(emulator[0].sdsskbins[0], emulator[0].sdsskbins[-1])
    plt.xticks([0.0015, 0.003, 0.006, 0.01, 0.015], ["0.0015", "0.003", "0.006", "0.01", "0.015"])
    plt.ylabel("Interpolation error (%)")

def make_2010_emulator(bestfitdir="hr2", bestfitbox=60., sdsskbins=np.array([])):
    """
        Make an emulator for the four cosmological knots in the 2010 simulations,
        for all redshift outputs.
    """
    #Parameters of the 2010 simulations
    #Note how every knot has a different best-fit simulation!
    #As I recall there was a bug with the B knots that made small scales wrong somehow, so I had to use bf2 for the C knot.
    #The other two are just to allow for different box sizes
    basedir = "/home/spb/codes/Lyman-alpha/MinParametricRecon/runs/"
    zzz = { n : 4.2-0.2*n for n in xrange(12) }
    omega0 = 0.2669
    hubble0 = 0.71
    #I checked that these are the same parameters for all 4 knots
    bestfit_params = {'AA':0.94434469,'B': 0.93149282,'C': 0.91868144,'D': 0.9060194}
    quads = {}
    for snap in zzz.keys():
        AA_knot = qe.Knot("AA",basedir, bestfit="boxcorr400",snapnum=snap,zz=zzz[snap], om=omega0, H0=hubble0, bf_params=bestfit_params, box=120.)
        B_knot = qe.Knot("B",basedir, bestfit="best-fit",snapnum=snap,zz=zzz[snap], om=omega0, H0=hubble0, bf_params=bestfit_params, box=60.)
        C_knot = qe.Knot("C",basedir, bestfit="bf2",snapnum=snap,zz=zzz[snap], om=omega0, H0=hubble0, bf_params=bestfit_params, box=60.)
        D_knot = qe.Knot("D",basedir, bestfit="bfD",snapnum=snap,zz=zzz[snap], om=omega0, H0=hubble0, bf_params=bestfit_params, box=48.)
        #Best-fit model to use when not computing differences.
        #There are also hr2a, hr3, hr4 and hr4a directories with flux power spectra in them.
        #I have no memory of what these files are - there are no simulation snapshots.
        emubf = qe.FluxPowSimulation(os.path.join(basedir, bestfitdir), snap, bestfit_params, zz=zzz[snap], om=omega0, box=bestfitbox, H0=hubble0)
        quads[snap] = qe.QuadraticEmulator((AA_knot, B_knot, C_knot, D_knot),emubf,sdsskbins=sdsskbins)
    return quads

if __name__=='__main__':
    basedir = "/home/spb/codes/Lyman-alpha/MinParametricRecon/runs/"
    quads = make_2010_emulator(bestfitdir="bf2")
    params = {'AA':0.94434469,'B': 1.23149282 ,'C': 0.56868144,'D': 0.9060194}
    plot_err(os.path.join(basedir, "B1.2C0.55"),params, quads,0.2669, 60, 0.71)
    plt.figure()
    sdsskbins=np.array([0.00178,0.00224,0.00282,0.00355,0.00447,0.00562,0.00708,0.00891,0.01122,0.01413,0.01778])
    #Why does this not work?
    #quads = make_2010_emulator(bestfitdir="bfD", bestfitbox=48., sdsskbins=sdsskbins)
    quads = make_2010_emulator(bestfitdir="bf2", bestfitbox=60.)
    params = {'AA':0.94434469,'B': 0.93149282,'C': 1.21868144 ,'D': 0.6560194}
    plot_err(os.path.join(basedir, "C1.2D0.65"),params, quads,0.2669, 48, 0.71)
