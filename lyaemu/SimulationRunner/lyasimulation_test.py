"""Tests for the lyman alpha simulation runner."""
import scipy.interpolate as interp
import numpy as np
from SimulationRunner import lyasimulation

def build_restrict_interp(power, lower, upper):
    """Build an interpolator for a restricted range of x values"""
    index = np.searchsorted(power[:,0], [lower,upper])
    (imin, imax) = (np.max([0,index[0]-5]), np.min([len(power[:,0])-1,index[-1]+5]))
    newint = interp.interp1d(np.log(power[imin:imax:,0]), np.log(power[imin:imax,1]), kind='cubic')
    return newint

def check_change_power_spectrum(test_knotpos, test_knotval, matpow):
    """Test that multiplying the power spectrum by some knots gives an accurate answer."""
    #Get the modified power spectrum
    kval = matpow[:,0]
    newpk = lyasimulation.change_power_spectrum_knots(test_knotpos, test_knotval, matpow)
    #Check the kvalues are still the same for comparison to the transfer function
    assert np.all([k in newpk[:,0] for k in kval])
    #Build interpolators for the new power spectrum
    #Only interpolate a subset of Pk for speed
    newpkint = build_restrict_interp(newpk, test_knotpos[0]/3., test_knotpos[-1]*3)
    #Build interpolator for old power spectrum
    pkint = build_restrict_interp(matpow, test_knotpos[0]/3., test_knotpos[-1]*3)
    #Build interpolator for knots
    ext_knotpos = np.concatenate([[kval[0],],test_knotpos, [kval[-1],]])
    ext_knotval = np.concatenate([[test_knotval[0],],test_knotval, [test_knotval[-1],]])
    knotint = interp.interp1d(ext_knotpos, ext_knotval, kind='linear')
    #Check that the interpolator works
    assert np.all(np.abs(knotint(test_knotpos) / test_knotval-1) < 1e-5)
    lg_knotpos = np.log(test_knotpos)
    #Check modification worked at the knot values
    assert np.all(np.abs(np.exp(newpkint(lg_knotpos)) / (np.exp(pkint(lg_knotpos)) * test_knotval) - 1) < 1e-3)
    #Pick some random k values distributed uniformly in log space
    krand = (lg_knotpos[-1]-lg_knotpos[0]+0.2)*np.random.random(250)+lg_knotpos[0]-0.1
    #Check that the modification was accurate at random positions
    #print(np.max(np.abs(np.exp(newpkint(krand)) / (np.exp(pkint(krand)) * knotint(np.exp(krand))) - 1)))
    assert np.all(np.abs(np.exp(newpkint(krand)) / (np.exp(pkint(krand)) * knotint(np.exp(krand))) - 1) < 0.01)


def test_change_power_spectrum():
    """Perform the power spectrum check for a number of different knot values and positions"""
    #The 2010 paper had the knots at:
    #k = 0.475 0.75 1.19, 1.89
    #(knotpos, knotval)
    tests = [(np.array([0.475, 0.75, 1.19, 1.89]), np.array([0.475, 0.75, 1.19, 1.89])),
             (np.array([0.475, 0.75, 1.19, 1.89]), np.array([1.2, 1., 1., 1.])),
             (np.array([0.475, 0.75, 1.19, 1.89]), np.array([1.2, 0.5, 1.2, 0.5])),
             (np.array([0.05, 0.1, 10]), np.array([1.3, 0.3, 1.1]))]
    matpow = np.loadtxt("testdata/ics_matterpow_99.dat")
    #Copy array so that we don't get changed in-place
    [check_change_power_spectrum(kp, kv, matpow) for (kp, kv) in tests]
