"""Modules to get the matter power spectrum from a simulation box and build a simple test emulator."""
import os.path
import math
import numpy as np
import scipy.interpolate
from .coarse_grid import Emulator

class MatterPowerEmulator(Emulator):
    """Build an emulator based on the matter power spectrum instead of the flux power spectrum, for testing."""
    def load(self,dumpfile="emulator_params.json"):
        """Load parameters from a textfile. Reset the k values to something sensible for matter power."""
        super().load(dumpfile=dumpfile)
        self.kf = np.logspace(np.log10(3*math.pi/60.),np.log10(2*math.pi/60.*256),20)

    def _get_fv(self, pp,myspec):
        """Helper function to get a single matter power vector."""
        di = self.get_outdir(pp)
        (_,_) = myspec
        fv = get_matter_power(di,kk=self.kf, redshift = 3.)
        return fv

def get_matter_power(base, kk, redshift = 3.):
    """Gets the matter power spectrum at a single redshift, rebinned onto the given k."""
    for snap in range(1000):
        snapdir = os.path.join(base,"powerspec_"+str(snap).rjust(3,'0')+".txt")
        if not os.path.exists(snapdir):
            #We ran out of snapshots
            print(snapdir)
            break
        (time, kk_sim, pk_sim) = get_folded_power(snapdir)
        assert len(kk_sim) > len(kk)
        if np.abs(1/time - 1 - redshift) < 0.01:
            #Rebin flux power to have desired k bins
            rebinned=scipy.interpolate.interpolate.interp1d(kk_sim,pk_sim)
            pk = rebinned(kk)
            return pk
    raise IOError("No power spectra found")

def get_folded_power(fname1):
    """Get the matter power spectrum from the internal Gadget estimator"""
    (time, kk_a1,pk_a1,kk_b1,pk_b1)=loadfolded(fname1)
    ind = np.where(kk_a1 > kk_b1[-1])
    kk_aa1 = np.ravel(kk_a1[ind])
    pk_aa = np.ravel(pk_a1[ind])/kk_a1[ind]**3
    pk_b1 = pk_b1/kk_b1**3
    return (time, np.concatenate([kk_b1,kk_aa1]), np.concatenate([pk_b1, pk_aa]))

def loadfolded(fname):
    """Load the folded power spectrum file"""
    f_in= np.fromfile(fname, sep=' ',count=-1)
    #Load header
    scale=1000
    time=f_in[0]
    bins_a=int(f_in[1])
    #Read large scale power spectrum data
    adata=f_in[4:(10*bins_a+4)].reshape(bins_a,10)
    #Read second header
    b_off=10*bins_a+4
    time=f_in[b_off]
    bins_b=int(f_in[b_off+1])
    #Read small-scale data
#     print os.path.basename(fname)+' at time '+str(round((1./time-1.),2))
    bdata=f_in[(b_off+4):(10*bins_b+4+b_off)].reshape(bins_b,10)
    (kk_a, pk_a) = GetFoldedPower(adata,bins_a)
    (kk_b, pk_b) = GetFoldedPower(bdata,bins_b)
    #Ignore the sample variance dominated modes near the edge of the small-scale bins.
    if kk_a[0] > kk_b[0]:
        ind=np.where(kk_a > 4*kk_a[0])
        kk_a=kk_a[ind]
        pk_a=pk_a[ind]
    else:
        ind=np.where(kk_b > 4*kk_b[0])
        kk_b=kk_b[ind]
        pk_b=pk_b[ind]
    return (time, scale*kk_a, pk_a, scale*kk_b, pk_b)

def GetFoldedPower(adata, bins):
    """Returns the dimensionless Delta parameter"""
    #Set up variables
    # k
    K_A = adata[:,0]
    #Number of modes in a bin
    ModeCount_A = adata[:,4]
    ModePowUncorrected_A = adata[:,6]
    # This is a volume conversion factor# 4 M_PI : [k/[2M_PI:Box]]::3
    ConvFac_A =  adata[:,9]
    MinModeCount = 50
    TargetBins = 200
    assert np.all(K_A) > 0
    logK_A=np.log10(K_A)
    MinDlogK = (np.max(logK_A) - np.min(logK_A))/TargetBins
    istart=0
    iend=0
    k_list_A = []#np.array([])
    Pk_list = []#np.array([])
#     count_list_A =[]# np.array([])
    count=0
    targetlogK=MinDlogK+logK_A[istart]
    while iend < bins:
        count+=ModeCount_A[iend]
        iend+=1
        if count >= MinModeCount and logK_A[iend-1] >= targetlogK:
            pk = np.sum(ModeCount_A[istart:iend]*ModePowUncorrected_A[istart:iend]*ConvFac_A[istart:iend])/count
            kk = np.sum(ModeCount_A[istart:iend]*K_A[istart:iend])/count
            #Earlier versions did: (ConvFac_A[b]*Specshape_A[b])
            #This is a correction from what is done in pm_periodic.
            #I think it is some sort of bin weighted average.
            #In pm_periodic he divides each mode by Specshape_A(k_mode)
            #, and then sums them. So we can use the Corrected powers and multiply by Specshape,
            #or we can just use the uncorrected versions. They give the same answer.
            k_list_A.append(kk)
            Pk_list.append(pk)
#             count_list_A.append(count)
            istart=iend
            targetlogK=logK_A[istart]+MinDlogK
            count=0
            assert pk >= 0
    return (np.array(k_list_A), np.array(Pk_list))
