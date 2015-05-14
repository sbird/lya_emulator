#!/usr/bin/env python
# vim: set fileencoding=UTF-8 :

"""
Secondary Emulators for the flux PDF and the matter power spectrum, all using the same basic quadratic form.
"""
import numpy as np
from quadratic_emulator import EmulatedQuantity
from plot_emulator import PlotEmulatedQuantity
import matplotlib.pyplot as plt
from wheref import wheref
import math
import re
import os.path

class FluxPdf(EmulatedQuantity):
    """The PDF is an instance of the EmulatedQuantity class. Perhaps poor naming"""
    def __init__(self, Snaps=(), Zz=np.array([3.0,2.8,2.6,2.4,2.2,2.0]), sdsskbins=np.arange(0,20),knotpos=np.array([]), om=0.266,H0=0.71,box=48.0,base="",suf="flux-pdf/", ext="_flux_pdf.txt",):
        if base == "":
            base=os.path.expanduser("~/Lyman-alpha/MinParametricRecon/runs/")
        # Snapshots
        if np.size(Snaps) == 0:
            Snaps=("snapshot_006","snapshot_007","snapshot_008","snapshot_009","snapshot_010","snapshot_011")
        else:
            Snaps=Snaps
        EmulatedQuantity.__init__(self, Snaps,Zz,sdsskbins,knotpos, om, H0,box,base,suf,ext)

    def loadpk(self, path, box):
        flux_pdf = np.loadtxt(self.base+path)
        return(flux_pdf[:,0], flux_pdf[:,1])

    def plot_compare_two(self, one, onebox, two,twobox,colour=""):
        """ Compare two power spectra directly. Smooths result.
        plot_compare_two(first P(k), second P(k))"""
        (onek,oneP)=self.loadpk(one,onebox)
        (twok,twoP)=self.loadpk(two,twobox)
        assert np.all(onek == twok)
        relP=oneP/twoP
        plt.title("Relative flux PDF "+one+" and "+two)
        plt.ylabel(r"$F_2(k)/F_1(k)$")
        plt.xlabel(r"$Flux$")
        line=plt.semilogy(onek,relP, color=colour)
        return line

    def plot_power(self,path, redshift, colour="black"):
        """ Plot absolute power spectrum, not relative"""
        (k,Pdf)=self.loadpk(path+self.suf+self.pre+self.GetSnap(redshift)+self.ext,self.box)
        plt.semilogy(k,Pdf, color=colour, linewidth="1.5")
        plt.ylabel("P(k) /(h-3 Mpc3)")
        plt.xlabel("k /(h MPc-1)")
        plt.title("PDF at z="+str(redshift))
        return(k, Pdf)

    def calc_z(self, redshift,s_knot):
        """ Calculate the flux derivatives for a single redshift
                Output: (kbins d2P...kbins dP (flat vector of length 2x21))"""
        #Array to store answers.
        #Format is: k x (dP, d²P, χ²)
        npvals=np.size(s_knot.pvals)
        nk=21
        results=np.zeros(2*nk)
        pdifs=s_knot.pvals-s_knot.p0
        #This is to rescale by the mean flux, for generating mean flux tables.
        ###
        #tau_eff=0.0023*(1+redshift)**3.65
        #tmin=0.2*((1+redshift)/4.)**2
        #tmax=0.5*((1+redshift)/4.)**4
        #teffs=tmin+s_knot.pvals*(tmax-tmin)/30.
        #pdifs=teffs/tau_eff-1.
        ###
        ured=np.ceil(redshift*5)/5.
        lred=np.floor(redshift*5)/5.
        usnap=self.GetSnap(ured)
        lsnap=self.GetSnap(lred)
        #Load the data
        (k,uPFp0)=self.loadpk(s_knot.bstft+self.suf+usnap+self.ext,s_knot.bfbox)
        uPower=np.zeros((npvals,np.size(k)))
        for i in np.arange(0,np.size(s_knot.names)):
            (k,uPower[i,:])=self.loadpk(s_knot.names[i]+self.suf+usnap+self.ext, s_knot.bfbox)
        (k,lPFp0)=self.loadpk(s_knot.bstft+self.suf+lsnap+self.ext,s_knot.bfbox)
        lPower=np.zeros((npvals,np.size(k)))
        for i in np.arange(0,np.size(s_knot.names)):
            (k,lPower[i,:])=self.loadpk(s_knot.names[i]+self.suf+lsnap+self.ext, s_knot.bfbox)
        PowerFluxes=5*((redshift-lred)*uPower+(ured-redshift)*lPower)
        PFp0=5*((redshift-lred)*uPFp0+(ured-redshift)*lPFp0)
        #So now we have an array of data values.
        #Pass each k value to flux_deriv in turn.
        for k in np.arange(0,nk):
            (dPF, d2PF,_)=self.flux_deriv(PowerFluxes[:,k]/PFp0[k], pdifs)
            results[k]=d2PF
            results[nk+k]=dPF
        return results

    def Getkbins(self):
        """Get the kbins to interpolate onto"""
        return np.arange(0,20,1)+0.5

    def plot_z(self,Knot,redshift,title="",ylabel="",legend=True):
        """ Plot comparisons between a bunch of sims on one graph
                plot_z(Redshift, Sims to use ( eg, A1.14).
                Note this will clear current figures."""
        #Load best-fit
        (simk,BFPk)=self.loadpk(Knot.bstft+self.suf+self.pre+self.GetSnap(redshift)+self.ext,self.bfbox)
        #Setup figure plot.
        ind=wheref(self.Zz, redshift)
        plt.figure(ind[0][0])
        plt.clf()
        if title != '':
            plt.title(title+" at z="+str(redshift),)
        plt.ylabel(ylabel)
        plt.xlabel(r"$\mathcal{F}$")
        line=np.array([])
        legname=np.array([])
        for sim in Knot.names:
            (k,Pk)=self.loadpk(sim+self.suf+self.pre+self.GetSnap(redshift)+self.ext,self.box)
            assert np.all(k == simk)
            line=np.append(line, plt.semilogy(simk,Pk/BFPk,linestyle="-", linewidth=1.5))
            legname=np.append(legname,sim)
        if legend:
            plt.legend(line,legname)
        return

    def GetFlat(self,directory):
        """Get a power spectrum in the flat format we use
        for inputting some cosmomc tables"""
        #For z=2.07 we need to average snap_011 and snap_010
        z=2.07
        (k1,PF_a)=self.loadpk(directory+self.suf+"snapshot_011"+self.ext, self.box)
        (k2,PF_b)=self.loadpk(directory+self.suf+"snapshot_010"+self.ext, self.box)
        PF1=(z-2.0)*5*(PF_b-PF_a)+PF_a
        z=2.52
        (k3,PF_a)=self.loadpk(directory+self.suf+"snapshot_009"+self.ext, self.box)
        (k4,PF_b)=self.loadpk(directory+self.suf+"snapshot_008"+self.ext, self.box)
        PF2=(z-2.4)*5*(PF_b-PF_a)+PF_a
        z=2.94
        (k5,PF_a)=self.loadpk(directory+self.suf+"snapshot_007"+self.ext, self.box)
        (k6,PF_b)=self.loadpk(directory+self.suf+"snapshot_006"+self.ext, self.box)
        assert np.all(k1==k2)
        assert np.all(k1==k3)
        assert np.all(k1==k4)
        assert np.all(k1==k5)
        assert np.all(k1==k6)
        PF3=(z-2.8)*5*(PF_b-PF_a)+PF_a
        return (PF1, PF2, PF3)


class MatterPow(EmulatedQuantity, PlotEmulatedQuantity):
    """ A class to load and plot matter power spectra """
    def __init__(self, Snaps=(),Zz=np.array([]),sdsskbins=np.array([]),knotpos=np.array([]), om=0.266,ob=0.0449, H0=0.71,box=60.0,base="/home/spb41/Lyman-alpha/MinParametricRecon/runs/",suf="matter-power/", ext=".0", matpre="PK-by-"):
        EmulatedQuantity.__init__(self, Snaps,Zz,sdsskbins,knotpos, om, H0,box,base,suf,ext)
        self.ob=ob
        self.pre=matpre
        self.ymin=0.4
        self.ymax=1.6
        self.figprefix="/matter-figure"

    def plot_z(self,Sims,redshift,title="Relative Matter Power",ylabel=r"$\mathrm{P}(k,p)\,/\,\mathrm{P}(k,p_0)$",legend=True):
        return super(MatterPow, self).plot_z(Sims,redshift,title,ylabel,legend=legend)

    def loadpk(self, path,box):
        """Load a Pk. Different function due to needing to be different for each class"""
        #Load baryon P(k)
        matter_power=np.loadtxt(self.base+path)
        scale=self.H0/box
        #Adjust Fourier convention.
        simk=matter_power[1:,0]*scale*2.0*math.pi
        Pkbar=matter_power[1:,1]/scale**3
        #Load DM P(k)
        matter_power=np.loadtxt(self.base+re.sub("by","DM",path))
        PkDM=matter_power[1:,1]/scale**3
        Pk=(Pkbar*self.ob+PkDM*(self.om-self.ob))/self.om
        return (simk,Pk)

    def plot_power(self,path, redshift,camb_filename=""):
        """ Plot absolute power spectrum, not relative"""
        (k_g, Pk_g)=super(MatterPow, self).plot_power(path,redshift)
        sigma=2.0
        pkg=np.loadtxt(self.base+path+self.suf+self.pre+self.GetSnap(redshift)+self.ext)
        samp_err=pkg[1:,2]
        sqrt_err=np.array(np.sqrt(samp_err))
        plt.loglog(k_g,Pk_g*(1+sigma*(2.0/sqrt_err+1.0/samp_err)),linestyle="-.",color="black")
        plt.loglog(k_g,Pk_g*(1-sigma*(2.0/sqrt_err+1.0/samp_err)),linestyle="-.",color="black")
        if camb_filename != "":
            camb=np.loadtxt(camb_filename)
            #Adjust Fourier convention.
            k=camb[:,0]*self.H0
            #NOW THERE IS NO h in the T anywhere.
            Pk=camb[:,1]
            plt.loglog(k/self.H0, Pk, linestyle="--")
        plt.xlim(0.01,k_g[-1]*1.1)
        return(k_g, Pk_g)
