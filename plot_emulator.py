"""Store various functions to plot things from the interpolator"""

import matplotlib.pyplot as plt
import numpy as np
from quadratic_emulator import EmulatedQuantity
from smooth import rebin,bounded_rebin
from wheref import wheref
from save_figure import save_figure
import os.path

def plot_error_sdss(interp, Sim, bstft, box, derivs, params,qarams=np.array([]),zzz=np.array([]),colour="",ylabel="",title="",ymax=0, ymin=0, legend=False):
    """ Make an interpolation error plot"""
    line=np.array([])
    legname=np.array([])
    sdss=interp.sdsskbins
    if np.size(zzz) == 0:
        zzz=interp.Zz
    for zz in zzz:
        err=interp.Get_Error_z(Sim,bstft,box,derivs,params,zz,qarams)
        sdmpc=interp.GetSDSSkbins(zz)
        err=rebin(err, interp.Getkbins(),sdmpc)
        #Rebin onto SDSS kbins.
        if colour != "":
            line=np.append(line,plt.semilogx(sdss,err,color=colour))
        else:
            line=np.append(line,plt.semilogx(sdss,err))
        legname=np.append(legname,"z="+str(zz))
    if title != "":
        plt.title(title)
    if ylabel != "":
        plt.ylabel(ylabel)
    plt.xlabel(r"$k_v\; (\mathrm{s}\,\mathrm{km}^{-1})$")
    if ymax != 0 and ymin !=0:
        plt.ylim(ymin,ymax)
    plt.xlim(sdss[0], sdss[-1])
    plt.xticks(np.array([sdss[0],3e-3,5e-3,0.01,sdss[-1]]),("0.0014","0.003","0.005","0.01","0.0178"))
    if legend:
        plt.legend(line, legname,bbox_to_anchor=(0., 0, 1., .25), loc=3,ncol=3, mode="expand", borderaxespad=0.)

def plot_error_all(interp, Sim, bstft, box, derivs, params,zzz=np.array([]),colour="",ylabel="",title="",ymax=0, ymin=0, legend=False):
    """ Make an interpolation error plot"""
    line=np.array([])
    legname=np.array([])
    if np.size(zzz) == 0:
        zzz=interp.Zz
    for zz in zzz:
        err=interp.Get_Error_z(Sim,bstft, box,derivs,params,zz)
        if colour != "":
            line=np.append(line,plt.semilogx(interp.Getkbins(),err,color=colour))
        else:
            line=np.append(line,plt.semilogx(interp.Getkbins(),err))
        legname=np.append(legname,"z="+str(zz))
    if title != "":
        plt.title(title)
    if ylabel != "":
        plt.ylabel(ylabel)
    plt.xlabel(r"$k\; (h\,\mathrm{Mpc}^{-1})$")
    plt.semilogx(interp.knotpos,np.ones(len(interp.knotpos)),"ro")
    if ymax != 0 and ymin !=0:
        plt.ylim(ymin,ymax)
    plt.xlim(interp.Getkbins()[0]*0.95, 4)
    if legend:
        plt.legend(line, legname,bbox_to_anchor=(0., 0, 1., .25), loc=3,ncol=3, mode="expand", borderaxespad=0.)

class PlotEmulatedQuantity(EmulatedQuantity):
    """Various plotting methods extracted from the emulator class"""
    def plot_z(self,a_knot,redshift,title="",ylabel="", legend=True):
        """ Plot comparisons between a bunch of sims on one graph
            plot_z(Redshift, Sims to use ( eg, A1.14).
            Note this will clear current figures."""
        #Load best-fit
        (simk,BFPk)=self.loadpk(a_knot.bstft+self.suf+self.pre+self.GetSnap(redshift)+self.ext,self.bfbox)
        #Setup figure plot.
        ind=wheref(self.Zz, redshift)
        plt.figure(ind[0][0])
        plt.clf()
        if title != '':
            plt.title(title+" at z="+str(redshift))
        plt.ylabel(ylabel)
        plt.xlabel(r"$k\; (\mathrm{Mpc}^{-1})$")
        line=np.array([])
        legname=np.array([])
        for sim in a_knot.names:
            (k,Pk)=self.loadpk(sim+self.suf+self.pre+self.GetSnap(redshift)+self.ext,self.box)
            oi = np.where(simk <= k[-1])
            ti = np.where(simk[oi] >= k[0])
            relP=rebin(Pk, k, simk[oi][ti])
            relP=relP/rebin(BFPk, simk, simk[oi][ti])
            line=np.append(line, plt.semilogx(simk[oi][ti]/self.H0,relP,linestyle="-"))
            legname=np.append(legname,sim)
        if legend:
            plt.legend(line,legname)
            plt.semilogx(self.knotpos,np.ones(len(self.knotpos)),"ro")
        plt.ylim(self.ymin,self.ymax)
        plt.xlim(simk[0]*0.8, 10)
        return

    def plot_all(self, a_knot,zzz=np.array([]), out=""):
        """ Plot a whole suite of snapshots: plot_all(Knot, outdir) """
        if np.size(zzz) == 0:
            zzz=self.Zz    #lolz
        for z in zzz:
            self.plot_z(a_knot,z)
            if out != "":
                save_figure(out+self.figprefix+str(z))
        return

    def plot_power(self,path, redshift,colour="black"):
        """ Plot absolute power spectrum, not relative"""
        (k_g,Pk_g)=self.loadpk(path+self.suf+self.pre+self.GetSnap(redshift)+self.ext,self.box)
        plt.loglog(k_g,Pk_g, color=colour)
        plt.xlim(0.01,k_g[-1]*1.1)
        plt.ylabel("P(k) /(h-3 Mpc3)")
        plt.xlabel("k /(h MPc-1)")
        plt.title("Power spectrum at z="+str(redshift))
        return(k_g, Pk_g)

    def plot_power_all(self, a_knot,zzz=np.array([]), out=""):
        """ Plot absolute power for all redshifts """
        if np.size(zzz) == 0:
            zzz=self.Zz    #lolz
        for z in zzz:
            ind=wheref(self.Zz, z)
            plt.figure(ind[0][0])
            for sim in a_knot.names:
                self.plot_power(sim,z)
            if out != "":
                save_figure(out+self.figprefix+str(z))
        return

    def plot_compare_two(self, one, onebox, two,twobox,colour=""):
        """ Compare two power spectra directly. Smooths result.
        plot_compare_two(first P(k), second P(k))"""
        (onek,oneP)=self.loadpk(one,onebox)
        (twok,twoP)=self.loadpk(two,twobox)
        relP = bounded_rebin(onek, oneP, twok, twoP, onek)
        plt.title("Relative Power spectra "+one+" and "+two)
        plt.ylabel(r"$P_2(k)/P_1(k)$")
        plt.xlabel(r"$k\; (h\,\mathrm{Mpc}^{-1})$")
        if colour == "":
            line=plt.semilogx(onek,relP)
        else:
            line=plt.semilogx(onek,relP,color=colour)
        plt.semilogx(self.knotpos,np.ones(len(self.knotpos)),"ro")
        ind=np.where(onek < 10)
        plt.ylim(min(relP[ind])*0.98,max(relP[ind])*1.01)
        plt.xlim(onek[0]*0.8, 10)
        return line

    def plot_compare_two_sdss(self, onedir, twodir, zzz=np.array([]), out="", title="", ylabel="", ymax=0,ymin=0, colour="",legend=False):
        """ Plot a whole redshift range of relative power spectra on the same figure.
            plot_all(onedir, twodir)
            Pass onedir and twodir as relative to basedir.
            ie, for default settings something like
            best-fit/flux-power/"""
        if np.size(zzz) == 0:
            zzz=self.Zz    #lolz
        line=np.array([])
        legname=np.array([])
        sdss=self.sdsskbins
        plt.xlabel(r"$k_v\; (s\,\mathrm{km}^{-1})$")
        for z in zzz:
            sim=self.pre+self.GetSnap(z)+self.ext
            (onek,oneP)=self.loadpk(os.path.join(onedir,sim),self.bfbox)
            (twok,twoP)=self.loadpk(os.path.join(twodir,sim),self.box)
            Pk = bounded_rebin(onek, oneP, twok, twoP, self.GetSDSSkbins(z))
            if colour == "":
                line=np.append(line, plt.semilogx(sdss,Pk))
            else:
                line=np.append(line, plt.semilogx(sdss,Pk,color=colour))
            legname=np.append(legname,"z="+str(z))
        plt.title(title)
        if ylabel != "":
            plt.ylabel(ylabel)
        if legend:
            plt.legend(line, legname,bbox_to_anchor=(0., 0, 1., .25), loc=3,ncol=3, mode="expand", borderaxespad=0.)
        if ymax != 0 and ymin !=0:
            plt.ylim(ymin,ymax)
        plt.xlim(sdss[0],sdss[-1])
        plt.xticks(np.array([sdss[0],3e-3,5e-3,0.01,sdss[-1]]),("0.0014","0.003","0.005","0.01","0.0178"))
        if out != "":
            save_figure(out)
        return plt.gcf()

    def plot_compare_two_all(self, onedir, twodir, zzz=np.array([]), out="", title="", ylabel="", ymax=0,ymin=0, colour="",legend=False):
        """ Plot a whole redshift range of relative power spectra on the same figure.
            plot_all(onedir, twodir)
            Pass onedir and twodir as relative to basedir.
            ie, for default settings something like
            best-fit/flux-power/
            onedir uses bfbox, twodir uses box"""
        if np.size(zzz) == 0:
            zzz=self.Zz    #lolz
        line=np.array([])
        legname=np.array([])
        for z in zzz:
            line=np.append(line, self.plot_compare_two(onedir+self.pre+self.GetSnap(z)+self.ext,self.bfbox,twodir+self.pre+self.GetSnap(z)+self.ext,self.box,colour))
            legname=np.append(legname,"z="+str(z))
        if title == "":
            plt.title("Relative Power spectra "+onedir+" and "+twodir)
        else:
            plt.title(title)
        if ylabel != "":
            plt.ylabel(ylabel)
        if legend:
            plt.legend(line, legname,bbox_to_anchor=(0., 0, 1., .25), loc=3,ncol=3, mode="expand", borderaxespad=0.)
        if ymax != 0 and ymin !=0:
            plt.ylim(ymin,ymax)
        if out != "":
            save_figure(out)
        return plt.gcf()

