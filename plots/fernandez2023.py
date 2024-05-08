"""Functions to create the figures presented in Fernandez et al 2023."""

import os.path as path
import numpy as np
import matplotlib.pyplot as plt
import h5py
import json
import glob
import matplotlib
import sys
sys.path.append('../')
from lyaemu import coarse_grid as cg
from lyaemu import gpemulator as gp
from lyaemu import likelihood as lk
from lyaemu import lyman_data as ld
from lyaemu.meanT import t0_likelihood as tlk
from lyaemu.meanT import t0_coarse_grid as tcg
from lyaemu.meanT import t0_gpemulator as tgp
import itertools as it
import getdist.plots as gdplt
from getdist.mcsamples import loadMCSamples
import os
import classylss.binding as CLASS


# set up the ticks and axes
plt.rc('xtick',labelsize=26)
plt.rc('ytick',labelsize=26)
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.major.width'] = 1.75
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['xtick.minor.width'] = 1.25
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.major.width'] = 1.75
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['ytick.minor.width'] = 1.25
plt.rcParams['axes.linewidth'] = 2

# some colors, and light (l) and lighter (ll) versions of them
c_flatirons = '#8B2131'
c_flatirons_l = '#c22e44'
c_flatirons_ll = '#d85b6e'
c_sunshine = '#CA9500'
c_sunshine_l = '#ffbe09'
c_sunshine_ll = '#ffcf46'
c_skyline = '#1D428A'
c_skyline_l = '#295dc3'
c_skyline_ll = '#5583db'
c_midnight = '#0E2240'



# corner plot for cosmological parameters
# chain_dirs can be a list of filepath/filename
def cosmo_corner(chain_dirs, savefile=None, labels=None):
    gd_samples = []
    for chain_dir in chain_dirs:
        nn, gr = np.loadtxt(os.path.abspath(chain_dir+'.progress'), usecols=(0, 3)).T
        gd_samples.append(loadMCSamples(chain_dir, settings={'ignore_rows':nn[np.where(gr < 1)[0][0]]/nn[-1]}))
    for gd_sample in gd_samples:
        print('Using', gd_sample.numrows, 'samples')

    gd_samples[0].paramNames.parWithName('Ap').label = 'A_\\mathrm{P}/10^{-9}'
    gd_samples[0].paramNames.parWithName('ns').label = 'n_\\mathrm{P}'
    params = ['tau0', 'dtau0', "ns", "Ap", "omegamh2"]
    plimits = np.array([[1.0, 1.25],[-0.35, 0.15],[0.86, 1.05], [1.2e-9, 2.0e-9], [0.14, 0.146]])
    gticks = np.array([[1.0,1.2],[-0.2,0.0],[0.9,0.97,1.03], [1.3e-9,1.6e-9,1.9e-9], [0.141,0.144]])
    gtlabels = np.array([['1.0','1.2'],['-0.2','0.0'],['0.9','0.97','1.03'], ['1.3','1.6','1.9'], ['0.141','0.144']])

    gdplot = gdplt.get_subplot_plotter()
    gdplot.settings.axes_fontsize = 20
    gdplot.settings.axes_labelsize = 28
    gdplot.settings.legend_fontsize = 20
    gdplot.settings.tight_layout = True
    gdplot.settings.figure_legend_loc = 'upper right'

    gdplot.triangle_plot(gd_samples, params, legend_labels=labels, filled=True, contour_lws=2.5, contour_ls='-', contour_colors=[c_sunshine, c_skyline, c_flatirons, c_midnight])
    for pi in range(4):
        for pi2 in range(pi + 1):
            ax = gdplot.subplots[pi, pi2]
            if pi != pi2:
                ax.set_ylim(plimits[pi])
                ax.set_yticks(gticks[pi], gtlabels[pi])
            ax.set_xlim(plimits[pi2])
            ax.set_xticks(gticks[pi2], gtlabels[pi2])
    if savefile is not None:
        gdplot.export(savefile)
    plt.show()

def astro_corner(chain_dirs, savefile=None, labels=None, bhprior=False):
    """Corner plots for astrophysical parameters"""
    gd_samples = []
    for chain_dir in chain_dirs:
        nn, gr = np.loadtxt(os.path.abspath(chain_dir+'.progress'), usecols=(0, 3)).T
        gd_samples.append(loadMCSamples(chain_dir, settings={'ignore_rows':nn[np.where(gr < 1)[0][0]]/nn[-1]}))
    for gd_sample in gd_samples:
        print('Using', gd_sample.numrows, 'samples')

    gd_samples[0].paramNames.parWithName('herei').label = 'z^{HeII}_i'
    gd_samples[0].paramNames.parWithName('heref').label = 'z^{HeII}_f'
    gd_samples[0].paramNames.parWithName('alphaq').label = '\\alpha_{q}'
    gd_samples[0].paramNames.parWithName('hireionz').label = 'z^{HI}'
    gd_samples[0].paramNames.parWithName('hub').label = r'v_\mathrm{scale}'
#     gd_samples[0].paramNames.parWithName('bhfeedback').label = '\\epsilon_{AGN}'

    params = ["herei", "heref", "alphaq", "hireionz", "a_lls", "a_dla", "fSiIII", 'hub']
    plimits = np.array([[3.5, 4.5], [2.2, 3.2], [1.3, 3.0], [6.5, 8.0],[-0.2, 0.25], [-0.035, 0.035], [0.006, 0.013],[0.65, 0.75]])
    gticks = np.array([[3.6,4.0,4.4], [2.4,2.7,3.0], [1.8,2.3,2.8], [7,7.5], [-0.1,0.1], [-0.02,0.02], [0.008,0.011], [0.68,0.72]])
    gtlabels = np.array([['3.6','4.0','4.4'], ['2.4','2.7','3.0'], ['1.8','2.3','2.8'], ['7.0','7.5'], ['-0.1','0.1'], ['-0.02','0.02'], ['0.008','0.011'], [0.68,0.72]])

    gdplot = gdplt.get_subplot_plotter()
    gdplot.settings.axes_fontsize = 20
    gdplot.settings.axes_labelsize = 28
    gdplot.settings.legend_fontsize = 24
    gdplot.settings.tight_layout = True
    gdplot.settings.figure_legend_loc = 'upper right'

    gdplot.triangle_plot(gd_samples, params, legend_labels=labels, filled=True, contour_lws=2.5, contour_ls='-', contour_colors=[c_sunshine, c_skyline, c_flatirons, c_midnight])
    for pi in range(5):
        for pi2 in range(pi + 1):
            ax = gdplot.subplots[pi, pi2]
            if pi != pi2:
                ax.set_ylim(plimits[pi])
                ax.set_yticks(gticks[pi], gtlabels[pi])
            if pi == 4 and pi2 == 4:
                if bhprior:
                    pmean, psigma = 0.05, 0.01
                    ax.plot([pmean, pmean], [0, 2], '-', lw=2.25, color=c_skyline_ll, zorder=0)
                    ax.fill_between([pmean-psigma, pmean-psigma, pmean+psigma, pmean+psigma], [0, 2, 2, 0], color=c_skyline_ll, alpha=0.5, zorder=0)
                    ax.fill_between([pmean-2*psigma, pmean-2*psigma, pmean+2*psigma, pmean+2*psigma], [0, 2, 2, 0], color=c_skyline_ll, alpha=0.25, zorder=0)
            ax.set_xlim(plimits[pi2])
            ax.set_xticks(gticks[pi2], gtlabels[pi2])
    if savefile is not None:
        gdplot.export(savefile)
    plt.show()

def temp_corner(chain_dirs, savefile=None, labels=None):
    """Corner plots for parameters constrained by temperature data"""
    gd_samples = []
    for chain_dir in chain_dirs:
        nn, gr = np.loadtxt(os.path.abspath(chain_dir+'.progress'), usecols=(0, 3)).T
        gd_samples.append(loadMCSamples(chain_dir, settings={'ignore_rows':nn[np.where(gr < 1)[0][0]]/nn[-1]}))
    for gd_sample in gd_samples:
        print('Using', gd_sample.numrows, 'samples')

    gd_samples[0].paramNames.parWithName('Ap').label = 'A_\\mathrm{P}/10^{-9}'
    gd_samples[0].paramNames.parWithName('herei').label = 'z^{HeII}_i'
    gd_samples[0].paramNames.parWithName('heref').label = 'z^{HeII}_f'
    gd_samples[0].paramNames.parWithName('alphaq').label = '\\alpha_{q}'
    gd_samples[0].paramNames.parWithName('hireionz').label = 'z^{HI}'

    params = ["Ap", "herei", "heref", "alphaq", "hireionz"]
    plimits = np.array([[1.2e-9, 2.6e-9], [3.5, 4.1], [2.6, 3.2], [1.4, 2.5], [6.5, 8.0]])
    gticks = np.array([[1.6e-9, 2.2e-9], [3.7,3.9], [2.8,3.0], [1.8,2.2], [7,7.5]])
    gtlabels = np.array([[1.6, 2.2], ['3.7','3.9'], ['2.8','3.0'], ['1.8','2.2'], ['7.0','7.5']])

    gdplot = gdplt.get_subplot_plotter()
    gdplot.settings.axes_fontsize = 20
    gdplot.settings.axes_labelsize = 28
    gdplot.settings.legend_fontsize = 24
    gdplot.settings.tight_layout = True
    gdplot.settings.figure_legend_loc = 'upper right'

    gdplot.triangle_plot(gd_samples, params, legend_labels=labels, filled=True, contour_lws=2.5, contour_ls='-', contour_colors=[c_sunshine, c_skyline, c_flatirons, c_midnight])
    for pi in range(5):
        for pi2 in range(pi + 1):
            ax = gdplot.subplots[pi, pi2]
            if pi != pi2:
                ax.set_ylim(plimits[pi])
                ax.set_yticks(gticks[pi], gtlabels[pi])
            ax.set_xlim(plimits[pi2])
            ax.set_xticks(gticks[pi2], gtlabels[pi2])
    if savefile is not None:
        gdplot.export(savefile)
    plt.show()

def find_sigma8(spectralp, ap, h0, omh2):
    """Get sigma8 for a power spectrum
    ."""
    #Precision
    #pre_params = {'tol_background_integration': 1e-9, 'tol_perturb_integration' : 1.e-7, 'tol_thermo_integration':1.e-5, 'k_per_decade_for_pk': 50, 'k_bao_width': 8, 'k_per_decade_for_bao':  200, 'neglect_CMB_sources_below_visibility' : 1.e-30, 'transfer_neglect_late_source': 3000., 'l_max_g' : 50, 'l_max_ur':150}
    #Class takes omega_m h^2 as parameters
    omegab = 0.0486
    ocdm = omh2/h0**2 - omegab
    preparams = {'h':h0, 'Omega_cdm':ocdm,'Omega_b':omegab, 'Omega_k': 0, 'n_s': spectralp}
    preparams['A_s'] = (0.4/(2*np.pi))**(spectralp - 1) * ap
    #Pass options for the power spectrum
    preparams.update({'output': 'mPk', 'z_pk': 0})
    #Make the power spectra module
    engine = CLASS.ClassEngine(preparams)
    powspec = CLASS.Spectra(engine)
#     print("sigma_8(z=0) = ", powspec.sigma8, "A_s = ",powspec.A_s, 'Ap ',ap, 'np', spectralp, flush=True)
    return powspec.sigma8

def print_latex_table(chain_dirs, labels):
    """
    Print Latex table of the 1 and 2 sigma contours.
    """
    params = np.array(["ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2", "hireionz", 'tau0', 'dtau0',"a_lls", "a_dla", "fSiIII"])
    gd_samples = []
    for chain_dir in chain_dirs:
        nn, gr = np.loadtxt(os.path.abspath(chain_dir+'.progress'), usecols=(0, 3)).T
        gd_sample = loadMCSamples(chain_dir, settings={'ignore_rows':nn[np.where(gr < 1)[0][0]]/nn[-1]})
        gd_sample.paramNames.parWithName('Ap').label = r'A_\mathrm{P}/10^{-9}'
        gd_sample.paramNames.parWithName('ns').label = r'n_\mathrm{P}'
        gd_sample.paramNames.parWithName('herei').label = r'z^{HeII}_i'
        gd_sample.paramNames.parWithName('heref').label = r'z^{HeII}_f'
        gd_sample.paramNames.parWithName('alphaq').label = r'\alpha_{q}'
        gd_sample.paramNames.parWithName('hireionz').label = r'z^{HI}'
        gd_sample.paramNames.parWithName('hub').label = r'v_\mathrm{scale}'
        gd_sample.paramNames.parWithName('tau0').label = '\\tau_0'
        gd_sample.thin(40)
        AsVec = (0.4/(2*np.pi))**(gd_sample['ns']-1) * gd_sample['Ap'] * 1e9
        gd_sample.addDerived(paramVec=AsVec, name=r"A_\mathrm{s}/10^{-9}")
        print("samples ",np.size(AsVec),flush=True)
        sigmaVec = [find_sigma8(np, ap, h0, omh2) for (np, ap, h0, omh2) in zip(gd_sample['ns'], gd_sample['Ap'], gd_sample['hub'], gd_sample['omegamh2'])]
        gd_sample.addDerived(paramVec=sigmaVec, name=r"\sigma_8")
        gd_samples.append(gd_sample)

    # This does not work, despite docs: Traceback (most recent call last):
    # print(ResultTable(ncol=2,results=gd_samples,paramList=params, limit=2, titles=labels).tableTex())
    # File "/rhome/sbird/.local/lib/python3.9/site-packages/getdist/types.py", line 300, in __init__
    #  self.tableParamNames = self.tableParamNames.filteredCopy(paramList)
    # AttributeError: 'MCSamples' object has no attribute 'filteredCopy'
    #print(ResultTable(ncol=2,results=gd_samples,paramList=params, limit=2, titles=labels).tableTex())
    for gd, label in zip(gd_samples, labels):
        print(label)
        print(gd.getTable(columns=1, limit=1).tableTex())
        print(gd.getTable(columns=1, limit=2).tableTex())
        print(gd.PCA(params))
#         print(gd.getLikeStats())

def full_corner(chain_dirs, savefile=None, labels=None, simpar=None):
    """
    Full corner plot. for chain_dirs
    simpar: Pass array of correct parameters if known (ie, if input is a simulation).
    """
    gd_samples = []
    for chain_dir in chain_dirs:
        nn, gr = np.loadtxt(os.path.abspath(chain_dir+'.progress'), usecols=(0, 3)).T
        gd_samples.append(loadMCSamples(chain_dir, settings={'ignore_rows':nn[np.where(gr < 1)[0][0]]/nn[-1]}))
    for gd_sample in gd_samples:
        print('Using', gd_sample.numrows, 'samples')

    gd_samples[0].paramNames.parWithName('Ap').label = 'A_\\mathrm{P}/10^{-9}'
    gd_samples[0].paramNames.parWithName('ns').label = 'n_\\mathrm{P}'
    gd_samples[0].paramNames.parWithName('herei').label = 'z^{HeII}_i'
    gd_samples[0].paramNames.parWithName('heref').label = 'z^{HeII}_f'
    gd_samples[0].paramNames.parWithName('alphaq').label = '\\alpha_{q}'
    gd_samples[0].paramNames.parWithName('hireionz').label = 'z^{HI}'
    gd_samples[0].paramNames.parWithName('hub').label = r'v_\mathrm{scale}'
#     gd_samples[0].paramNames.parWithName('bhfeedback').label = '\\epsilon_{AGN}'
    gd_samples[0].paramNames.parWithName('tau0').label = '\\tau_0'

    params = np.array(["ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2", "hireionz", 'tau0', 'dtau0',"a_lls", "a_dla", "fSiIII"])
    plimits = np.array([[0.8, 1.05], [1.2e-9, 2.6e-9], [3.5, 4.5], [2.2, 3.2], [1.3, 3.0], [0.65, 0.75], [0.14, 0.146], [6.5, 8.0], [0.92, 1.28],[-0.4, 0.25],[-0.2, 0.25],  [-0.035, 0.035], [0.006, 0.013]])
    gticks = np.array([[0.9,1.03], [1.6e-9,2.2e-9], [3.6,4.0,4.4], [2.4,2.7,3.0], [1.8,2.3,2.8], [0.68,0.72], [0.141,0.144], [7,7.5], [1.0,1.2],[-0.2,0.1], [-0.1,0.1], [-0.02,0.02], [0.008,0.011]])
    gtlabels = np.array([['0.9','1.03'], ['1.6','2.2'], ['3.6','4.0','4.4'], ['2.4','2.7','3.0'], ['1.8','2.2','2.8'], ['0.68','0.72'], ['0.141','0.144'], ['7.0','7.5'], ['1.0','1.2'],['-0.2','0.1'], ['-0.1','0.1'], ['-0.02','0.02'], ['0.008','0.011']])
    if simpar is not None:
        params = params[:-3]
        plimits = plimits[:-3]
        gticks = gticks[:-3]
        gtlabels = gtlabels[:-3]
    nparams = np.size(params)

    gdplot = gdplt.get_subplot_plotter()
    gdplot.settings.axes_fontsize = 20
    gdplot.settings.axes_labelsize = 28
    gdplot.settings.legend_fontsize = 34
    gdplot.settings.tight_layout = True
    gdplot.settings.figure_legend_loc = 'upper right'

    gdplot.triangle_plot(gd_samples, params, legend_labels=labels, filled=True, contour_lws=2.5, contour_ls='-', contour_colors=[c_sunshine, c_skyline, c_flatirons, c_midnight])
    for pi in range(nparams):
        for pi2 in range(pi + 1):
            ax = gdplot.subplots[pi, pi2]
            if pi != pi2:
                ax.set_ylim(plimits[pi])
                ax.set_yticks(gticks[pi], gtlabels[pi])
            ax.set_xlim(plimits[pi2])
            ax.set_xticks(gticks[pi2], gtlabels[pi2])
            if simpar is not None:
                #Because of epsilon_AGN
                skip = pi2 > nparams-3
                ax.axvline(x=simpar[pi2 + skip], ls='--', color="black", lw=2.2)
                if pi != pi2:
                    skip = pi > nparams-3
                    ax.axhline(y=simpar[pi+skip], ls='--', color="black", lw=2.2)
    if savefile is not None:
        gdplot.export(savefile)
    plt.show()


def plot_correlation(correl_file = "correlation-z26-46-t0.txt"):
    """Plot a heat map of the correlation matrix"""
    correl = np.loadtxt(correl_file)
    plt.imshow(correl, vmin=-1, vmax=1)
    plt.colorbar()
    labels = [r'$n_\mathrm{P}$', r'$A_\mathrm{P}/10^{-9}$', r'$z^{HeII}_i$', r'$z^{HeII}_f$', r'$\alpha_{q}$', r'$v_\mathrm{scale}$', r'$\Omega_M h^2$', r'$z^{HI}$', r'$\tau_0$', r'$d\tau_0$', r'$\alpha_{lls}$', r'$\alpha_{DLA}$', r'fSiIII']
    # Show all ticks and label them with the respective list entries
    plt.xticks(np.arange(len(labels)), labels=labels, size=15)
    plt.yticks(np.arange(len(labels)), labels=labels, size=15)
    # Rotate the tick labels and set their alignment.
    plt.setp(plt.gca().get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    plt.savefig("correlation_z26_46_t0.pdf")
    plt.show()

def plot_samples(lores_json, hires_json, savefile=None, t0_samps=None):
    """plot the samples, parameter limits"""
    # t0_samps is the indices of the samples that are to be highlighted
    # get samples
    hires = np.array(json.load(open(hires_json, 'r'))['sample_params'])
    lores = np.array(json.load(open(lores_json, 'r'))['sample_params'])

    # Difference between FPS and T0 samples
    if t0_samps is not None:
        t0 = lores[t0_samps]
        uu, nn = np.unique(np.concatenate([lores, t0]), axis=0, return_counts=True)
        lores = uu[np.where(nn==1)]
    nsim, npar = np.shape(lores)

    # get parameter limits
    plimits = np.array(json.load(open(lores_json, 'r'))['param_limits'])

    # parameter names - update formatting from the json file
    names = [r'$\bf{n_P}$', r'$\bf{A_p}$', r'$\bf{z^{HeII}_i}$', r'$\bf{z^{HeII}_f}$', r'$\bf{\alpha_q}$', r'$v_\mathrm{scale}$', r'$\bf{\Omega_M h^2}$',
             r'$\bf{z^{HI}}$', r'$\bf{\epsilon_{AGN}}$']

    # make the plot
    yy = np.ones(nsim)
    fig, ax = plt.subplots(figsize=(10.625, 11), nrows=npar, ncols=1)
    for i in range(npar):
        ax[i].set_yticks([i])
        ax[i].set_yticklabels([names[i]], fontsize=26)
        ax[i].plot(lores[:, i], i*yy, 'x', color=c_midnight, ms=15, mew=2.5, alpha=0.66)
        if t0_samps is not None:
            ax[i].plot(t0[:, i], i*np.ones(np.shape(t0)[0]), 's', color=c_skyline_ll, ms=15, mew=3, mfc='none')
        ax[i].plot(hires[:, i], i*np.ones(np.shape(hires)[0]), 'o', color=c_flatirons_l, ms=15, mew=3, mfc='none')
        ax[i].set_xlim(plimits[i])
        ax[i].set_xticks(plimits[i])
        ax[i].set_xticklabels(plimits[i])

        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        ax[i].yaxis.set_ticks_position('left')
        ax[i].xaxis.set_ticks_position('bottom')
        ax[i].patch.set_facecolor('none')

    fig.subplots_adjust(hspace=2, wspace=0)
    if savefile is not None:
        fig.savefig(savefile, bbox_inches='tight', pad_inches=0)
    plt.show()

def plot_fps_obs_pred(basedir, chain_dirs, traindir=None, HRbasedir=None, savefile=None, labels=None, datapf=None):
    """plot the flux power spectrum observations, and some max posterior predictions"""
    # get the observations
    boss = ld.BOSSData()
    boss_err = boss.covar_diag.reshape(13,-1)[::-1]
    if datapf is None:
        bosspf = boss.get_pf().reshape(13,-1)[::-1]
    else:
        bosspf = datapf
    bosskf = boss.kf.reshape(13,-1)
    bosspf *= bosskf  / np.pi
    boss_err *= bosskf**2 / np.pi**2
    call_names = ['dtau0', 'tau0', 'ns', 'Ap', 'herei', 'heref', 'alphaq', 'hub', 'omegamh2', 'hireionz', 'bhfeedback', 'a_lls', 'a_dla', 'fSiIII']
    datacorr = True
    #Simulated data has no corrections
    if datapf is not None:
        call_names = call_names[:-3]
        datacorr = False
    # set up the likelihood class
    like = lk.LikelihoodClass(basedir, tau_thresh=1e6, max_z=4.6, min_z=2.2, traindir=traindir, HRbasedir=HRbasedir, data_corr=datacorr, loo_errors=True)
    zz = np.round(like.zout, 1)

    okf, pred, std = [], [], []
    for chaindir in chain_dirs:
        # get best parameters for each chain
        gd_sample1 = loadMCSamples(chaindir)
        best_par = []
        for i in range(np.size(call_names)):
            getgd = gd_sample1.get1DDensity(call_names[i])
            probs = getgd.P
            pvals = getgd.x
            best = pvals[np.where(probs == probs.max())][0]
            #Avoid the edges a little for likelihood calculation
            if best >= like.param_limits[i,1]:
                best*=0.9999
            if best <= like.param_limits[i,0]:
                best/=0.9999
            best_par.append(best)
        best_par = np.array(best_par)
        okfi, predi, stdi = like.get_predicted(best_par[:like.ndim-len(like.data_params)])
        print(chaindir, " Best: ", best_par)
        okf.append(okfi)
        tot_chi2 = 0
        for bb in range(like.zout.size):
            if datacorr:
                predi[bb] = predi[bb]*like.get_data_correction(okfi[bb], best_par, like.zout[bb])
            predi[bb] = okfi[bb] * predi[bb] / np.pi
            stdi[bb] = okfi[bb] * stdi[bb] / np.pi
            chi2_bin = like.likelihood_per_zbin(like.zout[bb], best_par, include_emu=False, data_power=datapf, per_dof=True)
            tot_chi2 += chi2_bin
            print("zz: %g chi2/dof: %g" % (like.zout[bb], chi2_bin))
        print(chaindir, "Total chi^2 per degree",tot_chi2/like.zout.size)
        pred.append(predi)
        std.append(stdi)

    nrows, ncols = 3, 2
    colors = [c_sunshine, c_skyline_l, c_flatirons]
    linestyles = ['-.', '--', '-']
    zorders = [1,3,2]
    fig, axes = plt.subplots(figsize=(10.625*2, 11*1.75), nrows=nrows, ncols=ncols, sharex=True, gridspec_kw={'height_ratios': [1, 1, 1]})
    axes = axes.flatten()
    for mm, ax in enumerate(axes):
        mplot = [2*mm, 2*mm+1]
        if mm == 5:
            mplot.append(2*mm+2)
        for m in mplot:
            for ii in range(len(pred)):
#                 ax.errorbar(okf[ii][m], pred[ii][m], yerr=std[ii][m], fmt='-', color=colors[ii], lw=2)
                ax.errorbar(okf[ii][m], pred[ii][m], color=colors[ii], lw=4, ls=linestyles[ii], zorder=zorders[ii])
            ax.plot(bosskf[m], bosspf[m], '-o', color=c_midnight, lw=2, zorder=0)
            ax.fill_between(bosskf[m], bosspf[m]-np.sqrt(boss_err[m]), bosspf[m]+np.sqrt(boss_err[m]), color=c_midnight, alpha=0.35, zorder=0)
        ax.text(0.014, 1.5*np.min(bosspf[m]), r'z: '+str(zz[np.min(mplot)])+'-'+str(zz[np.max(mplot)]), fontsize=30)
        if mm % 2 == 0:
            ax.tick_params(which='both', direction='inout', right=False, labelright=False, labelleft=True, length=12)
            ax.tick_params(which='minor', length=8, labelright=False, labelleft=False)
        else:
            ax.tick_params(which='both', direction='inout', right=True, left=False, labelright=True, labelleft=False, length=12)
            ax.tick_params(which='minor', length=8, labelright=False, labelleft=False)
        # ax.set_ylim(ymin=0)
#         ax.set_yscale('log')
    panel0_label_pos = np.max(bosspf[0]+np.sqrt(boss_err[0])) * 0.935
    axes[0].text(0.15e-2, panel0_label_pos, 'Chabanier 2019', fontsize=24, color=c_midnight)
    for ii in range(len(pred)):
        axes[0].text(0.15e-2, panel0_label_pos-((ii+1)*0.05), labels[ii], fontsize=24, color=colors[ii])
    # add figure centered x- and y-axis labels
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.ylabel(r'$k P_F(k) / \pi$', size=30, labelpad=25)
    plt.xlabel('k [s/km]', size=30)
    fig.subplots_adjust(hspace=0, wspace=0)
    if savefile is not None:
        plt.savefig(savefile)
    plt.show()

def plot_t0_obs_pred(basedir, chain_dirs, HRbasedir=None, savefile=None, labels=None, datadir=None, dataparams=None):
    """Plot the mean temperature observations, and some max posterior prediction"""
    # get the observations
    gaikwad = np.loadtxt('../lyaemu/data/Gaikwad/Gaikwad_2020b_T0_Evolution_All_Statistics.txt').T # temperature at mean density
    if datadir is not None:
        simt0 = tlk.load_data(datadir+'/emulator_meanT.hdf5', dataparams, max_z=3.8, min_z=2.0)
        #Last element left alone as sim doesn't have z=2.0
        #Also order needs to be reversed
        gaikwad[7][1:] = simt0[::-1]
    # get an emulator, to make the prediction
    temu = tcg.T0Emulator(basedir, max_z=4.6, min_z=2.2)
    temu.load()
    if HRbasedir is not None:
        gpemu = temu.get_MFemulator(HRbasedir=HRbasedir, max_z=4.6, min_z=2.2)
    else:
        gpemu = temu.get_emulator(max_z=4.6, min_z=2.2)

    # names in the chain files for each parameter
    call_names = ['ns', 'Ap', 'herei', 'heref', 'alphaq', 'hub', 'omegamh2', 'hireionz', 'bhfeedback']
    pred, std = [], []
    for chaindir in chain_dirs:
        # get best parameters for each chain
        gd_sample1 = loadMCSamples(chaindir)
        best_par = []
        for i in range(np.size(call_names)):
            getgd = gd_sample1.get1DDensity(call_names[i])
            probs = getgd.P
            pvals = getgd.x
            best = pvals[np.where(probs == probs.max())][0]
            best_par.append(best)
        pred.append(gpemu.predict(np.array(best_par))[0].flatten())
        std.append(gpemu.predict(np.array(best_par))[1].flatten())

    colors = [c_sunshine, c_skyline_l, c_flatirons, c_flatirons_ll]
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10.625, 8))
    plt.setp(ax, xticks=[4.6,4.2,3.8,3.4,3.0,2.6,2.2], xlim=[4.7,2.1], ylim=[0.7,1.65])
    for ii in range(len(pred)):
        ax.plot(temu.myspec.zout[4:], pred[ii]/1e4, linestyle='--', color=colors[ii], linewidth=4., alpha=0.95)
    ax.plot(gaikwad[0], gaikwad[7]/1e4, color=c_midnight, linestyle='-', marker='o', lw=2.5)
    ax.fill_between(gaikwad[0], gaikwad[7]/1e4-gaikwad[8]/1e4, gaikwad[7]/1e4+gaikwad[8]/1e4, color=c_midnight, alpha=0.5)
    ax.text(4.55, 1.55, 'Gaikwad 2021 (Flux Power)', fontsize=24, color=c_midnight)
    if labels is not None:
        for ii in range(len(pred)):
            ax.text(4.55, 1.55-((1+ii)*0.08), labels[ii], fontsize=24, color=colors[ii])
    ax.tick_params(which='both', direction='inout', right=False, labelright=False, labelleft=True, length=12)
    ax.tick_params(which='minor', length=8, labelright=False, labelleft=False)
    ax.set_xlabel("Redshift", fontsize=26)
    ax.set_ylabel(r"Temperature ($\times 10^4$)", fontsize=26)
    if savefile is not None:
        plt.savefig(savefile, bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_1d_marginals(basedir, chains, savefile=None, labels=None):
    """plot showing the emulator errors across each parameter space, along with
       the resulting posteriors, and the training samples.
       chains is a list of the filepath/filename for each chain set"""
    # get chains
    gd_sample = []
    for chainfile in chains:
        nn, gr = np.loadtxt(os.path.abspath(chainfile+'.progress'), usecols=(0, 3)).T
        gd_sample.append(loadMCSamples(chainfile, settings={'ignore_rows':nn[np.where(gr < 1)[0][0]]/nn[-1]}))
    # get parameter limits, adjust Ap scale for plotting
    plimits = np.array([[0.8, 0.995], [1.2e-9, 2.6e-9], [3.5, 4.1], [2.6, 3.2], [1.3, 2.5], [0.65, 0.75], [0.14, 0.146], [6.5, 8.0], [0.92, 1.28],[-0.4, 0.25]])
    nsteps = 30
    params = np.linspace(plimits[:,0], plimits[:,1], nsteps).T

    plimits[1] *= 10**9
    params[1] *= 10**9
    # parameter names
    names = [r'$\bf{n_P}$', r'$10^{9}\bf{A_p}$', r'$\bf{z^{HeII}_i}$', r'$\bf{z^{HeII}_f}$', r'$\bf{\alpha_q}$', r'$\bf{v}_\mathrm{scale}$', r'$\bf{\Omega_M h^2}$',
             r'$\bf{z^{HI}}$', r'$\bf{\tau_0}$', r'$\bf{d\tau_0}$'] #r'$\epsilon_{AGN}$',
    call_names = ['ns', 'Ap', 'herei', 'heref', 'alphaq', 'hub', 'omegamh2', 'hireionz', 'tau0', 'dtau0']
    assert len(names) == len(call_names)
    rounder = 2*np.ones(len(names), dtype=np.int)
    rounder[0] = 3
    rounder[6] = 3
    colors_dist = [c_sunshine, c_flatirons, c_skyline_ll]

    nrows = len(call_names)//2
    # make the plot
    fig, ax = plt.subplots(figsize=(10.625*2, 14), nrows=nrows, ncols=2)
    for i in range(nrows):
        for j in range(2):
            cc = int(2*i+j)
            ax[i,j].set_yticks([0.1])
            ax[i,j].set_yticklabels([names[cc]], fontsize=30)
            use_ticks = np.copy(plimits[cc,:])
            for k in range(np.size(gd_sample)):
                probs = gd_sample[k].get1DDensity(call_names[cc]).P
                pvals = gd_sample[k].get1DDensity(call_names[cc]).x
                if call_names[cc] == 'Ap':
                    pvals *= 10**9
                #Use best fit unless it is close to an existing tick
                best = pvals[np.where(probs == probs.max())]
                if np.min(np.abs(best - use_ticks)/(plimits[cc,1]-plimits[cc,0])) > 0.2:
                    use_ticks = np.append(use_ticks, best)
                ax[i,j].plot(pvals, probs, color=colors_dist[k], lw=4, label=labels[k])
            ax[i,j].set_xlim(plimits[cc])
            ax[i,j].set_xticks(use_ticks, np.round(use_ticks, rounder[cc]))#, rotation=30)
            if cc == 0:
                ticks = ax[i,j].xaxis.get_majorticklabels()
                ticks[0].set_ha("left")
                ticks[1].set_ha("right")
            ax[i,j].set_ylim([0, 1.03])

            ax[i,j].spines['right'].set_visible(False)
            ax[i,j].spines['top'].set_visible(False)
            ax[i,j].spines['left'].set_visible(False)
            ax[i,j].yaxis.set_ticks_position('left')
            ax[i,j].xaxis.set_ticks_position('bottom')
            ax[i,j].patch.set_facecolor('none')

    ax[2,1].legend(loc=[-1., 4.2], fontsize=28, numpoints=2, ncol=2)
    fig.subplots_adjust(hspace=0.5, wspace=0.25)
    if savefile is not None:
        fig.savefig(savefile, bbox_inches='tight', pad_inches=0)
    plt.show()

def plot_err_dists(basedir, loo_file, chains, traindir=None, savefile=None, labels=None):
    """plot showing the emulator errors across each parameter space, along with
       the resulting posteriors, and the training samples.
       chains is a list of the filepath/filename for each chain set"""
    # get samples
    lores = np.array(json.load(open(basedir+'/emulator_params.json', 'r'))['sample_params'])
    nsim, npar = lores.shape
    hires = np.array(json.load(open(basedir+'/hires/emulator_params.json', 'r'))['sample_params'])
    hfnsim = hires.shape[0]
    # get loo errors
    ff = h5py.File(loo_file, 'r')
    MFfpp, MFfpt = MFfpp, MFfpt = ff['flux_predict'][:], ff['flux_true'][:]
    ff.close()
    loo_error = np.mean(np.abs(MFfpp-MFfpt))
    # get chains
    gd_sample = []
    for chainfile in chains:
        nn, gr = np.loadtxt(os.path.abspath(chainfile+'.progress'), usecols=(0, 3)).T
        gd_sample.append(loadMCSamples(chainfile, settings={'ignore_rows':nn[np.where(gr < 1)[0][0]]/nn[-1]}))
    # get parameter limits, adjust Ap scale for plotting
    plimits = np.array(json.load(open(basedir+'/emulator_params.json', 'r'))['param_limits'])

    # get the emulator errors across each parameter space
    like = lk.LikelihoodClass(basedir, tau_thresh=1e6, optimise_GP=True, traindir=traindir, HRbasedir=basedir+'/hires')
    nsteps = 30
    params = np.linspace(plimits[:,0], plimits[:,1], nsteps).T
    midp = np.concatenate([np.array([-0.075, 1.]), (plimits[:,0] + (plimits[:,1]-plimits[:,0])/2)])
    errors = np.zeros([params.shape[0], nsteps, 13, 35])
    for i in range(params.shape[0]):
        for j in range(nsteps):
            newp = np.copy(midp)
            newp[i+2] = params[i, j]
            # calculate error
            _, _, std = like.get_predicted(newp)
            errors[i, j] = std

    plimits[1] *= 10**9
    lores[:,1] *= 10**9
    hires[:,1] *= 10**9
    params[1] *= 10**9
    # parameter names
    names = [r'$\bf{n_P}$', r'$10^{9}\bf{A_p}$', r'$\bf{z^{HeII}_i}$', r'$\bf{z^{HeII}_f}$', r'$\bf{\alpha_q}$', r'$\bf{v}_\mathrm{scale}$', r'$\bf{\Omega_M h^2}$',
             r'$\bf{z^{HI}}$', r'$\tau_0$', r'$d\tau_0$'] #r'$\epsilon_{AGN}$',
    call_names = ['ns', 'Ap', 'herei', 'heref', 'alphaq', 'hub', 'omegamh2', 'hireionz', 'tau0', 'dtau0']
    assert len(names) == len(call_names)
    rounder = 2*np.ones(len(names), dtype=np.int)
    rounder[0] = 3
    rounder[6] = 3
    colors_dist = np.array([c_flatirons, c_sunshine, c_skyline])

    # make the plot
    fig, ax = plt.subplots(figsize=(10.625*2, 14), nrows=4, ncols=2)
    for i in range(4):
        for j in range(2):
            cc = int(2*i+j)
            ax[i,j].set_yticks([0.1])
            ax[i,j].set_yticklabels([names[cc]], fontsize=30)
            ax[i,j].plot(lores[:, cc], np.ones(nsim)*0.1, 'x', color=c_midnight, ms=15, mew=2.5, alpha=0.66, label='LF Samples')
            ax[i,j].plot(hires[:, cc], np.ones(hfnsim)*0.1, 'o', color=c_flatirons_l, ms=17, mew=3, mfc='none', label='HF Samples')
            ax[i,j].plot(params[cc], errors[cc, :].mean(axis=1).mean(axis=1), 'o', color=c_sunshine, label='GP Predicted Errors')
            ax[i,j].plot(params[cc], loo_error*np.ones(nsteps), '--', color=c_flatirons, lw=2.5, label='Leave-One-Out Errors')
            use_ticks = np.copy(plimits[cc,:])
            for k in range(np.size(gd_sample)):
                probs = gd_sample[k].get1DDensity(call_names[cc]).P
                pvals = gd_sample[k].get1DDensity(call_names[cc]).x
                if call_names[cc] == 'Ap':
                    pvals *= 10**9
                best = pvals[np.where(probs == probs.max())]
                #Use best fit unless it is close to an existing tick
                if np.min(np.abs(best/use_ticks[:2+k]-1)) > 0.2:
                    use_ticks = np.append(use_ticks, best)
                ax[i,j].plot(pvals, probs, color=colors_dist[k], lw=2, label=labels[k])
            ax[i,j].set_xlim(plimits[cc])
            ax[i,j].set_xticks(use_ticks, np.round(use_ticks, rounder[cc]))#, rotation=30)
            if cc == 0:
                ticks = ax[i,j].xaxis.get_majorticklabels()
                ticks[0].set_ha("left")
                ticks[1].set_ha("right")
            ax[i,j].set_ylim([0, 1.03])

            ax[i,j].spines['right'].set_visible(False)
            ax[i,j].spines['top'].set_visible(False)
            ax[i,j].spines['left'].set_visible(False)
            ax[i,j].yaxis.set_ticks_position('left')
            ax[i,j].xaxis.set_ticks_position('bottom')
            ax[i,j].patch.set_facecolor('none')

    ax[2,1].legend(loc=[-1., 4.2], fontsize=28, numpoints=2, ncol=2)
    fig.subplots_adjust(hspace=0.5, wspace=0.25)
    if savefile is not None:
        fig.savefig(savefile, bbox_inches='tight', pad_inches=0)
    plt.show()

def get_params(savefile, data_index=21):
    """Get parameters from savefile"""
    with h5py.File(savefile, 'r') as data_hdf5:
        if np.size(np.shape(data_hdf5["params"])) == 1:
            simpar = data_hdf5['params'][1:]
        else:
            simpar = data_hdf5['params'][data_index][1:]
    return simpar

if __name__ == "__main__":
    #Plot chains from known truth data
    #Get simulation parameters
    tau_thresh=1e6
    basedir="../dtau-48-48/"
    savefile = basedir+'hires/mf_emulator_flux_vectors_tau'+str(int(tau_thresh))+".hdf5"
    simpar1 = np.concatenate([get_params(savefile, data_index=21), [1.15, 0.]])
    #Do plot
    full_corner(["chains/simdat/mf-48-48-z2.2-4.6",], "simdat.pdf", labels=["LOO"], simpar=simpar1)
    #Get simulation parameters
    savefile = basedir+'/ns0.881-seed/mf_emulator_flux_vectors_tau'+str(int(tau_thresh))+".hdf5"
    simpar2 = np.concatenate([get_params(savefile), [1.0, 0.]])
    #Do plot
    chain_dirs = ["chains/simdat/seed-loo-2.2-4.6","chains/simdat/seed-gperr-2.2-4.6","chains/simdat/seed-meant-loo-2.2-4.6" ]
    labels = ["Seed FPS", "Seed FPS+GPERR",r"Seed FPS+$T_0$" ]
    full_corner(chain_dirs, "simdat-seed.pdf", labels=labels, simpar=simpar2)
    #Make a plot of the best-fit P_F(k) with a different seed
    traindir=basedir+"/trained_mf"
    with h5py.File(savefile, 'r') as data_hdf5:
            datapf = data_hdf5["flux_vectors"][:]
            datapf=datapf.reshape(13, -1)
    plot_fps_obs_pred(basedir, chain_dirs, traindir=traindir, HRbasedir=None, savefile="seed-best-fit.pdf", labels=labels, datapf=datapf)
    plot_t0_obs_pred(basedir, chain_dirs, HRbasedir=None, savefile="seed-best-fit-t0.pdf", labels=labels, datadir=basedir+'ns0.881-seed', dataparams=None)
    #Make corner plot of best-fit P_F(k)
    chain_dirs = ["chains/fps-meant/mf-48-48-z2.6-4.6",
                  "chains/fps-meant/mf-48-48-z2.4-4.6",
                  "chains/fps-meant/mf-48-48-z2.2-4.6"]
    labels = [r"FPS,+ $T_0$ z = $2.6$ - $4.6$",
              r"FPS + $T_0$, z = $2.4$ - $4.6$",
              r"FPS + $T_0$, z = $2.2$ - $4.6$"]
    full_corner(chain_dirs, "allp_corner24.pdf", labels=labels)
    #For proposal
    chain_dirs = ["chains/fps-only/mf-48-z2.6-4.6",
                  "chains/fps-only/mf-48-z2.2-4.6"]
    labels = [r"z = $2.6$ - $4.6$",
              r"z = $2.2$ - $4.6$"]
    cosmo_corner(chain_dirs, "cosmo_corner_proposal.pdf", labels=labels)
    #Main results
    chain_dirs = ["chains/fps-only/mf-48-z2.2-4.6",
                  "chains/fps-only/mf-48-z2.6-4.6",
                  "chains/fps-meant/mf-48-48-z2.6-4.6"]
    labels = [r"FPS z = $2.2$ - $4.6$",
              r"FPS z = $2.6$ - $4.6$",
              r"FPS + $T_0$, z = $2.6$ - $4.6$"]
    print_latex_table(chain_dirs, labels=labels)
    plot_correlation(correl_file = "correlation-z26-46-t0.txt")
    full_corner(chain_dirs, "allp_corner.pdf", labels=labels)
    cosmo_corner(chain_dirs, "cosmo_corner.pdf", labels=labels)
    astro_corner(chain_dirs, "astro_corner.pdf", labels=labels)
    plot_fps_obs_pred(basedir, chain_dirs, traindir=traindir, HRbasedir=basedir+'/hires', savefile="fps_data_fit.pdf", labels=labels)
    plot_t0_obs_pred(basedir, chain_dirs, HRbasedir=basedir+'/hires', savefile="t0_best_fit.pdf", labels=labels)
    plot_1d_marginals(basedir, chain_dirs, savefile="all_1d_marginals.pdf", labels=labels)
    plot_err_dists(basedir, basedir+"/loo_fps.hdf5", chain_dirs, traindir=traindir, savefile="all_1d_best_fit.pdf", labels=labels)
    plot_err_dists(basedir, basedir+"/loo_fps.hdf5", ["chains/fps-meant/mf-48-48-z2.6-4.6", "chains/fps-meant/mf-48-48-z2.6-4.6-gpemu"], traindir=traindir, savefile="loo_vs_emu_error_wlegend.pdf", labels=["Posterior", "GPERR Posterior"])
    #2 percent CV plot
    chain_dirs = ["chains/fps-only/mf-48-z2.6-4.6",
                  "chains/fps-only/mf-48-z2.2-4.6",
                  "chains/fps-only/mf-48-z2.6-4.6-cv02",
                  "chains/fps-only/mf-48-z2.2-4.6-cv02"]
    labels = [r"DR14 z = $2.6$ - $4.6$",
              r"DR14 z = $2.2$ - $4.6$",
              r"CV 2\% z = $2.6$ - $4.4$",
              r"CV 2\% z = $2.2$ - $4.4$"]
    #print_latex_table(chain_dirs, labels=labels)
    full_corner(chain_dirs, "cv02_allp_corner.pdf", labels=labels)
    cosmo_corner(chain_dirs, "cv02_cosmo_corner.pdf", labels=labels)
    astro_corner(chain_dirs, "cv02_astro_corner.pdf", labels=labels)
    #DR9 plot
    chain_dirs = ["chains/fps-only/mf-48-z2.6-4.6",
                  "chains/fps-only/mf-48-z2.2-4.6",
                  "chains/fps-only/mf-48-dr9-z2.6-4.4",
                  "chains/fps-only/mf-48-dr9-z2.2-4.4"]
    labels = [r"DR14 z = $2.6$ - $4.6$",
              r"DR14 z = $2.2$ - $4.6$",
              r"DR9 z = $2.6$ - $4.4$",
              r"DR9 z = $2.2$ - $4.4$"]
    #print_latex_table(chain_dirs, labels=labels)
    full_corner(chain_dirs, "dr9_allp_corner.pdf", labels=labels)
    cosmo_corner(chain_dirs, "dr9_cosmo_corner.pdf", labels=labels)
    astro_corner(chain_dirs, "dr9_astro_corner.pdf", labels=labels)
    #MeanT only plot
    chain_dirs = ["chains/meant-only/bpdf-48-emu",
                  "chains/meant-only/curvature-48-emu",
                  "chains/meant-only/fps-48-emu",
                  "chains/meant-only/wavelet-48-emu"]
    labels = [r"BPDF",
              r"Curvature",
              r"FPS",
              r"Wavelet"]
    temp_corner(chain_dirs, "datasets_t0_corner.pdf", labels=labels)
    plot_t0_obs_pred(basedir, chain_dirs, HRbasedir=basedir+'/hires', savefile="datasets_t0_best_fit.pdf", labels=labels)
