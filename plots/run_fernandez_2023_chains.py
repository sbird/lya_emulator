# main chains run for "Inference via the Lyman-a Forest"
# run using: mpiexec -n 4 --bind-to core python3 chain_script.py
# ON LAPTOP: INCLUDE --bind-by core ARGUMENT SO THAT IT DOESN'T TRY TO USE ALL CORES

# meant only chains should run quickly (a few minutes)
# chains including the flux power can take much longer (several hours each)

import numpy as np
import sys
# append path to lya_emulator_full
# sys.path.append('path/to/emulator/code/')
from lyaemu import likelihood as lk
from lyaemu.meanT import t0_likelihood as t0lk
basedir = '../dtau-48-48'
chaindir = 'chains'

####################
######### meant only
####################
# only meant with no priors (fps)
like = t0lk.T0LikelihoodClass(basedir, max_z=3.8, min_z=2.2, optimise_GP=False, HRbasedir=basedir+'/hires', dataset='fps', loo_errors=False)
like.do_sampling(savefile=chaindir+'/meant-only/fps-48-emu', burnin=1e4, nsamples=3e5, hprior='none', oprior=False, bhprior=False, dataset='fps')

# now meant only with other datasets
like = t0lk.T0LikelihoodClass(basedir, max_z=3.8, min_z=2.2, optimise_GP=False, HRbasedir=basedir+'/hires', dataset='bpdf', loo_errors=False)
like.do_sampling(savefile=chaindir+'meant-only/bpdf-48-emu', burnin=1e4, nsamples=3e5, hprior='none', oprior=False, bhprior=False, dataset='bpdf')

like = t0lk.T0LikelihoodClass(basedir, max_z=3.8, min_z=2.2, optimise_GP=False, HRbasedir=basedir+'/hires', dataset='wavelet', loo_errors=False)
like.do_sampling(savefile=chaindir+'meant-only/wavelet-48-emu', burnin=1e4, nsamples=3e5, hprior='none', oprior=False, bhprior=False, dataset='wavelet')

like = t0lk.T0LikelihoodClass(basedir, max_z=3.8, min_z=2.2, optimise_GP=False, HRbasedir=basedir+'/hires', dataset='curvature', loo_errors=False)
like.do_sampling(savefile=chaindir+'meant-only/curvature-48-emu', burnin=1e4, nsamples=3e5, hprior='none', oprior=False, bhprior=False, dataset='curvature')


##################
######### FPS only
##################
# main fps only runs
like = lk.LikelihoodClass(basedir, tau_thresh=1e6, optimise_GP=False, traindir=basedir+'/trained_mf', HRbasedir=basedir+'/hires', loo_errors=True, min_z=2.2, max_z=4.6)
like.do_sampling(savefile=chaindir+'fps-only/mf-48-z2.2-4.6', burnin=1e4, nsamples=3e5, pscale=100, include_emu_error=True, use_meant=False, hprior='none', oprior=False, bhprior=False)

like = lk.LikelihoodClass(basedir, tau_thresh=1e6, optimise_GP=False, traindir=basedir+'/trained_mf', HRbasedir=basedir+'/hires', loo_errors=True, min_z=2.6, max_z=4.6)
like.do_sampling(savefile=chaindir+'fps-only/mf-48-z2.6-4.6', burnin=1e4, nsamples=3e5, pscale=100, include_emu_error=True, use_meant=False, hprior='none', oprior=False, bhprior=False)

like = lk.LikelihoodClass(basedir, tau_thresh=1e6, optimise_GP=False, traindir=basedir+'/trained_mf', HRbasedir=basedir+'/hires', loo_errors=False, min_z=2.2, max_z=4.6)
like.do_sampling(savefile=chaindir+'fps-only/mf-48-z2.2-4.6-emuerr', burnin=1e4, nsamples=6e5, pscale=100, include_emu_error=True, use_meant=False, hprior='none', oprior=False, bhprior=False)

like = lk.LikelihoodClass(basedir, tau_thresh=1e6, optimise_GP=False, traindir=basedir+'/trained_mf', HRbasedir=basedir+'/hires', loo_errors=False, min_z=2.6, max_z=4.6)
like.do_sampling(savefile=chaindir+'fps-only/mf-48-z2.6-4.6-emuerr', burnin=1e4, nsamples=3e5, pscale=100, include_emu_error=True, use_meant=False, hprior='none', oprior=False, bhprior=False)


# increase BOSS error --- REQUIRES MANUALLY ADDING MULTIPLYING FACTOR TO BOSS ERRORS IN THE CODE
# like = lk.LikelihoodClass(basedir, tau_thresh=1e6, optimise_GP=False, traindir=basedir+'/trained_mf', HRbasedir=basedir+'/hires', loo_errors=False, min_z=2.2, max_z=4.6)
# like.do_sampling(savefile=chaindir+'fps-only/mf-48-2xboss', burnin=1e4, nsamples=3e5, pscale=80, include_emu_error=True, use_meant=False, hprior='none', oprior=False, bhprior=False)


# DR9 checks
like = lk.LikelihoodClass(basedir, tau_thresh=1e6, optimise_GP=False, traindir=basedir+'/trained_mf', HRbasedir=basedir+'/hires', loo_errors=False, min_z=2.6, max_z=4.4, sdss="dr9")
like.do_sampling(savefile=chaindir+'fps-only/mf-48-dr9-z2.6-4.4', burnin=1e4, nsamples=3e5, pscale=100, include_emu_error=True, use_meant=False, hprior='none', oprior=False, bhprior=False)

like = lk.LikelihoodClass(basedir, tau_thresh=1e6, optimise_GP=False, traindir=basedir+'/trained_mf', HRbasedir=basedir+'/hires', loo_errors=False, min_z=2.2, max_z=4.4, sdss="dr9")
like.do_sampling(savefile=chaindir+'fps-only/mf-48-dr9-z2.2-4.4', burnin=1e4, nsamples=3e5, pscale=100, include_emu_error=True, use_meant=False, hprior='none', oprior=False, bhprior=False)




##################
######### FPS + T0
##################
# meant_fac for various options
mtf_fullz_loo = 8.38
mtf_fullz_emu = 8.38
mtf_partz_loo = 8.07
mtf_partz_emu = 8.03

# main chains
like = lk.LikelihoodClass(basedir, tau_thresh=1e6, optimise_GP=False, traindir=basedir+'/trained_mf', HRbasedir=basedir+'/hires', loo_errors=True, min_z=2.6, max_z=4.6, use_meant=True)
like.do_sampling(savefile=chaindir+'fps-meant/mf-48-48-z2.6-4.6', burnin=1e4, nsamples=3e5, pscale=100, include_emu_error=True, use_meant=True, hprior='none', oprior=False, bhprior=False, meant_fac=mtf_partz_loo)

like = lk.LikelihoodClass(basedir, tau_thresh=1e6, optimise_GP=False, traindir=basedir+'/trained_mf', HRbasedir=basedir+'/hires', loo_errors=False, min_z=2.6, max_z=4.6, use_meant=True)
like.do_sampling(savefile=chaindir+'fps-meant/mf-48-48-z2.6-4.6-emuerr', burnin=1e4, nsamples=3e5, pscale=100, include_emu_error=True, use_meant=True, hprior='none', oprior=False, bhprior=False, meant_fac=mtf_partz_emu)

like = lk.LikelihoodClass(basedir, tau_thresh=1e6, optimise_GP=False, traindir=basedir+'/trained_mf', HRbasedir=basedir+'/hires', loo_errors=True, min_z=2.2, max_z=4.6, use_meant=True)
like.do_sampling(savefile=chaindir+'fps-meant/mf-48-48-z2.2-4.6', burnin=1e4, nsamples=3e5, pscale=100, include_emu_error=True, use_meant=True, hprior='none', oprior=False, bhprior=False, meant_fac=mtf_fullz_loo)

like = lk.LikelihoodClass(basedir, tau_thresh=1e6, optimise_GP=False, traindir=basedir+'/trained_mf', HRbasedir=basedir+'/hires', loo_errors=False, min_z=2.2, max_z=4.6, use_meant=True)
like.do_sampling(savefile=chaindir+'fps-meant/mf-48-48-z2.2-4.6-emuerr', burnin=1e4, nsamples=3e5, pscale=100, include_emu_error=True, use_meant=True, hprior='none', oprior=False, bhprior=False, meant_fac=mtf_fullz_emu)


# including priors
like = lk.LikelihoodClass(basedir, tau_thresh=1e6, optimise_GP=False, traindir=basedir+'/trained_mf', HRbasedir=basedir+'/hires', loo_errors=False, min_z=2.6, max_z=4.6, use_meant=True)
like.do_sampling(savefile=chaindir+'fps-meant/mf-48-48-z2.6-4.6-priors', burnin=1e4, nsamples=3e5, pscale=100, include_emu_error=True, use_meant=True, hprior='none', oprior=True, bhprior=True, meant_fac=mtf_partz_emu)

like = lk.LikelihoodClass(basedir, tau_thresh=1e6, optimise_GP=False, traindir=basedir+'/trained_mf', HRbasedir=basedir+'/hires', loo_errors=False, min_z=2.2, max_z=4.6, use_meant=True)
like.do_sampling(savefile=chaindir+'fps-meant/mf-48-48-z2.2-4.6-priors', burnin=1e4, nsamples=3e5, pscale=100, include_emu_error=True, use_meant=True, hprior='none', oprior=True, bhprior=True, meant_fac=mtf_fullz_emu)


##### Likelihood function checks
# Using a simulated HR dataset
like = lk.LikelihoodClass(basedir, tau_thresh=1e6, optimise_GP=False, traindir=basedir+'/trained_mf', HRbasedir=basedir+'/hires', loo_errors=False, min_z=2.2, max_z=4.6, use_meant=True)
#ns909 tau = 1.15
like.do_sampling(datadir=basedir+'/hires', data_index=21, savefile=chaindir+'/like-test/mf-48-48-z2.2-4.6', burnin=1e4, nsamples=3e5, pscale=100, include_emu_error=True, use_meant=True, hprior='none', oprior=False, bhprior=False, meant_fac=mtf_fullz_emu)
#Using a simulated LR dataset with a different seed
like = lk.LikelihoodClass(basedir, tau_thresh=1e6, optimise_GP=False, traindir=basedir+'/trained_mf', HRbasedir=None, loo_errors=False, min_z=2.2, max_z=4.6, use_meant=False)
#ns909 tau = 1.15
like.do_sampling(datadir=basedir+"/ns0.881-seed", data_index=0, savefile=chaindir+'/like-test/seed', burnin=1e4, nsamples=3e5, pscale=100, include_emu_error=True, use_meant=False, hprior='none', oprior=False, bhprior=False)
