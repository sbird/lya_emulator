# Tests the reduced (2D Gaussian) likelihood.
from lyaemu import redlikelihood as rlk

##### Likelihood function checks
rlk.do_sampling(like="ReducedLymanAlpha", savefile='chains/reduced-me')
rlk.do_sampling(like="ChabReducedLymanAlpha", savefile='chains/reduced-chab')
rlk.do_sampling(like="PlanckReduced", savefile='chains/reduced-planck')
