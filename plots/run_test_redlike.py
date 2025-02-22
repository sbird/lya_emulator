# Tests the reduced (2D Gaussian) likelihood.
from lyaemu import redlikelihood as rlk

##### Likelihood function checks
rlike = rlk.ReducedLymanAlpha()
rlike.do_sampling(savefile='chains/reduced-me')
rlike = rlk.ChabReducedLymanAlpha()
rlike.do_sampling(savefile='chains/reduced-chab')
