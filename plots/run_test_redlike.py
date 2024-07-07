# Tests the reduced (2D Gaussian) likelihood.
from lyaemu import redlikelihood as rlk

##### Likelihood function checks
rlike = rlk.ReducedLymanAlpha()
like.do_sampling(savefile='chains/reduced-me')
like.do_sampling(savefile='chains/reduced-chab', chabanier=True)
