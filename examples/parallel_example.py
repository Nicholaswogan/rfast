import numpy as np
import pickle
from multiprocessing import Pool
import dynesty
import rfast

# This examples demonstrates how to do a parallel retrieval.
# It also demonstrates how you can do a retrieval which omits
# a gas. This is useful for computing detection significances
# of gases.

# number of processes
NPROCESS = 4

# initialize radiative transfer and retrieval
r = rfast.Rfast('ModernEarth/rfast_inputs.scr')
r.initialize_retrieval("ModernEarth/rfast_rpars.txt")

# Make fake data of Modern Earth twin.
F1, F2 = r.genspec_scr()
dat, err = r.noise(F2)

# functions that need to be fed to nested sampler.
# we pass data as globals.
def lnlike(x_t):
    return rfast.lnlike_nest(x_t, r, dat, err)

def prior_transform(u):
    return rfast.prior_transform(u, r)
    
# make another instance with O2 removed.
r_noO2 = rfast.Rfast('ModernEarth/rfast_inputs.scr')
r_noO2.initialize_retrieval("ModernEarth/rfast_rpars.txt")
r_noO2.remove_gas('o2')
    
def lnlike_noO2(x_t):
    return rfast.lnlike_nest(x_t, r_noO2, dat, err)
    
def prior_transform_noO2(u):
    return rfast.prior_transform(u, r_noO2)
        
if __name__ == "__main__":
    
    # retrieval with all gases
    with Pool(NPROCESS) as pool:
        sampler = dynesty.NestedSampler(lnlike, prior_transform, r.retrieval.nret, \
                                        nlive = 1000, pool=pool, queue_size=NPROCESS)
        sampler.run_nested(dlogz=0.001)

    with open('nested_results.pkl', 'wb') as f:
        pickle.dump(sampler.results, f)
        
    # Another retrieval which does not retrieve O2
    with Pool(NPROCESS) as pool:
        sampler = dynesty.NestedSampler(lnlike_noO2, prior_transform_noO2, r_noO2.retrieval.nret, \
                                        nlive = 1000, pool=pool, queue_size=NPROCESS)
        sampler.run_nested(dlogz=0.001)

    with open('nested_results_noO2.pkl', 'wb') as f:
        pickle.dump(sampler.results, f)




