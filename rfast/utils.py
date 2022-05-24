import numpy as np
from scipy import optimize
from scipy import special
import warnings

def rho(sigma):
    return 1 - special.erf(sigma/np.sqrt(2))

def bayes_factor(sigma):
    r = rho(sigma)
    return - 1/(np.exp(1)*r*np.log(r))

def objective(x, bayes_factor_input):
    sigma = x[0]
    return bayes_factor(sigma) - bayes_factor_input

def sigma_significance(bayes_factor_input):
    if bayes_factor_input > 1e8:
        warnings.warn("Bayes factors larger than 1e8 can not be computed. Returning sigma = 6.392455915996625")
        return 6.392455915996625
    if bayes_factor_input <= 1:
        return 0.9004526284839545
    initial_cond = np.array([6.0])
    sol = optimize.root(objective, initial_cond, args = (bayes_factor_input,))
    if not sol.success:
        raise Exception("Root solving failed: "+sol.message)
    return sol.x[0]

def detection_sigma(lnB):
    """Computes detection sigma from bayes factor.

    Parameters
    ----------
    lnB : float
        The natural log of the bayes factor

    Returns
    -------
    float
        Detection "sigma" significance.

    """
    if lnB < np.log(2e1):
        return sigma_significance(np.exp(lnB))
    
    logp = np.arange(-100.00,-0.00,.01) #reverse order
    logp = logp[::-1] # original order
    P = 10.0**logp
    Barr = -1./(np.exp(1)*P*np.log(P))

    sigma = np.arange(0.1,100.10,.01)
    p_p = special.erfc(sigma/np.sqrt(2.0))

    B = np.exp(lnB)
    pvalue = 10.0**np.interp(np.log10(B),np.log10(Barr),np.log10(P))
    sig = np.interp(pvalue,p_p[::-1],sigma[::-1])

    return sig