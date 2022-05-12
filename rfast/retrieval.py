import warnings
import ast

import numpy as np
import numba as nb
from scipy import stats
import astropy as ap

from .objects import RfastBaseClass
from .objects import GENSPEC_INPUTS, RETRIEVABLE_PARAMS

############################
### Retrieval parameters ###
############################

class RetrieveParams(RfastBaseClass):

    def __init__(self, scr, rpars_txt):
        # read input table
        tab = ap.io.ascii.read(rpars_txt, data_start=1, delimiter='|')
        par = np.array(tab['col1'])
        lab = np.array(tab['col2'])
        ret = np.array(tab['col3'])
        log = np.array(tab['col4'])
        shp = np.array(tab['col5'])
        p1 = np.array(tab['col6'])
        p2 = np.array(tab['col7'])

        # 3 relevent numbers
        # number of gases being retrieved (ngas)
        # number of total parameters being retrieved (nret)
        # nret - ngas + 1 = ngenspec

        # nret
        param_names = []
        param_labels = []

        log_space = []
        gauss_prior = []
        p1_n = []
        p2_n = []

        # ngas
        gas_names = []
        gas_inds = []

        # ngenspec
        genspec_names = []
        genspec_inds = []

        # number of retrieved, logged, and gas parameters
        # check that retrieved gases are active; check for retrieved Mp and gp (a big no-no!)
        nret = 0
        nlog = 0
        ngas = 0
        nlgas = 0
        mf = False
        gf = False
        for i in range(par.size):
            if (ret[i].lower() == 'y'):

                param_names.append(par[i].lower())
                param_labels.append(ast.literal_eval(lab[i]))

                nret = nret + 1
                if (par[i] == 'Mp'):
                    mf = True
                elif (par[i] == 'gp'):
                    gf = True

                if log[i].lower() == 'log':
                    nlog = nlog + 1
                    log_space.append(True)
                elif log[i].lower() == 'lin':
                    log_space.append(False)
                else:
                    raise Exception('"' + log[i] + '" is not an option.')

                if shp[i].lower() == 'g':
                    gauss_prior.append(True)
                elif shp[i].lower() == 'f':
                    gauss_prior.append(False)
                else:
                    raise Exception('"' + shp[i] + '" is not an option.')

                p1_n.append(p1[i])
                p2_n.append(p2[i])

                if (par[i][0] == 'f' and par[i] != 'fc'):
                    gas_names.append(par[i][1:].lower())
                    tmp = np.where(scr.species_r == par[i][1:].lower())[0]
                    if tmp.size != 1:
                        raise Exception(
                            'Gas "' + par[i][1:].lower() + '" can not be retrieved')
                    else:
                        gas_inds.append(tmp[0])

                    ngas = ngas + 1
                    if (log[i].lower() == 'log'):
                        nlgas = nlgas + 1
                    if (len(scr.species_r) <= 1):
                        if not (scr.species_r == par[i][1:].lower()):
                            raise Exception(
                                "rfast warning | major | requested retrieved gas is not radiatively active; ", par[i][1:].lower())
                    else:
                        if not any(scr.species_r == par[i][1:].lower()):
                            raise Exception(
                                "rfast warning | major | requested retrieved gas is not radiatively active; ", par[i][1:].lower())
                else:
                    # not a gas
                    genspec_names.append(par[i])
                    tmp = np.where(GENSPEC_INPUTS == par[i])[0]
                    tmp1 = np.where(RETRIEVABLE_PARAMS == par[i])[0]
                    if tmp1.size != 1:
                        raise Exception(
                            'Parameter "' + par[i] + '" can not be retrieved')
                    else:
                        genspec_inds.append(tmp[0])

        # we add mixing ratios if gases are retrieved
        if ngas > 0:
            genspec_names.insert(0, "f0")
            genspec_inds.insert(0, 0)
            ngenspec = nret - ngas + 1
        else:
            ngenspec = nret

        # warning if no parameters are retrieved
        if (nret == 0):
            raise Exception(
                "rfast warning | major | zero requested retrieved parameters")

        # warning that you cannot retrieve on both Mp and gp
        if (mf and gf):
            warnings.warn(
                "rfast warning | major | cannot retrieve on both Mp and gp")

        # warning if clr retrieval is requested but no gases are retrieved
        if (scr.clr and ngas == 0):
            warnings.warn(
                "rfast warning | minor | center-log retrieval requested but no retrieved gases")

        # warning that clr treatment assumes gases retrieved in log space
        if (scr.clr and ngas != nlgas):
            warnings.warn(
                "rfast warning | minor | requested center-log retrieval transforms all gas constraints to log-space")

        # warning if clr retrieval and number of included gases is smaller than retrieved gases
        if (scr.clr and ngas < len(scr.f0)):
            raise Exception(
                "rfast warning | major | center-log retrieval functions only if len(f0) equals number of retrieved gases")

        # set data attributes

        # actual parameters that are being retrieved
        self.nret = nret
        self.param_names = np.array(param_names, str)
        self.param_labels = np.array(param_labels, str)

        # Determine if parameters are being retrieved.
        # If they are, then we will save their index relative to self.param_names
        self.retrieving_pt = 'pt' in param_names
        if self.retrieving_pt:
            self.pt_ind = param_names.index('pt')
        else:
            self.pt_ind = None

        self.retrieving_dpc = 'dpc' in param_names
        if self.retrieving_dpc:
            self.dpc_ind = param_names.index('dpc')
        else:
            self.dpc_ind = None

        self.retrieving_pmax = 'pmax' in param_names
        if self.retrieving_pmax:
            self.pmax_ind = param_names.index('pmax')
        else:
            self.pmax_ind = None

        self.retrieving_gp = "gp" in param_names
        if self.retrieving_gp:
            self.gp_ind = param_names.index('gp')
        else:
            self.gp_ind = None

        # length nret
        self.gauss_prior = np.array(gauss_prior)
        self.log_space = np.array(log_space)
        self.p1 = np.array(p1_n)
        self.p2 = np.array(p2_n)

        # Gases
        self.ngas = ngas
        self.gas_names = np.array(gas_names, str)
        self.gas_inds = np.array(gas_inds)  # indexs correspond to r.scr.f0

        # genspec inputs
        self.ngenspec = ngenspec
        self.genspec_names = np.array(genspec_names, str)
        # indexs correspond to r.scr_genspec_inputs
        self.genspec_inds = np.array(genspec_inds)

        # if clr retrieval stuff
        self.ximin = None
        self.ximax = None
        if scr.clr:
            n = len(scr.f0) + 1
            self.ximin = (n - 1.) / n * (np.log(scr.fmin) -
                                         np.log((1. - scr.fmin) / (n - 1.)))
            self.ximax = (n - 1) / n * \
                (np.log(1 - n * scr.fmin) - np.log(scr.fmin))

        # no new attributes
        self._freeze()

###############################
### Transforming parameters ###
###############################

@nb.njit()
def transform_parameters(x, log_space, clr, ng, fmin):
    x_t = np.empty(x.size)

    if not clr:
        for i in range(x.size):
            if log_space[i]:
                x_t[i] = np.log10(x[i])
            else:
                x_t[i] = x[i]
    else:
        gx = np.exp(
            (np.sum(np.log(x[:ng])) + np.log(max(fmin, 1 - np.sum(x[:ng])))) / (len(x[:ng]) + 1))
        x_t[:ng] = np.log(x[:ng] / gx)
        for i in range(ng, x_t.size):
            if log_space[i]:
                x_t[i] = np.log10(x[i])
            else:
                x_t[i] = x[i]

    return x_t


@nb.njit()
def untransform_parameters(x_t, log_space, clr, ng):
    x = np.empty(x_t.size)

    if not clr:
        for i in range(x.size):
            if log_space[i]:
                x[i] = 10.0**x_t[i]
            else:
                x[i] = x_t[i]
    else:
        clrs = np.sum(np.exp(x_t[:ng])) + np.exp(-np.sum(x_t[:ng]))
        x[:ng] = np.exp(x_t[:ng]) / clrs
        for i in range(ng, x_t.size):
            if log_space[i]:
                x[i] = 10.0**x_t[i]
            else:
                x[i] = x_t[i]

    return x

#####################
### MCMC sampling ###
#####################

def lnlike(r, x, f0, dat, err):
    retrieval = r.retrieval
    scr = r.scr

    # ngas_params
    x0_params = [f0] + list(x[retrieval.ngas:])

    # we must build up inputs to genspec. Big list
    # x0 = [f0,pmax,Rp,Mp,gp,As,pt,dpc,tauc0,fc,t0,a,gc,wc,Qc,alpha,mb,rayb]
    x0 = r.scr_genspec_inputs.copy()

    for i in range(len(x0_params)):
        ind = retrieval.genspec_inds[i]
        x0[ind] = x0_params[i]

    # if gravity (gp) is not being retrieved,
    # then we must overwrite the scr_genspec_inputs value
    if not retrieval.retrieving_gp:
        x0[4] = -1

    # call the forward model
    F_out = r.genspec_x(x0, degrade_F1=False)

    return -0.5 * (np.sum((dat - F_out)**2 / err**2))


def lnprior(r, x_t, x, f0, pt, dpc, pmax):
    retrieval = r.retrieval
    scr = r.scr

    # if center-log prior, then we shift index to only consider
    # non-gas paramters
    if scr.clr:
        prior_start = retrieval.ngas
    else:
        prior_start = 0

    # sum gausian priors
    lng = 0.0
    for i in range(prior_start, retrieval.nret):
        if retrieval.gauss_prior[i]:
            # -(1/2)*((x-mu)/sigma)**2
            # mu == p1 is center of distribution
            # sigma == p2 is the standard deviation
            lng += - 0.5 * (x[i] - retrieval.p1[i])**2 / retrieval.p2[i]**2

    # cloud base pressure
    if scr.cld:
        pb = pt + dpc
    else:
        pb = -1

    # prior limits
    within_explicit_priors = True
    for i in range(prior_start, retrieval.nret):
        if not retrieval.p1[i] <= x[i] <= retrieval.p2[i]:
            within_explicit_priors = False
            break

    if scr.clr:
        for i in range(retrieval.ngas):
            if not retrieval.ximin <= x_t[i] <= retrieval.ximax:
                within_explicit_priors = False
                break

    within_implicit_priors = True
    if not scr.clr:
        if not np.sum(f0) <= 1.0:
            within_implicit_priors = False
    elif not pb <= pmax:
        within_implicit_priors = False

    within_priors = within_explicit_priors and within_implicit_priors

    if within_priors:
        out = lng
    else:
        out = -np.inf

    return out

def lnprob(x_t, r, dat, err):
    x = untransform_parameters(
        x_t, r.retrieval.log_space, r.scr.clr, r.retrieval.ngas)

    # Make a copy of mixing ratios in the input file.
    # replace elements of array with retrieved gases
    f0 = r.scr.f0.copy()
    f_gases = x[:r.retrieval.ngas]
    f0[r.retrieval.gas_inds] = f_gases
    # we also need to grab some cloud parameters,
    # but only if they are actually being retrieved
    if r.retrieval.retrieving_pt:
        pt = x[r.retrieval.pt_ind]
    else:
        pt = r.scr.pt
    if r.retrieval.retrieving_dpc:
        dpc = x[r.retrieval.dpc_ind]
    else:
        dpc = r.scr.dpc
    if r.retrieval.retrieving_pmax:
        pmax = x[r.retrieval.pmax_ind]
    else:
        pmax = r.scr.pmax

    lp = lnprior(r, x_t, x, f0, pt, dpc, pmax)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(r, x, f0, dat, err)

#######################
### Nested Sampling ###
#######################

def lnlike_nest(x_t, r, dat, err):
    retrieval = r.retrieval
    scr = r.scr
    
    x = untransform_parameters(
        x_t, retrieval.log_space, scr.clr, retrieval.ngas)

    # Make a copy of mixing ratios in the input file.
    # replace elements of array with retrieved gases
    f0 = scr.f0.copy()
    f_gases = x[:r.retrieval.ngas]
    f0[retrieval.gas_inds] = f_gases
    # we also need to grab some cloud parameters,
    # but only if they are actually being retrieved
    if retrieval.retrieving_pt:
        pt = x[r.retrieval.pt_ind]
    else:
        pt = r.scr.pt
    if retrieval.retrieving_dpc:
        dpc = x[r.retrieval.dpc_ind]
    else:
        dpc = r.scr.dpc
    if retrieval.retrieving_pmax:
        pmax = x[r.retrieval.pmax_ind]
    else:
        pmax = r.scr.pmax
    
    # enforce implicit priors here.
    within_implicit_priors = True
    if not np.sum(f0) <= 1.0:
        within_implicit_priors = False
    if scr.cld:
        pb = pt + dpc
        if not pb <= pmax:
            within_implicit_priors = False
        
    if within_implicit_priors:
        out = lnlike(r, x, f0, dat, err)
    else:
        out = -1.0e100
    
    return out

# functions for converting a uniform distribution
# to various different distributions
def quantile_to_gauss(quantile, mu, sigma):
    return stats.norm.ppf(quantile,loc=mu,scale=sigma)
    
def quantile_to_uniform(quantile, lower_bound, upper_bound):
    return quantile*(upper_bound - lower_bound) + lower_bound

def prior_transform(u, r):
    retrieval = r.retrieval
    scr = r.scr
    
    # initialize output array
    x = np.empty(u.size)
    
    # Transform all retrieved parameters
    for i in range(retrieval.nret):
        
        # priors are always saved in linear space (p1, p2)
        # so we must transform them to log10 space if
        # necessary
        if retrieval.gauss_prior[i]:
            if retrieval.log_space[i]:
                mu = np.log10(retrieval.p1[i])
                sigma = np.log10(retrieval.p2[i])
            else:
                mu = retrieval.p1[i]
                sigma = retrieval.p2[i]
            
            x[i] = quantile_to_gauss(u[i], mu, sigma)
        else:
            if retrieval.log_space[i]:
                lower_bound = np.log10(retrieval.p1[i])
                upper_bound = np.log10(retrieval.p2[i])
            else:
                lower_bound = retrieval.p1[i]
                upper_bound = retrieval.p2[i]
                
            x[i] = quantile_to_uniform(u[i], lower_bound, upper_bound)
            
    # We enforce sum(f0) <= 1 and the cloud base stuff in the log-likelihood function
                
    return x