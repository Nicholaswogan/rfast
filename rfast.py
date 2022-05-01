import warnings
import os

import numpy as np
import numba as nb
import astropy as ap
import emcee
from multiprocessing import Pool

import rfast_routines as rtns
import rfast_atm_routines as atm_rtns
import rfast_opac_routines as opac_rtns

from _rfast_objects import GasParams, RfastInputs, RfastBaseClass
from _rfast_objects import GENSPEC_INPUTS, RETRIEVABLE_PARAMS

# main Rfast class
class Rfast(RfastBaseClass):
  def __init__(self, scr_file, Nres=3):
    # Read input scr file
    scr = RfastInputs(scr_file)
    self.scr = scr
    if scr.clr:
      raise Exception("Center log stuff does not work")

    # set info for all radiatively active gases, including background gas
    self.gparams = GasParams(self.scr.bg)
    
    # lams, laml, res, modes, snr0, lam0
    # Inputs needed to regrid everything
    
    # generate wavelength grids
    lam, dlam = rtns.gen_spec_grid(self.scr.lams, self.scr.laml,
                                   np.float_(self.scr.res), Nres=0)
    lam_hr, dlam_hr = rtns.gen_spec_grid(self.scr.lams, self.scr.laml,
                                         np.float_(self.scr.res) *
                                         self.scr.smpl,
                                         Nres=np.rint(Nres * self.scr.smpl))
    mode = rtns.modes_to_mode(
        lam, self.scr.lams, self.scr.laml, self.scr.modes)

    # initialize opacities and convolution kernels
    sigma_interp, cia_interp, ncia, ciaid, kern = \
        rtns.init(lam, dlam, lam_hr, self.scr.species_l,
                  self.scr.species_c, self.scr.opdir, self.scr.pf, self.scr.tf, mode=mode)

    # initialize cloud asymmetry parameter, single scattering albedo, extinction efficiency
    gc, wc, Qc = opac_rtns.init_cloud_optics(lam_hr, self.scr.g1, self.scr.g2,
                                             self.scr.g3, self.scr.w, self.scr.lamc0, self.scr.grey, self.scr.cld, self.scr.opdir)

    # initialize disk integration quantities
    threeD = rtns.init_3d(self.scr.src, self.scr.ntg)

    # Save data for later use
    self.lam = lam
    self.dlam = dlam
    self.lam_hr = lam_hr
    self.dlam_ir = dlam_hr
    self.mode = mode
    self.sigma_interp = sigma_interp
    self.cia_interp = cia_interp
    self.ncia = ncia
    self.ciaid = ciaid
    self.kern = kern
    self.gc = gc
    self.wc = wc
    self.Qc = Qc
    self.threeD = threeD
    
    self.default_genspec_inputs = \
        [scr.f0, scr.pmax, scr.Rp, scr.Mp, scr.gp, scr.As, scr.pt, scr.dpc, \
         scr.tauc0, scr.fc, scr.t0, scr.a, self.gc, self.wc, self.Qc, scr.alpha, \
         self.gparams.mb, self.gparams.rayb]
    
    # attribute for retrieval things
    self.retrieval = None
    
    # prevent new attributes 
    self._freeze()

  def genspec_scr(self, degrade_F1=True, omit_gases = None):
    # we can omit gases
    if omit_gases is not None:
      return self._genspec_scr_omit_gases(degrade_F1, omit_gases)
      
    F1_hr, F2_hr = self._genspec_scr_hr()
    F2 = rtns.kernel_convol(self.kern, F2_hr)
    if degrade_F1:
      F1 = rtns.kernel_convol(self.kern, F1_hr)
      out = (F1, F2)
    else:
      out = F2
      
    return out
  
  def genspec_x(self, x, degrade_F1=True):
    F1_hr, F2_hr = self._genspec_x_hr(x)
    if degrade_F1:
      F1 = rtns.kernel_convol(self.kern, F1_hr)
      F2 = rtns.kernel_convol(self.kern, F2_hr)
      # "distance" scaling for thermal emission case
      if (self.scr.src == 'thrm'):
        Rp = x[2]
        F1 = F1 * (Rp / scr.Rp)**2  # SHOULD THIS LINE BE HERE?
        F2 = F2 * (Rp / scr.Rp)**2
      out = (F1, F2)
    else:
      F2 = rtns.kernel_convol(self.kern, F2_hr)
      # "distance" scaling for thermal emission case
      if (self.scr.src == 'thrm'):
        Rp = x[2]
        F2 = F2 * (Rp / scr.Rp)**2
      out = F2

    return out

  def _genspec_scr_hr(self):
    scr = self.scr
    F1_hr, F2_hr = self._genspec_x_hr(self.default_genspec_inputs, rdtmp=scr.rdgas, rdgas=scr.rdgas)
    return F1_hr, F2_hr

  def _genspec_x_hr(self, x, rdtmp=False, rdgas=False):
    if not isinstance(x,list):
      raise Exception("input 'x' to _genspec_x_hr must be a list")
    
    # unpack x
    f0, pmax, Rp, Mp, gp, As, pt, dpc, tauc0, fc, t0, a, gc, wc, Qc, alpha, mb, rayb = x

    # unpack classes
    scr = self.scr
    gparams = self.gparams

    # initialize atmospheric model
    p, t, z, grav, f, fb, m, nu = \
        atm_rtns.setup_atm(
            scr.Nlev,
            gparams.Ngas,
            gparams.gasid,
            gparams.mmw0,
            scr.pmin,
            pmax,
            t0,
            rdtmp,
            scr.fntmp,
            scr.skptmp,
            scr.colt,
            scr.colpt,
            scr.psclt,
            scr.species_r,
            f0,
            rdgas,
            scr.fnatm,
            scr.skpatm,
            scr.colr,
            scr.colpr,
            scr.psclr,
            scr.mmr,
            mb,
            Mp,
            Rp,
            scr.cld,
            pt,
            dpc,
            tauc0,
            scr.p10,
            scr.fp10,
            scr.src,
            scr.ref,
            gparams.nu0,
            gp=gp
        )
    # scale cloud optical depths based on extinction efficiency
    if scr.cld:
      tauc = tauc0 * Qc
    else:
      tauc = np.zeros(len(scr.lam_hr))

    # call forward model
    F1_hr, F2_hr = \
        rtns.gen_spec(
            scr.Nlev,
            Rp,
            a,
            As,
            scr.em,
            p,
            t,
            t0,
            m,
            z,
            grav,
            scr.Ts,
            scr.Rs,
            scr.ray,
            gparams.ray0,
            rayb,
            f,
            fb,
            gparams.mmw0,
            scr.mmr,
            scr.ref,
            nu,
            alpha,
            self.threeD,
            gparams.gasid,
            self.ncia,
            self.ciaid,
            scr.species_l,
            scr.species_c,
            scr.cld,
            scr.sct,
            scr.phfc,
            fc,
            pt,
            dpc,
            gc,
            wc,
            tauc,
            scr.src,
            self.sigma_interp,
            self.cia_interp,
            self.lam_hr,
            pf=scr.pf,
            tf=scr.tf
        )

    return F1_hr, F2_hr
    
  def _genspec_scr_omit_gases(self, degrade_F1, omit_gases):
    """Computes a spectrum but omits a gas"""
    
    # Check type
    if not isinstance(omit_gases, list):
      raise Exception('"omit_gases" must be a list')
    # Check for duplicates
    omit_gases_s = set(omit_gases)
    if len(omit_gases_s) != len(omit_gases):
      raise Exception('"omit_gases" contains duplicates.')
    # Check that it is a subset
    if not (omit_gases_s.issubset(set(self.scr.species_l)) \
            or omit_gases_s.issubset(set(self.scr.species_c))):
      raise Exception('"omit_gases" contains elements which'+ \
                      ' are not in the list of radiatively active species')
                    
    species_l = []
    for i in range(len(self.scr.species_l)):
      if self.scr.species_l[i] not in omit_gases:
        species_l.append(self.scr.species_l[i])
    species_c = []
    for i in range(len(self.scr.species_c)):
      if self.scr.species_c[i] not in omit_gases:
        species_c.append(self.scr.species_c[i])
    species_l = np.array(species_l) 
    species_c = np.array(species_c)
    
    # Save things that will change
    species_l_s = self.scr.species_l.copy()
    species_c_s = self.scr.species_c.copy()
    sigma_interp_s = self.sigma_interp
    cia_interp_s = self.cia_interp
    ncia_s = self.ncia.copy()
    ciaid_s = self.ciaid.copy()
    kern_s = self.kern.copy()
    
    # Overwrite with new stuff
    self.scr.species_l = species_l
    self.scr.species_c = species_c
    self.sigma_interp, self.cia_interp, \
    self.ncia, self.ciaid, self.kern = \
        rtns.init(self.lam, self.dlam, self.lam_hr, species_l,
                  species_c, self.scr.opdir, self.scr.pf, self.scr.tf, mode=self.mode)
    
    # compute spectrum
    F1_hr, F2_hr = self._genspec_scr_hr()
    F2 = rtns.kernel_convol(self.kern, F2_hr)
    if degrade_F1:
      F1 = rtns.kernel_convol(self.kern, F1_hr)
      out = (F1, F2)
    else:
      out = F2
    
    # return variables to original state
    self.scr.species_l = species_l_s
    self.scr.species_c = species_c_s
    self.sigma_interp = sigma_interp_s 
    self.cia_interp = cia_interp_s
    self.ncia = ncia_s
    self.ciaid = ciaid_s
    self.kern = kern_s
    
    return out

  def noise(self, F2):
    scr = self.scr
    # vectors of lam0 and snr0 to handle wavelength dependence
    lam0v = np.zeros(len(self.lam))
    snr0v = np.zeros(len(self.lam))

    # snr0 constant w/wavelength case
    if(len(scr.snr0) == 1):
      if (scr.ntype != 'cppm'):
        err = rtns.noise(scr.lam0, scr.snr0, self.lam,
                         self.dlam, F2, scr.Ts, scr.ntype)
      else:
        err = np.zeros(F2.shape[0])
        err[:] = 1 / snr0v
    else:  # otherwise snr0 is bandpass dependent
      err = np.zeros(len(self.lam))
      for i in range(len(scr.snr0)):
        ilam = np.where(np.logical_and(
            self.lam >= scr.lams[i], self.lam <= scr.laml[i]))
        if (len(scr.lam0) == 1):  # lam0 may be bandpass dependent
          lam0i = scr.lam0
        else:
          lam0i = scr.lam0[i]
        if (scr.ntype != 'cppm'):
          erri = noise(lam0i, scr.snr0[i], self.lam,
                       self.dlam, F2, scr.Ts, scr.ntype)
          err[ilam] = erri[ilam]
        else:
          err[ilam] = 1 / scr.snr0[i]

    # generate faux spectrum, with random noise if requested
    data = np.copy(F2)
    if scr.rnd:
      for k in range(len(self.lam)):
        data[k] = np.random.normal(F2[k], err[k], 1)
        if data[k] < 0:
          data[k] = 0.

    return data, err

  def write_raw_file(self, F1, F2):
    # write data file
    names = src_to_names(self.scr.src, is_noise=False)
    data_out = ap.table.Table([self.lam, self.dlam, F1, F2], names=names)
    ap.io.ascii.write(data_out, self.scr.dirout + self.scr.fns +
                      '.raw', format='fixed_width', overwrite=True)

  def write_dat_file(self, F1, F2, data, err):
    # write data file
    names = src_to_names(self.scr.src, is_noise=True)
    data_out = ap.table.Table(
        [self.lam, self.dlam, F1, F2, data, err], names=names)
    ap.io.ascii.write(data_out, self.scr.dirout + self.scr.fns +
                      '.dat', format='fixed_width', overwrite=True)
                        
  # retrieval
  def initialize_retrieval(self, rpars_txt):
    # Set retrieval parameters
    self.retrieval = RetrieveParams(self.scr, rpars_txt)
    
  def retrieve(self, dat, err, overwrite = False):
    retrieval = self.retrieval
    # check for initialization
    if retrieval is None:
      raise Exception("You must first intialize a retrieval before retrieving")
    
    # check input dat and var
    if not isinstance(dat, np.ndarray) or not isinstance(err, np.ndarray):
      raise ValueError("Input dat and var must be numpy ndarrays")
    
    if dat.size != self.lam.size or err.size != self.lam.size:
      raise ValueError("Input dat and var have the wrong size")
    
    # compute the initial guess
    # gases
    gas_guess = np.empty(retrieval.nret_gas)
    for i in range(retrieval.nret_gas):
      ind = retrieval.ret_gas_inds[i]
      gas_guess[i] = self.scr.f0[ind]
    
    # parameters that are not gases
    param_guess = np.empty(retrieval.nret_param - 1)
    for i in range(1,retrieval.nret_param):
      ind = retrieval.ret_param_inds[i]
      param_guess[i-1] = self.default_genspec_inputs[ind]
    
    # put them together, and transform
    guess = np.append(gas_guess, param_guess)
    guess_t = transform_parameters(guess, retrieval.log_space, \
              self.scr.clr, retrieval.nret_gas, self.scr.fmin)

    ndim = len(guess_t)
    h5_filename = self.scr.dirout+self.scr.fnr+'.h5'
    if not self.scr.restart:
      if os.path.isfile(h5_filename):
        if overwrite:
          os.remove(h5_filename)
        else:
          raise Exception("rfast warning | major | h5 file already exists")
          
      backend  = emcee.backends.HDFBackend(h5_filename)
      backend.reset(self.scr.nwalkers, ndim)
      # initialize walkers as a cloud around guess
      pos = [guess_t + 1e-4*np.random.randn(ndim) for i in range(self.scr.nwalkers)]
    else:
      if not os.path.isfile(h5_filename):
        raise Exception("rfast warning | major | h5 does not exist for restart")
      else:
        # otherwise initialize walkers from existing backend
        backend = emcee.backends.HDFBackend(h5_filename)
        pos = backend.get_last_sample()
        
    if len(self.scr.nprocess) == 0:
      nprocess = os.cpu_count()
    else:
      nprocess = int(self.scr.nprocess)
      
    # arguments
    args = (self, dat, err,)
    
    with Pool(nprocess) as pool:
      sampler = emcee.EnsembleSampler(self.scr.nwalkers, ndim, lnprob, backend=backend, pool=pool, args = args)
      sampler.run_mcmc(pos, self.scr.nstep, progress=self.scr.progress)


class RetrieveParams(RfastBaseClass):

  def __init__(self, scr, rpars_txt):
    # read input table
    tab  = ap.io.ascii.read(rpars_txt,data_start=1,delimiter='|')
    par = np.array(tab['col1'])
    lab = np.array(tab['col2'])
    ret = np.array(tab['col3'])
    log = np.array(tab['col4'])
    shp = np.array(tab['col5'])
    p1 = np.array(tab['col6'])
    p2 = np.array(tab['col7'])
      
    # 3 relevent numbers
    # number of gases being retrieved (nret_gas)
    # number of total parameters being retrieved (nret)
    # nret - nret_gas + 1 = nret_param
    log_space = [] # nret
    gauss_prior = []
    p1_n = []
    p2_n = []
    param_name_inds = {}
    
    # nret_gas
    ret_gas_names = []
    ret_gas_inds = []
    
    # nret_param
    ret_param_names = [] 
    ret_param_inds = []

    # number of retrieved, logged, and gas parameters
    # check that retrieved gases are active; check for retrieved Mp and gp (a big no-no!)
    nret  = 0
    nlog  = 0
    nret_gas  = 0
    nlgas = 0
    mf    = False
    gf    = False
    for i in range(par.size):
      if (ret[i].lower() == 'y'):
        
        param_name_inds[par[i].lower()] = nret
  
        nret = nret + 1
        if (par[i] == 'Mp'):
          mf  = True
        elif (par[i] == 'gp'):
          gf  = True
        
        if log[i].lower() == 'log':
          nlog = nlog + 1
          log_space.append(True)
        elif log[i].lower() == 'lin':
          log_space.append(False)
        else:
          raise Exception('"'+log[i]+'" is not an option.')
          
        if shp[i].lower() == 'g':
          gauss_prior.append(True)
        elif shp[i].lower() == 'f':
          gauss_prior.append(False)
        else:
          raise Exception('"'+shp[i]+'" is not an option.')
          
        p1_n.append(p1[i])
        p2_n.append(p2[i])
      
        if (par[i][0] == 'f' and par[i] != 'fc'):
          ret_gas_names.append(par[i][1:].lower())
          tmp = np.where(scr.species_r==par[i][1:].lower())[0]
          if tmp.size != 1:
            raise Exception('Gas "'+par[i][1:].lower()+'" can not be retrieved')
          else:
            ret_gas_inds.append(tmp[0])

          nret_gas = nret_gas + 1
          if (log[i].lower() == 'log'):
            nlgas = nlgas + 1
          if (len(scr.species_r) <= 1):
            if not (scr.species_r == par[i][1:].lower()):
              raise Exception("rfast warning | major | requested retrieved gas is not radiatively active; ",par[i][1:].lower())
          else:
            if not any(scr.species_r == par[i][1:].lower()):
              raise Exception("rfast warning | major | requested retrieved gas is not radiatively active; ",par[i][1:].lower())
        else:
          # not a gas
          ret_param_names.append(par[i])
          tmp = np.where(GENSPEC_INPUTS==par[i])[0]
          tmp1 = np.where(RETRIEVABLE_PARAMS==par[i])[0]
          if tmp1.size != 1:
            raise Exception('Parameter "'+par[i]+'" can not be retrieved')
          else:
            ret_param_inds.append(tmp[0])
    
    # we add mixing ratios if gases are retrieved
    if nret_gas > 0:
      ret_param_names.insert(0,"f0")
      ret_param_inds.insert(0,0)
      nret_param = nret - nret_gas + 1
    else:
      nret_param = nret
    
    # warning if no parameters are retrieved
    if (nret == 0):
      raise Exception("rfast warning | major | zero requested retrieved parameters")
    
    # warning that you cannot retrieve on both Mp and gp
    if (mf and gf):
      warnings.warn("rfast warning | major | cannot retrieve on both Mp and gp")

    # warning if clr retrieval is requested but no gases are retrieved
    if (scr.clr and nret_gas == 0):
      warnings.warn("rfast warning | minor | center-log retrieval requested but no retrieved gases")

    # warning that clr treatment assumes gases retrieved in log space
    if (scr.clr and nret_gas != nlgas):
      warnings.warn("rfast warning | minor | requested center-log retrieval transforms all gas constraints to log-space")

    # warning if clr retrieval and number of included gases is smaller than retrieved gases
    if (scr.clr and nret_gas < len(scr.f0)):
      raise Exception("rfast warning | major | center-log retrieval functions only if len(f0) equals number of retrieved gases")
    
    # set data attributes
    self.nret = nret
    self.gauss_prior = np.array(gauss_prior)
    self.log_space = np.array(log_space)
    self.param_name_inds = param_name_inds
    self.p1 = np.array(p1_n)
    self.p2 = np.array(p2_n)
    
    self.nret_gas = nret_gas
    self.ret_gas_names = np.array(ret_gas_names)
    self.ret_gas_inds = np.array(ret_gas_inds)
    self.nret_param = nret_param
    self.ret_param_names = np.array(ret_param_names)
    self.ret_param_inds = np.array(ret_param_inds)
    
    # no new attributes
    self._freeze()


# Utility functions

def src_to_names(src, is_noise=False):
  if (src == 'diff' or src == 'cmbn'):
    names = ['wavelength (um)', 'd wavelength (um)', 'albedo', 'flux ratio']
  elif (src == 'thrm'):
    names = ['wavelength (um)', 'd wavelength (um)',
             'Tb (K)', 'flux (W/m**2/um)']
  elif (src == 'scnd'):
    names = ['wavelength (um)', 'd wavelength (um)', 'Tb (K)', 'flux ratio']
  elif (src == 'trns'):
    names = ['wavelength (um)', 'd wavelength (um)',
             'zeff (m)', 'transit depth']
  elif (src == 'phas'):
    names = ['wavelength (um)', 'd wavelength (um)', 'reflect', 'flux ratio']

  if is_noise:
    names = names + ['data', 'uncertainty']

  return names
  
# MCMC function things

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
    gx = np.exp((np.sum(np.log(x[:ng])) + np.log(max(fmin, 1-np.sum(x[:ng]))))/(len(x[:ng]) + 1))
    x_t[:ng] = np.log(x[:ng]/gx)
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
    clrs =  np.sum(np.exp(x_t[:ng])) + np.exp(-np.sum(x_t[:ng]))
    x[:ng] = np.exp(x_t[:ng])/clrs
    for i in range(ng, x_t.size):
      if log_space[i]:
        x[i] = 10.0**x_t[i]
      else:
        x[i] = x_t[i]
        
  return x
  
def lnlike(x, r, f0, dat, err):
  retrieval = r.retrieval
  scr = r.scr

  # ngas_params
  x0_params = [f0]+list(x[retrieval.nret_gas:])
  
  # we must build up inputs to genspec. Big list
  # x0 = [f0,pmax,Rp,Mp,gp,As,pt,dpc,tauc0,fc,t0,a,gc,wc,Qc,alpha,mb,rayb]
  x0 = r.default_genspec_inputs.copy()
  
  for i in range(len(x0_params)):
    ind = retrieval.ret_param_inds[i]
    x0[ind] = x0_params[i]
  
  # call the forward model
  F_out = r.genspec_x(x0, degrade_F1=False)
  
  return -0.5*(np.sum((dat-F_out)**2/err**2))

# @nb.njit()  
def lnprior(x, f0, pt, dpc, pmax, cld, gauss_prior, p1, p2):
  # still need to impliment center log stuff
  
  # sum gausian priors
  lng = 0.0
  for i in range(len(gauss_prior)):
    if gauss_prior[i]:
      lng -= 0.5*(x[i] - p1[i])**2/p2[i]**2
  
  # cloud base pressure
  if cld:
    pb = pt + dpc
  else:
    pb = -1
  
  # prior limits
  within_explicit_priors = True
  for i in range(len(x)):
    if not p1[i] <= x[i] <= p2[i]:
      within_explicit_priors = False
      break
  
  within_implicit_priors = True
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
  x = untransform_parameters(x_t, r.retrieval.log_space, r.scr.clr, r.retrieval.nret_gas)
  
  # Make a copy of mixing ratios in the input file.
  # replace elements of array with retrieved gases
  f0 = r.scr.f0.copy()
  f_gases = x[:r.retrieval.nret_gas]
  f0[r.retrieval.ret_gas_inds] = f_gases
  # we also need to grab some cloud parameters,
  # but only if they are actually being retrieved
  if 'pt' in r.retrieval.param_name_inds:
    pt = x[r.retrieval.param_name_inds['pt']]
  else:
    pt = r.scr.pt
  if 'dpc' in r.retrieval.param_name_inds:
    dpc = x[r.retrieval.param_name_inds['dpc']]
  else:
    dpc = r.scr.dpc
  if 'pmax' in r.retrieval.param_name_inds:
    pmax = x[r.retrieval.param_name_inds['pmax']]
  else:
    pmax = r.scr.pmax

  lp = lnprior(x, f0, pt, dpc, pmax, r.scr.cld, r.retrieval.gauss_prior, r.retrieval.p1, r.retrieval.p2)
  if not np.isfinite(lp):
    return -np.inf
  return lp + lnlike(x, r, f0, dat, err)
  
  
  
