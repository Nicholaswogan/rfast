import warnings

import numpy as np
import numba as nb
import astropy as ap
import rfast_routines as rtns
import rfast_atm_routines as atm_rtns
import rfast_opac_routines as opac_rtns

# main Rfast class
class Rfast():
  def __init__(self, scr_file, Nres=3):
    # Read input scr file
    self.scr = RfastInputs(scr_file)

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
    x = scr.f0, scr.pmax, scr.Rp, scr.Mp, scr.gp, scr.As, scr.pt, scr.dpc, \
        scr.tauc0, scr.fc, scr.t0, scr.a, self.gc, self.wc, self.Qc, scr.alpha, \
        self.gparams.mb, self.gparams.rayb
    F1_hr, F2_hr = self._genspec_x_hr(x, rdtmp=scr.rdgas, rdgas=scr.rdgas)
    return F1_hr, F2_hr

  def _genspec_x_hr(self, x, rdtmp=False, rdgas=False):
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
    scr = self.scr
    
    # read input table
    tab  = ap.io.ascii.read(rpars_txt,data_start=1,delimiter='|')
    par  = tab['col1']
    lab  = tab['col2']
    ret  = tab['col3']
    log  = tab['col4']
    shp  = tab['col5']
    p1   = tab['col6']
    p2   = tab['col7']
    
    # number of read-in parameters
    npar = par.shape[0]

    # number of retrieved, logged, and gas parameters
    # check that retrieved gases are active; check for retrieved Mp and gp (a big no-no!)
    log_space = []
    nret  = 0
    nlog  = 0
    ngas  = 0
    nlgas = 0
    mf    = False
    gf    = False
    for i in range(npar):
      if (ret[i].lower() == 'y'):
        nret = nret + 1
        if (par[i] == 'Mp'):
          mf  = True
        if (par[i] == 'gp'):
          gf  = True
        if (log[i].lower() == 'log'):
          nlog = nlog + 1
          log_space.append(True)
        else:
          log_space.append(False)
        if (par[i][0] == 'f' and par[i] != 'fc'):
          ngas = ngas + 1
          if (log[i].lower() == 'log'):
            nlgas = nlgas + 1
          if (len(scr.species_r) <= 1):
            if not (scr.species_r == par[i][1:].lower()):
              raise Exception("rfast warning | major | requested retrieved gas is not radiatively active; ",par[i][1:].lower())
          else:
            if not any(scr.species_r == par[i][1:].lower()):
              raise Exception("rfast warning | major | requested retrieved gas is not radiatively active; ",par[i][1:].lower())
              
    log_space = np.array(log_space)
      
    # warning if no parameters are retrieved
    if (nret == 0):
      raise Exception("rfast warning | major | zero requested retrieved parameters")
    
    # warning that you cannot retrieve on both Mp and gp
    if (mf and gf):
      warnings.warn("rfast warning | major | cannot retrieve on both Mp and gp")

    # warning if clr retrieval is requested but no gases are retrieved
    if (scr.clr and ngas == 0):
      warnings.warn("rfast warning | minor | center-log retrieval requested but no retrieved gases")

    # warning that clr treatment assumes gases retrieved in log space
    if (scr.clr and ngas != nlgas):
      warnings.warn("rfast warning | minor | requested center-log retrieval transforms all gas constraints to log-space")

    # warning if clr retrieval and number of included gases is smaller than retrieved gases
    if (scr.clr and ngas < len(scr.f0)):
      raise Exception("rfast warning | major | center-log retrieval functions only if len(f0) equals number of retrieved gases")
    
    
    # Need to know what parameters are being retrieved, out the parameters 
    # that are considered in the model.
    
    

# Data objects for convenient storage

class GasParams():
  def __init__(self, bg):
    self.Ngas, \
    self.gasid, \
    self.mmw0, \
    self.ray0, \
    self.nu0, \
    self.mb, \
    self.rayb = atm_rtns.set_gas_info(bg)

class RfastInputs():
  def __init__(self, filename_scr):
    self.fnr, \
    self.fnn, \
    self.fns, \
    self.dirout, \
    self.Nlev, \
    self.pmin, \
    self.pmax, \
    self.bg,\
    self.species_r, \
    self.f0, \
    self.rdgas, \
    self.fnatm, \
    self.skpatm, \
    self.colr, \
    self.colpr, \
    self.psclr, \
    self.imix,\
    self.t0, \
    self.rdtmp, \
    self.fntmp, \
    self.skptmp, \
    self.colt, \
    self.colpt, \
    self.psclt,\
    self.species_l, \
    self.species_c,\
    self.lams, \
    self.laml, \
    self.res, \
    self.modes, \
    self.regrid, \
    self.smpl, \
    self.opdir,\
    self.Rp, \
    self.Mp, \
    self.gp, \
    self.a, \
    self.As, \
    self.em,\
    self.grey, \
    self.phfc, \
    self.w, \
    self.g1, \
    self.g2, \
    self.g3, \
    self.pt, \
    self.dpc, \
    self.tauc0, \
    self.lamc0, \
    self.fc,\
    self.ray, \
    self.cld, \
    self.ref, \
    self.sct, \
    self.fixp, \
    self.pf, \
    self.fixt, \
    self.tf, \
    self.p10, \
    self.fp10,\
    self.src,\
    self.alpha, \
    self.ntg,\
    self.Ts, \
    self.Rs,\
    self.ntype, \
    self.snr0, \
    self.lam0, \
    self.rnd,\
    self.clr, \
    self.fmin, \
    self.mmr, \
    self.nwalkers, \
    self.nstep, \
    self.nburn, \
    self.nprocess, \
    self.thin, \
    self.restart, \
    self.progress = rtns.inputs(filename_scr)

    # interpret mixing ratios as mass or volume, based on user input
    self.mmr = False
    if (self.imix == 1):
      self.mmr = True

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
def untransform_parameters(x, log_space, clr, ng):
  x_t = np.empty(x.size)
  
  if not clr:
    for i in range(x.size):
      if log_space[i]:
        x_t[i] = 10.0**x[i]
      else:
        x_t[i] = x[i]
  else:
    clrs =  np.sum(np.exp(x[:ng])) + np.exp(-np.sum(x[:ng]))
    x_t[:ng] = np.exp(x[:ng])/clrs
    for i in range(ng, x.size):
      if log_space[i]:
        x_t[i] = 10.0**x[i]
      else:
        x_t[i] = x[i]
        
  return x_t
  
# def lnprob(x, args):
#   r = args[0] # rfast object
#   # tranform parameters to normal space
#   x_t = untransform_parameters(x, r)
# 
#   # compute prior prior
#   lp = lnprior(x, r)
#   if not np.isfinite(lp):
#     return -np.inf
#   return lp + lnlike(x, r)
  
