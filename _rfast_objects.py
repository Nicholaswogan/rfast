from dataclasses import dataclass

import numpy as np
import rfast_routines as rtns
import rfast_atm_routines as atm_rtns

from _rfast_input import read_yaml_input

# Rfast base data class. The idea is to
# prevent the addition of attributes after
# initialization. This prevents bugs.
class RfastBaseClass:
  __isfrozen = False
  def __setattr__(self, key, value):
    if self.__isfrozen and not hasattr(self, key):
      raise TypeError( "%r is a frozen class" % self )
    object.__setattr__(self, key, value)

  def _freeze(self):
    self.__isfrozen = True
  
class GasParams(RfastBaseClass):
  def __init__(self, bg):
    self.Ngas, \
    self.gasid, \
    self.mmw0, \
    self.ray0, \
    self.nu0, \
    self.mb, \
    self.rayb = atm_rtns.set_gas_info(bg)
    
    self._freeze()

class RfastInputs(RfastBaseClass):
  def __init__(self, filename_scr):
    
    if filename_scr.endswith('.yaml'):
      # This generates all variables
      data = read_yaml_input(filename_scr)
      for key, val in data.items():
        exec("self."+key + ' = val')
    else:
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
      
    self._freeze()
      
      
RETRIEVABLE_PARAMS = \
np.array([
  'f0', 
  'pmax', 
  'Rp', 
  'Mp', 
  'gp', 
  'As', 
  'pt', 
  'dpc', 
  'tauc0', 
  'fc', 
  't0', 
  'a',
  'alpha', 
  'mb', 
  'rayb'
])

GENSPEC_INPUTS = \
np.array([
  'f0', 
  'pmax', 
  'Rp', 
  'Mp', 
  'gp', 
  'As', 
  'pt', 
  'dpc', 
  'tauc0', 
  'fc', 
  't0', 
  'a',
  'gc',
  'wc',
  'Qc',
  'alpha', 
  'mb', 
  'rayb'
])
      