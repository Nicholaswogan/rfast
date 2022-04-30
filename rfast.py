import numpy as np
import rfast_routines as rtns
import rfast_atm_routines as atm_rtns
import rfast_opac_routines as opac_rtns
import astropy as ap

# main Rfast class
class Rfast():
    def __init__(self, scr_file, Nres = 3):
        # obtain input parameters from script
        self.scr = RfastInputs(scr_file)
        
        # set info for all radiatively active gases, including background gas
        self.gparams = GasParams(self.scr.bg)
        
        # generate wavelength grids
        lam, dlam = rtns.gen_spec_grid(self.scr.lams, self.scr.laml, \
                                  np.float_(self.scr.res), Nres=0)
        lam_hr, dlam_hr = rtns.gen_spec_grid(self.scr.lams, self.scr.laml, \
                                        np.float_(self.scr.res)*self.scr.smpl, \
                                        Nres=np.rint(Nres*self.scr.smpl))                                
        mode = rtns.modes_to_mode(lam, self.scr.lams, self.scr.laml, self.scr.modes)

        # initialize opacities and convolution kernels
        sigma_interp, cia_interp, ncia, ciaid, kern = \
        rtns.init(lam, dlam, lam_hr, self.scr.species_l, \
        self.scr.species_c, self.scr.opdir, self.scr.pf, self.scr.tf, mode=mode)
        
        # initialize cloud asymmetry parameter, single scattering albedo, extinction efficiency
        gc, wc, Qc = opac_rtns.init_cloud_optics(lam_hr, self.scr.g1, self.scr.g2, \
        self.scr.g3, self.scr.w, self.scr.lamc0, self.scr.grey, self.scr.cld, self.scr.opdir)

        # initialize disk integration quantities
        threeD = rtns.init_3d(self.scr.src, self.scr.ntg)
        
        # Save data for later use
        self.lam = lam
        self.dlam = dlam
        self.lam_hr = lam_hr
        self.dlam_ir = dlam_hr
        self.sigma_interp = sigma_interp
        self.cia_interp = cia_interp
        self.ncia = ncia
        self.ciaid = ciaid
        self.kern = kern
        self.gc = gc
        self.wc = wc
        self.Qc = Qc
        self.threeD = threeD
        
    def genspec_scr(self, degrade_F1 = True):
        F1_hr, F2_hr = self._genspec_scr_hr()
        F2 = rtns.kernel_convol(self.kern, F2_hr)
        if degrade_F1:
            F1 = rtns.kernel_convol(self.kern, F1_hr)
            return (F1, F2)
        else:
            return F2
            
    def genspec_x(self, x, degrade_F1 = True):
        F1_hr, F2_hr = self._genspec_x_hr(x)
        if degrade_F1:
            F1 = rtns.kernel_convol(self.kern, F1_hr)
            F2 = rtns.kernel_convol(self.kern, F2_hr)
            # "distance" scaling for thermal emission case
            if (self.scr.src == 'thrm'):
                Rp = x[2]
                F1 = F1*(Rp/scr.Rp)**2 # SHOULD THIS LINE BE HERE?
                F2 = F2*(Rp/scr.Rp)**2
            out = (F1, F2)
        else:
            F2 = rtns.kernel_convol(self.kern, F2_hr)
            # "distance" scaling for thermal emission case
            if (self.scr.src == 'thrm'):
                Rp = x[2]
                F2 = F2*(Rp/scr.Rp)**2
            out = F2
            
        return out
        
    def _genspec_scr_hr(self):
        scr = self.scr
        x = scr.f0, scr.pmax, scr.Rp, scr.Mp, scr.gp, scr.As, scr.pt, scr.dpc, \
        scr.tauc0, scr.fc, scr.t0, scr.a, self.gc, self.wc, self.Qc, scr.alpha, \
        self.gparams.mb, self.gparams.rayb
        F1_hr, F2_hr = self._genspec_x_hr(x, rdtmp = scr.rdgas, rdgas = scr.rdgas)
        return F1_hr, F2_hr 
        
    def _genspec_x_hr(self, x, rdtmp = False, rdgas = False):
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
            gp = gp
        )
        # scale cloud optical depths based on extinction efficiency
        if scr.cld:
          tauc = tauc0*Qc
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

    def noise(self, F2):
        scr = self.scr
        # vectors of lam0 and snr0 to handle wavelength dependence
        lam0v = np.zeros(len(self.lam))
        snr0v = np.zeros(len(self.lam))
        
        # snr0 constant w/wavelength case
        if( len(scr.snr0) == 1 ):
            if (scr.ntype != 'cppm'):
                err = rtns.noise(scr.lam0, scr.snr0, self.lam, self.dlam, F2, scr.Ts, scr.ntype)
            else:
                err    = np.zeros(F2.shape[0])
                err[:] = 1/snr0v
        else: # otherwise snr0 is bandpass dependent
            err = np.zeros(len(self.lam))
            for i in range(len(scr.snr0)):
                ilam = np.where(np.logical_and(self.lam >= scr.lams[i], self.lam <= scr.laml[i]))
                if (len(scr.lam0) == 1): # lam0 may be bandpass dependent
                    lam0i = scr.lam0
                else:
                    lam0i = scr.lam0[i]
                if (scr.ntype != 'cppm'):
                    erri = noise(lam0i, scr.snr0[i], self.lam, self.dlam, F2, scr.Ts, scr.ntype)
                    err[ilam] = erri[ilam]
                else:
                    err[ilam] = 1/scr.snr0[i]

        # generate faux spectrum, with random noise if requested
        data = np.copy(F2)
        if scr.rnd:
            for k in range(len(self.lam)):
                data[k]  = np.random.normal(F2[k], err[k], 1)
                if data[k] < 0:
                    data[k] = 0.
                    
        return data, err
 
    def write_raw_file(self, F1, F2):    
        # write data file
        names = src_to_names(self.scr.src, is_noise = False)
        data_out = ap.table.Table([self.lam, self.dlam, F1, F2], names=names)
        ap.io.ascii.write(data_out, self.scr.dirout+self.scr.fns+'.raw', format='fixed_width', overwrite=True)
    
    def write_dat_file(self, F1, F2, data, err):    
        # write data file
        names = src_to_names(self.scr.src, is_noise = True)
        data_out = ap.table.Table([self.lam, self.dlam, F1, F2, data, err], names=names)
        ap.io.ascii.write(data_out, self.scr.dirout+self.scr.fns+'.dat', format='fixed_width', overwrite=True)
            
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
    def __init__(self,filename_scr):
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
def src_to_names(src, is_noise = False):
    if (src == 'diff' or src == 'cmbn'):
        names = ['wavelength (um)','d wavelength (um)','albedo','flux ratio']
    elif (src == 'thrm'):
        names = ['wavelength (um)','d wavelength (um)','Tb (K)','flux (W/m**2/um)']
    elif (src == 'scnd'):
        names = ['wavelength (um)','d wavelength (um)','Tb (K)','flux ratio']
    elif (src == 'trns'):
        names = ['wavelength (um)','d wavelength (um)','zeff (m)','transit depth']
    elif (src == 'phas'):
        names = ['wavelength (um)','d wavelength (um)','reflect','flux ratio']
    
    if is_noise:
        names = names + ['data','uncertainty']
        
    return names
     




