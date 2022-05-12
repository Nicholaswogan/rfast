import warnings
import os
from multiprocessing import Pool
import pickle

import numpy as np
import numba as nb
from scipy import stats
import dynesty
import astropy as ap
import emcee
from multiprocess import Process

from . import routines as rtns
from . import atm_routines as atm_rtns
from . import opac_routines as opac_rtns

from . import retrieval as ret_funcs

from .objects import GasParams, RfastInputs, RfastBaseClass
from .objects import RETRIEVABLE_PARAMS
from .input import src_to_names

# main Rfast class

class Rfast(RfastBaseClass):
    def __init__(self, scr_file, Nres=3):
        # Read input scr file
        scr = RfastInputs(scr_file)
        self.scr = scr

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

        # genspec inputs that are from the input file
        self.scr_genspec_inputs = \
            [scr.f0, scr.pmax, scr.Rp, scr.Mp, scr.gp, scr.As, scr.pt, scr.dpc,
             scr.tauc0, scr.fc, scr.t0, scr.a, self.gc, self.wc, self.Qc, scr.alpha,
             self.gparams.mb, self.gparams.rayb]

        # attributes for retrieval things
        self.retrieval = None
        self.dat = None
        self.err = None
        self.retrieval_processes = []

        # prevent new attributes
        self._freeze()

    ###################################
    ### Spectra generation routines ###
    ###################################

    def genspec_scr(self, degrade_F1=True, omit_gases=None):
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
        F1_hr, F2_hr = self._genspec_x_hr(
            self.scr_genspec_inputs, rdtmp=scr.rdgas, rdgas=scr.rdgas)
        return F1_hr, F2_hr

    def _genspec_x_hr(self, x, rdtmp=False, rdgas=False):
        if not isinstance(x, list):
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
        if not (omit_gases_s.issubset(set(self.scr.species_l))
                or omit_gases_s.issubset(set(self.scr.species_c))):
            raise Exception('"omit_gases" contains elements which' +
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

    ########################
    ### noise generation ###
    ########################

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
                    erri = rtns.noise(lam0i, scr.snr0[i], self.lam,
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

    ######################
    ### Saving results ###
    ######################

    def write_raw_file(self, F1, F2, filename=None):
        # write raw file
        names = src_to_names(self.scr.src, is_noise=False)
        data_out = ap.table.Table([self.lam, self.dlam, F1, F2], names=names)
        if filename is None:
            filename = self.scr.dirout + self.scr.fns + '.raw'
        ap.io.ascii.write(data_out, filename,
                          format='fixed_width', overwrite=True)

    def write_dat_file(self, F1, F2, data, err, filename=None):
        # write data file
        names = src_to_names(self.scr.src, is_noise=True)
        data_out = ap.table.Table(
            [self.lam, self.dlam, F1, F2, data, err], names=names)
        if filename is None:
            filename = self.scr.dirout + self.scr.fns + '.dat'
        ap.io.ascii.write(data_out, filename,
                          format='fixed_width', overwrite=True)

    #################
    ### Retrieval ###
    #################
    
    def _lnprob(self, x_t):
        """log-probability function for MCMC
        """
        return ret_funcs.lnprob(x_t, self, self.dat, self.err)
    
    def _lnlike_nest(self, x_t):
        """log-likelihood function for nested sampling
        """
        return ret_funcs.lnlike_nest(x_t, self, self.dat, self.err)
     
    def _prior_transform(self, u):
        """prior transformation for nested sampling   
        """
        return ret_funcs.prior_transform(u, self)
    
    def initialize_retrieval(self, rpars_txt):
        """Initialize the retrieval with the retrieval input file.
        """
        # Set retrieval parameters
        self.retrieval = ret_funcs.RetrieveParams(self.scr, rpars_txt)
        
    def _initial_guess(self):
        """Computes initial guess of parameters for emcee retrieval.
        """
        retrieval = self.retrieval
        # check for initialization
        if retrieval is None:
            raise Exception(
                "You must first intialize a retrieval to call this function")
        # gases
        gas_guess = np.empty(retrieval.ngas)
        for i in range(retrieval.ngas):
            ind = retrieval.gas_inds[i]
            gas_guess[i] = self.scr.f0[ind]
        # parameters that are not gases
        param_guess = np.empty(retrieval.ngenspec - 1)
        for i in range(1, retrieval.ngenspec):
            ind = retrieval.genspec_inds[i]
            param_guess[i - 1] = self.scr_genspec_inputs[ind]
        # put them together, and transform
        guess = np.append(gas_guess, param_guess)
        guess_t = ret_funcs.transform_parameters(guess, retrieval.log_space,
                                       self.scr.clr, retrieval.ngas, self.scr.fmin)
        
        return guess, guess_t

    ### MCMC retrievals

    def prepare_emcee_retrieval(self, overwrite=False, h5_file=None):
        """prepares a emcee retrieval.
        """
        retrieval = self.retrieval

        # compute the initial guess
        guess, guess_t = self._initial_guess()
                                       
        if h5_file is None:
            h5_filename = self.scr.dirout + self.scr.fnr + '.h5'
        else:
            h5_filename = h5_file
        if not self.scr.restart:
            if os.path.isfile(h5_filename):
                if overwrite:
                    os.remove(h5_filename)
                else:
                    raise Exception(
                        "rfast warning | major | h5 file already exists")

            backend = emcee.backends.HDFBackend(h5_filename)
            backend.reset(self.scr.nwalkers, retrieval.nret)
            # initialize walkers as a cloud around guess
            pos = [guess_t + 1e-4 *
                   np.random.randn(retrieval.nret) for i in range(self.scr.nwalkers)]
        else:
            if not os.path.isfile(h5_filename):
                raise Exception(
                    "rfast warning | major | h5 does not exist for restart")
            else:
                # otherwise initialize walkers from existing backend
                backend = emcee.backends.HDFBackend(h5_filename)
                pos = backend.get_last_sample()

        if len(self.scr.nprocess) == 0:
            nprocess = os.cpu_count()
        else:
            nprocess = int(self.scr.nprocess)

        return backend, pos

    def _emcee_retrieve(self, dat, err, progress, overwrite, h5_file):
        backend, pos = self.prepare_emcee_retrieval(overwrite=overwrite, h5_file=h5_file)
        self.dat = dat
        self.err = err
        sampler = emcee.EnsembleSampler(self.scr.nwalkers, self.retrieval.nret, \
                                        self._lnprob, backend=backend)
        sampler.run_mcmc(pos, self.scr.nstep, progress=progress)

    def emcee_retrieve(self, dat, err, progress=False, overwrite=False, h5_file=None):
        self._emcee_retrieve(dat, err, progress, overwrite, h5_file)

    def emcee_process(self, dat, err, h5_file):
        p = Process(target=self._emcee_retrieve, args=(
            dat, err, False, False, h5_file))
        p.start()
        out = {}
        out['process'] = p
        out['file'] = h5_file
        self.retrieval_processes.append(out)
    
    ### nested retrievals
                        
    def prepare_nested_retrieval(self, dat, err, **kwargs):
        retrieval = self.retrieval
        # check for initialization
        if retrieval is None:
            raise Exception(
                "You must first intialize a retrieval before retrieving")
                
        # check input dat and var
        if not isinstance(dat, np.ndarray) or not isinstance(err, np.ndarray):
            raise ValueError("Input dat and var must be numpy ndarrays")

        if dat.size != self.lam.size or err.size != self.lam.size:
            raise ValueError("Input dat and var have the wrong size")
            
        self.dat = dat
        self.err = err
        
        sampler = dynesty.NestedSampler(self._lnlike_nest, self._prior_transform, retrieval.nret, \
                                               **kwargs)
        return sampler
        
    def nested_retrieve(self, dat, err, progress=False, overwrite=False, file=None, dlogz = 0.001, **kwargs):
        if file is None:
            filename = self.scr.dirout + self.scr.fnr + '.pkl'
        else:
            filename = file
            
        if os.path.isfile(filename):
            if overwrite:
                os.remove(filename)
            else:
                raise Exception("File already exists")
        
        sampler = self.prepare_nested_retrieval(dat, err, **kwargs)
        sampler.run_nested(dlogz=dlogz, print_progress = progress)
            
        with open(filename, 'wb') as f:
            pickle.dump(sampler.results, f)
            
    def nested_process(self, dat, err, file, dlogz = 0.001):
        p = Process(target=self.nested_retrieve, args=(
            dat, err, False, False, file, dlogz))
        p.start()
        out = {}
        out['process'] = p
        out['file'] = file
        self.retrieval_processes.append(out)
        
    def monitor_retrievals(self):
        """Monitors retrieval processes that are currently active
        """
        nprocess = len(self.retrieval_processes)
        if nprocess == 0:
            print('No retrievals have been started')
        else:
            for process in self.retrieval_processes:
                if process['process'].is_alive():
                    tmp = "Running..."
                else:
                    tmp = "Completed."
                print(process['file'] + ': ', tmp)
        
