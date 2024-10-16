"""
JAX-adaptation of https://github.com/planck-npipe/hillipop/tree/master.
First class without desilike / cobaya dependency (would be good to have in the github above),
then desilike-specific wrapping.
"""

import os
import re
import logging
import itertools

import numpy as np
import fitsio

from desilike.likelihoods.base import BaseLikelihood
from desilike.jax import numpy as jnp
from .base import ClTheory, projection


t_cmb = 2.72548
k_b = 1.3806503e-23
h_pl = 6.626068e-34

# ------------------------------------------------------------------------------------------------
# Foreground class
# ------------------------------------------------------------------------------------------------
class fgmodel(object):
    """
    Class of foreground model for the Hillipop likelihood
    Units: Dl in muK^2
    Should return the model in Dl for a foreground emission given the parameters for all correlation of frequencies
    """
    logger = logging.getLogger('fgmodel')

    #reference frequency for residuals amplitudes
    f0 = 143

    # Planck effective frequencies
    fsz    = {100:100.24, 143: 143, 217: 222.044}
    fdust  = {100:105.2, 143:147.5, 217:228.1, 353:370.5} #alpha=4 from [Planck 2013 IX]
    fcib   = fdust
    fsyn   = {100:100,143:143,217:217}
    fradio = {100:100.4,143:140.5,217:218.6}

    def _f_tsz( self, freq):
        # Freq in GHz
        nu = freq*1e9
        xx=h_pl*nu/(k_b*t_cmb)
        return xx*( 1/np.tanh(xx/2.) ) - 4

    def _f_Planck( self, f, T):
        # Freq in GHz
        nu = f*1e9
        xx  = h_pl*nu /(k_b*T)
        return (nu**3.)/(np.exp(xx)-1.)

    #Temp Antenna conversion
    def _dBdT(self, f):
        # Freq in GHz
        nu  = f*1e9
        xx  = h_pl*nu /(k_b*t_cmb)
        return (nu)**4 * np.exp(xx) / (np.exp(xx)-1.)**2.

    def _tszRatio( self, f, f0):
        return self._f_tsz(f)/self._f_tsz(f0)

    def _cibRatio( self, f, f0, beta=1.75, T=25):
        return (f/f0)**beta * (self._f_Planck(f,T)/self._f_Planck(f0,T)) / ( self._dBdT(f)/self._dBdT(f0) )

    def _dustRatio( self, f, f0, beta=1.5, T=19.6):
        return (f/f0)**beta * (self._f_Planck(f,T)/self._f_Planck(f0,T)) / ( self._dBdT(f)/self._dBdT(f0) )

    def _radioRatio( self, f, f0, beta=-0.7):
        return (f/f0)**beta / ( self._dBdT(f)/self._dBdT(f0) )

    def _syncRatio( self, f, f0, beta=-0.7):
        return (f/f0)**beta / ( self._dBdT(f)/self._dBdT(f0) )

    def __init__(self, lmax, freqs, mode="TT", auto=False, **kwargs):
        """
        Create model for foreground
        """
        self.mode = mode
        self.lmax = lmax
        self.freqs = freqs
        self.name = None

        ell = np.arange(lmax + 1)
        self.ll2pi = ell * (ell + 1) / (3000*3001)

        # Build the list of cross frequencies
        self._cross_frequencies = list(
            itertools.combinations_with_replacement(freqs, 2)
            if auto
            else itertools.combinations(freqs, 2)
        )

    def _gen_dl_powerlaw( self, alpha, lnorm=3000):
        """
        Generate power-law Dl template
        Input: alpha in Cl
        """
        lmax = self.lmax if lnorm is None else max(self.lmax,lnorm)
        ell = np.arange( 2, lmax+1)

        template = np.zeros( lmax+1)
        template[ell] = ell*(ell+1)/2/np.pi * ell**(alpha)

        #normalize l=3000
        if lnorm is not None:
            template = template / template[lnorm]

        return template[:self.lmax+1]

    def _read_dl_template( self, filename, lnorm=3000):
        """
        Read FG template (in Dl, muK^2)
        WARNING: need to check file before reading...
        """
        #read dl template
        l,data = np.loadtxt( filename, unpack=True)
        l = np.array(l,int)
        self.logger.debug("Template: {}".format(filename))

        if max(l) < self.lmax:
            self.logger.info("WARNING: template {} has lower lmax (filled with 0)".format(filename))
        template = np.zeros( max(self.lmax,max(l)) + 1)
        template[l] = data

        #normalize l=3000
        if lnorm is not None:
            template = template / template[lnorm]

        return template[:self.lmax+1]

    def compute_dl(self, pars):
        """
        Return spectra model for each cross-spectra
        """
        pass
# ------------------------------------------------------------------------------------------------



# Subpixel effect
class subpix(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "SubPixel"
        self.fwhm = {100:9.68,143:7.30,217:5.02} #arcmin

    def compute_dl(self, pars):
        def _bl( fwhm):
            sigma = np.deg2rad(fwhm/60.) / np.sqrt(8.0 * np.log(2.0))
            ell = np.arange(self.lmax + 1)
            return np.exp(-0.5 * ell * (ell + 1) * sigma**2)

        dl_sbpx = []
        for f1, f2 in self._cross_frequencies:
            pxl = self.ll2pi / _bl( self.fwhm[f1]) / _bl( self.fwhm[f2])
            dl_sbpx.append( pars["Asbpx_{}x{}".format(f1,f2)] * pxl / pxl[2500] )

        if self.mode == "TT":
            return jnp.array(dl_sbpx)
        else:
            return 0.



# Point Sources
class ps(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "PS"

    def compute_dl(self, pars):
        dl_ps = []
        for f1, f2 in self._cross_frequencies:
            dl_ps.append( pars["Aps_{}x{}".format(f1,f2)] * self.ll2pi)

        if self.mode == "TT":
            return jnp.array(dl_ps)
        else:
            return 0.



# Radio Point Sources (v**alpha)
class ps_radio(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "PS radio"

    def compute_dl(self, pars):
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append(
                self.ll2pi
                * self._radioRatio( self.fradio[f1], self.f0, beta=pars['beta_radio'])
                * self._radioRatio( self.fradio[f2], self.f0, beta=pars['beta_radio'])
            )

        if self.mode == "TT":
            return pars["Aradio"] * jnp.array(dl)
        else:
            return 0.


# Infrared Point Sources
class ps_dusty(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "PS dusty"

    def compute_dl(self, pars):
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append(
                self.ll2pi
                * self._cibRatio( self.fcib[f1], self.f0, beta=pars['beta_dusty'])
                * self._cibRatio( self.fcib[f2], self.f0, beta=pars['beta_dusty'])
            )

        if self.mode == "TT":
            return pars["Adusty"] * jnp.array(dl)
        else:
            return 0.


# Galactic Dust
class dust(fgmodel):
    def __init__(self, lmax, freqs, filename=None, mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "Dust"

        self.dlg = []
        hdr = ["ell","100x100","100x143","100x217","143x143","143x217","217x217"]
        data = np.loadtxt( f"{filename}_{mode}.txt").T
        l = np.array(data[0],int)
        for f1, f2 in self._cross_frequencies:
            tmpl = np.zeros(max(l) + 1)
            tmpl[l] = data[hdr.index(f"{f1}x{f2}")]
            self.dlg.append( tmpl[:lmax+1])

        self.dlg = np.array(self.dlg)

    def compute_dl(self, pars):
        if self.mode == "TT":
            A = B = {100:pars["Ad100T"],143:pars["Ad143T"],217:pars["Ad217T"]}
        if self.mode == "EE":
            A = B = {100:pars["Ad100P"],143:pars["Ad143P"],217:pars["Ad217P"]}
        if self.mode == "TE":
            A = {100:pars["Ad100T"],143:pars["Ad143T"],217:pars["Ad217T"]}
            B = {100:pars["Ad100P"],143:pars["Ad143P"],217:pars["Ad217P"]}
        if self.mode == "ET":
            A = {100:pars["Ad100P"],143:pars["Ad143P"],217:pars["Ad217P"]}
            B = {100:pars["Ad100T"],143:pars["Ad143T"],217:pars["Ad217T"]}

        Ad = [A[f1]*B[f2] for f1, f2 in self._cross_frequencies]

        return jnp.array(Ad)[:, None] * self.dlg


class dust_model(fgmodel):
    def __init__(self, lmax, freqs, filename=None, mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "Dust model"

        self.dlg = []
        hdr = ["ell","100x100","100x143","100x217","143x143","143x217","217x217"]
        data = np.loadtxt( f"{filename}_{mode}.txt").T
        l = np.array(data[0],int)
        for f1, f2 in self._cross_frequencies:
            tmpl = np.zeros(max(l) + 1)
            tmpl[l] = data[hdr.index(f"{f1}x{f2}")]
            self.dlg.append( tmpl[:lmax+1])
        self.dlg = np.array(self.dlg)

    def compute_dl(self, pars):
        if   self.mode == "TT": beta1,beta2 = pars['beta_dustT'],pars['beta_dustT']
        elif self.mode == "TE": beta1,beta2 = pars['beta_dustT'],pars['beta_dustP']
        elif self.mode == "ET": beta1,beta2 = pars['beta_dustP'],pars['beta_dustT']
        elif self.mode == "EE": beta1,beta2 = pars['beta_dustP'],pars['beta_dustP']

        if   self.mode == "TT": ad1,ad2 = pars['AdustT'],pars['AdustT']
        elif self.mode == "TE": ad1,ad2 = pars['AdustT'],pars['AdustP']
        elif self.mode == "ET": ad1,ad2 = pars['AdustP'],pars['AdustT']
        elif self.mode == "EE": ad1,ad2 = pars['AdustP'],pars['AdustP']

        dl = []
        for xf, (f1, f2) in enumerate(self._cross_frequencies):
            dl.append( ad1 * ad2 * self.dlg[xf]
                       * self._dustRatio( self.fdust[f1], self.fdust[353], beta=beta1)
                       * self._dustRatio( self.fdust[f2], self.fdust[353], beta=beta2)
                       )
        return jnp.array(dl)


# Syncrothron model
class sync_model(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "Synchrotron"

        #check effective freqs
        for f in freqs:
            if f not in self.fsyn:
                raise ValueError( f"Missing SYNC effective frequency for {f}")

        alpha_syn = -2.5  #Cl template power-law TBC
        self.dl_syn = self._gen_dl_powerlaw( alpha_syn, lnorm=100)
        self.beta_syn = -0.7

    def compute_dl(self, pars):
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append( self.dl_syn
                       * self._syncRatio( self.fsyn[f1], self.f0, beta=self.beta_syn)
                       * self._syncRatio( self.fsyn[f2], self.f0, beta=self.beta_syn)
                       )
        if self.mode == "TT":
            return pars["AsyncT"] * jnp.array(dl)
        elif self.mode == "EE":
            return pars["AsyncP"] * jnp.array(dl)
        else:
            return 0.


# CIB model
class cib_model(fgmodel):
    def __init__(self, lmax, freqs, filename=None, mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "clustered CIB"

        #check effective freqs
        for f in freqs:
            if f not in self.fcib:
                raise ValueError( f"Missing CIB effective frequency for {f}")

        if filename is None:
            alpha_cib = -1.3
            self.dl_cib = self._gen_dl_powerlaw( alpha_cib)
        else:
            self.dl_cib = self._read_dl_template( filename)

    def compute_dl(self, pars):
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append( self.dl_cib
                       * self._cibRatio( self.fcib[f1], self.f0, beta=pars['beta_cib'])
                       * self._cibRatio( self.fcib[f2], self.f0, beta=pars['beta_cib'])
                       )
        if self.mode == "TT":
            return pars["Acib"] * jnp.array(dl)
        else:
            return 0.

# tSZ (one spectrum for all freqs)
class tsz_model(fgmodel):
    def __init__(self, lmax, freqs, filename="", mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        # template: Dl=l(l+1)/2pi Cl, units uK at 143GHz
        self.name = "tSZ"

        #check effective freqs for SZ
        for f in freqs:
            if f not in self.fsz:
                raise ValueError( f"Missing SZ effective frequency for {f}")

        # read Dl template (normalized at l=3000)
        sztmpl = self._read_dl_template(filename)

        self.dl_sz = []
        for f1, f2 in self._cross_frequencies:
            self.dl_sz.append( sztmpl[: lmax + 1]
                               * self._tszRatio( self.fsz[f1], self.f0)
                               * self._tszRatio( self.fsz[f2], self.f0)
                               )
        self.dl_sz = jnp.array(self.dl_sz)

    def compute_dl(self, pars):
        return pars["Atsz"] * self.dl_sz


# kSZ
class ksz_model(fgmodel):
    def __init__(self, lmax, freqs, filename="", mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        # template: Dl=l(l+1)/2pi Cl, units uK
        self.name = "kSZ"

        # read Dl template (normalized at l=3000)
        ksztmpl = self._read_dl_template(filename)

        self.dl_ksz = []
        for f1, f2 in self._cross_frequencies:
            self.dl_ksz.append(ksztmpl[: lmax + 1])
        self.dl_ksz = np.array(self.dl_ksz)

    def compute_dl(self, pars):
        if self.mode == "TT":
            return pars["Aksz"] * self.dl_ksz
        else:
            return 0.


# SZxCIB model
class szxcib_model(fgmodel):
    def __init__(self, lmax, freqs, filename=None, mode="TT", auto=False, **kwargs):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "SZxCIB"

        #check effective freqs for SZ
        for f in freqs:
            if f not in self.fsz:
                raise ValueError( f"Missing SZ effective frequency for {f}")

        #check effective freqs for dust
        for f in freqs:
            if f not in self.fcib:
                raise ValueError( f"Missing Dust effective frequency for {f}")

        self._is_template = filename
        if self._is_template:
            self.x_tmpl = self._read_dl_template(filename)
        elif "filenames" in kwargs:
            self.x_tmpl = self._read_dl_template(kwargs["filenames"][0])*self._read_dl_template(kwargs["filenames"][1])
        else:
            raise ValueError( f"Missing template for SZxCIB")

    def compute_dl(self, pars):
        dl_szxcib = []
        for f1, f2 in self._cross_frequencies:
            dl_szxcib.append( self.x_tmpl * (
                self._tszRatio(self.fsz[f2],self.f0) * self._cibRatio(self.fcib[f1],self.f0,beta=pars['beta_cib']) +
                self._tszRatio(self.fsz[f1],self.f0) * self._cibRatio(self.fcib[f2],self.f0,beta=pars['beta_cib'])
                )
            )

        if self.mode == "TT":
            return -1. * pars["xi"] * jnp.sqrt(pars["Acib"]*pars["Atsz"]) * jnp.array(dl_szxcib)
        else:
            return 0.


fg_list = {
    "sbpx": subpix,
    "ps": ps,
    "dust": dust,
    "dust_model": dust_model,
    "sync": sync_model,
    "ksz": ksz_model,
    "ps_radio": ps_radio,
    "ps_dusty": ps_dusty,
    "cib": cib_model,
    "tsz": tsz_model,
    "szxcib": szxcib_model,
}


class HillipopLikelihood(object):

    logger = logging.getLogger('hillipop')
    data_folder = 'planck_2020/hillipop'
    multipoles_range_file = 'data/binning_v4.2.fits'
    xspectra_basename = 'data/dl_PR4_v4.2'
    covariance_matrix_file = 'data/invfll_PR4_v4.2_TTTEEE.fits'
    foregrounds = {'TT': {
        'dust_model': 'foregrounds/DUST_Planck_PR4_model_v4.2',
        'tsz': 'foregrounds/SZ_Planck_PR4_model.txt',
        'ksz': 'foregrounds/kSZ_Planck_PR4_model.txt',
        'cib': 'foregrounds/CIB_Planck_PR4_model.txt',
        'szxcib': 'foregrounds/SZxCIB_Planck_PR4_model.txt',
        'ps_radio': None,
        'ps_dusty': None
    },
    'EE': {'dust_model': 'foregrounds/DUST_Planck_PR4_model_v4.2'},
    'TE': {'dust_model': 'foregrounds/DUST_Planck_PR4_model_v4.2'},
    }

    def __init__(self, data_folder=None, likelihood_name=None, proj_order=None):
        # Set path to data
        # If no path specified, use the modules path
        if data_folder is not None:
            self.data_folder = str(data_folder)
        if not os.path.exists(self.data_folder):
            raise IOError(f"The 'data_folder' directory does not exist. Check the given path [{self.data_folder}].")

        self.frequencies = [100, 100, 143, 143, 217, 217]
        self._mapnames = ["100A", "100B", "143A", "143B", "217A", "217B"]
        self._nmap = len(self.frequencies)
        self._nfreq = len(np.unique(self.frequencies))
        self._nxfreq = self._nfreq * (self._nfreq + 1) // 2
        self._nxspec = self._nmap * (self._nmap - 1) // 2
        self._xspec2xfreq, self._xspec2xfreq_matrix = self._xspec2xfreq()  # CHANGE
        self.logger.debug(f"frequencies = {self.frequencies}")

        # Get likelihood name and add the associated mode
        likelihood_name = likelihood_name or self.__class__.__name__
        likelihood_modes = [likelihood_name[i : i + 2] for i in range(0, len(likelihood_name), 2)]
        self._is_mode = {mode: mode in likelihood_modes for mode in ["TT", "TE", "EE", "BB"]}
        self._is_mode["ET"] = self._is_mode["TE"]
        self.logger.debug(f"mode = {self._is_mode}")

        # Multipole ranges
        filename = os.path.join(self.data_folder, self.multipoles_range_file)
        self._lmins, self._lmaxs = self._set_multipole_ranges(filename)
        self.lmax = np.max([max(l) for l in self._lmaxs.values()])

        # Data
        basename = os.path.join(self.data_folder, self.xspectra_basename)
        self._dldata = self._read_dl_xspectra(basename)

        # Weights
        dlsig = self._read_dl_xspectra(basename, hdu=2)
        for m, w8 in dlsig.items(): w8[w8 == 0] = np.inf
        self._dlweight = {k: 1. / v**2 for k, v in dlsig.items()}
#        self._dlweight = np.ones(np.shape(self._dldata))

        # Inverted Covariance matrix
        filename = os.path.join(self.data_folder, self.covariance_matrix_file)
        # Sanity check
        m = re.search(".*_(.+?).fits", self.covariance_matrix_file)
        if not m or likelihood_name != m.group(1):
            raise IOError("The covariance matrix mode differs from the likelihood mode. "
                f"Check the given path [{self.covariance_matrix_file}]")
        self.precision = self._read_invcovmatrix(filename)
        self.precision = self.precision.astype('float32')
        #self.precision = np.diag(self.precision)

        # Foregrounds
        self.fgs = {}  # list of foregrounds per mode [TT,EE,TE,ET]
        # Init foregrounds TT
        fgsTT = []
        if self._is_mode["TT"]:
            for name in self.foregrounds["TT"].keys():
                if name not in fg_list.keys():
                    raise ValueError(f"Unkown foreground model '{name}'!")
                self.logger.debug(f"Adding '{name}' foreground for TT")
                kwargs = dict(lmax=self.lmax, freqs=self.frequencies, mode="TT")
                if isinstance(self.foregrounds["TT"][name], str):
                    kwargs["filename"] = os.path.join(self.data_folder, self.foregrounds["TT"][name])
                elif name == "szxcib":
                    filename_tsz = self.foregrounds["TT"]["tsz"] and os.path.join(self.data_folder, self.foregrounds["TT"]["tsz"])
                    filename_cib = self.foregrounds["TT"]["cib"] and os.path.join(self.data_folder, self.foregrounds["TT"]["cib"])
                    kwargs["filenames"] = (filename_tsz,filename_cib)
                fgsTT.append(fg_list[name](**kwargs))
        self.fgs['TT'] = fgsTT

        # Init foregrounds EE
        fgsEE = []
        if self._is_mode["EE"]:
            for name in self.foregrounds["EE"].keys():
                if name not in fg_list.keys():
                    raise ValueError(f"Unkown foreground model '{name}'!")
                self.logger.debug(f"Adding '{name}' foreground for EE")
                kwargs = dict(lmax=self.lmax, freqs=self.frequencies)
                if isinstance(self.foregrounds["EE"][name], str):
                    kwargs["filename"] = os.path.join(self.data_folder, self.foregrounds["EE"][name])
                fgsEE.append(fg_list[name](mode="EE", **kwargs))
        self.fgs['EE'] = fgsEE

        # Init foregrounds TE
        fgsTE = []
        fgsET = []
        if self._is_mode["TE"]:
            for name in self.foregrounds["TE"].keys():
                if name not in fg_list.keys():
                    raise ValueError(f"Unkown foreground model '{name}'!")
                self.logger.debug(f"Adding '{name}' foreground for TE")
                kwargs = dict(lmax=self.lmax, freqs=self.frequencies)
                if isinstance(self.foregrounds["TE"][name], str):
                    kwargs["filename"] = os.path.join(self.data_folder, self.foregrounds["TE"][name])
                fgsTE.append(fg_list[name](mode="TE", **kwargs))
                fgsET.append(fg_list[name](mode="ET", **kwargs))
        self.fgs['TE'] = fgsTE
        self.fgs['ET'] = fgsET

        self.logger.debug("Initialized!")

        flatdata = []
        for mode in ['TT', 'EE']:
            if self._is_mode[mode]:
                R = self._xspectra_to_xfreq(self._dldata[mode], self._dlweight[mode])
                flatdata += self._select_spectra(R, mode)
        if self._is_mode["TE"] or self._is_mode["ET"]:
            Rl = 0
            Wl = 0
            # compute theory Dlth
            if self._is_mode["TE"]:
                RlTE, WlTE = self._xspectra_to_xfreq(self._dldata["TE"], self._dlweight["TE"], normed=False)
                Rl = Rl + RlTE
                Wl = Wl + WlTE
            if self._is_mode["ET"]:
                RlET, WlET = self._xspectra_to_xfreq(self._dldata["ET"], self._dlweight["ET"], normed=False)
                Rl = Rl + RlET
                Wl = Wl + WlET
            # select multipole range
            flatdata += self._select_spectra(Rl / Wl, "TE")
        self.flatdata = np.concatenate(flatdata)
        self.proj_order = proj_order
        if self.proj_order:
            from scipy import linalg
            proj, poly = [], []
            for i, data in enumerate(flatdata):
                size = data.size
                tmp = projection(size, order=min(size, self.proj_order))
                proj.append(tmp[0])
                poly.append(tmp[1])
            self._proj, poly = (jnp.asarray(linalg.block_diag(*tmp)) for tmp in [proj, poly])
            self._chi2_dd = self.flatdata.dot(self.precision).dot(self.flatdata)
            chi2_dt = self.flatdata.dot(self.precision).dot(poly.T)
            self._chi2_dt = jnp.asarray(- (chi2_dt + chi2_dt.T))
            self._chi2_tt = jnp.asarray(poly.dot(self.precision).dot(poly.T))

    def _xspec2xfreq(self):
        list_fqs = []
        for f1 in range(self._nfreq):
            for f2 in range(f1, self._nfreq):
                list_fqs.append((f1, f2))

        freqs = list(np.unique(self.frequencies))
        spec2freq = []
        for m1 in range(self._nmap):
            for m2 in range(m1 + 1, self._nmap):
                f1 = freqs.index(self.frequencies[m1])
                f2 = freqs.index(self.frequencies[m2])
                spec2freq.append(list_fqs.index((f1, f2)))

        # CHANGE
        matrix = np.zeros((len(list_fqs), len(spec2freq)))
        for ii, idx in enumerate(spec2freq):
            matrix[idx, ii] += 1
        return spec2freq, jnp.asarray(matrix)

    def _set_multipole_ranges(self, filename):
        """
        Return the (lmin, lmax) for each cross-spectra for each mode (TT, EE, TE, ET)
        array (nmode, nxspec)
        """
        self.logger.debug("Define multipole ranges")
        if not os.path.exists(filename):
            raise ValueError(f"File missing {filename}")

        lmins = {}
        lmaxs = {}
        with fitsio.FITS(filename) as hdus:
            for hdu in hdus[1:]:
                tag = hdu.read_header()['spec']
                lmins[tag] = hdu['LMIN'][:]
                lmaxs[tag] = hdu['LMAX'][:]
                if self._is_mode[tag]:
                    self.logger.debug(f"{tag}")
                    self.logger.debug(f"lmin: {lmins[tag]}")
                    self.logger.debug(f"lmax: {lmaxs[tag]}")
        lmins["ET"] = lmins["TE"]
        lmaxs["ET"] = lmaxs["TE"]

        return lmins, lmaxs

    def _read_dl_xspectra(self, basename, hdu=1):
        """
        Read xspectra from Xpol [Dl in K^2]
        Output: Dl (TT, EE, TE, ET) in muK^2
        """
        self.logger.debug("Reading cross-spectra")

        with fitsio.FITS(f"{basename}_{self._mapnames[0]}x{self._mapnames[1]}.fits") as hdus:
            nhdu = len(hdus)
        if nhdu < hdu:
            #no sig in file, uniform weight
            self.logger.info( "Warning: uniform weighting for combining spectra !")
            dldata = np.ones( (self._nxspec, 4, self.lmax+1))
        else:
            if nhdu == 1: hdu=0 #compatibility
            dldata = []
            for m1, m2 in itertools.combinations(self._mapnames, 2):
                data = fitsio.read( f"{basename}_{m1}x{m2}.fits", ext=hdu)*1e12
                tmpcl = list(data[[0,1,3],:self.lmax+1])
                data = fitsio.read( f"{basename}_{m2}x{m1}.fits", ext=hdu)*1e12
                tmpcl.append( data[3,:self.lmax+1])
                dldata.append( tmpcl)

        dldata = np.transpose(np.array(dldata), (1, 0, 2))
        return dict(zip(['TT', 'EE', 'TE', 'ET'],dldata))

    def _read_invcovmatrix(self, filename):
        """
        Read xspectra inverse covmatrix from Xpol [Dl in K^-4]
        Output: invkll [Dl in muK^-4]
        """
        self.logger.debug(f"Covariance matrix file: {filename}")
        if not os.path.exists(filename):
            raise ValueError(f"File missing {filename}")

        data = fitsio.read(filename)
        nel = int(np.sqrt(data.size))
        data = data.reshape((nel, nel)) / 1e24  # muK^-4

        nell = self._get_matrix_size()
        if nel != nell:
            raise ValueError(f"Incoherent covariance matrix (read:{nel}, expected:{nell})")

        return data

    def _get_matrix_size(self):
        """
        Compute covariance matrix size given activated mode
        Return: number of multipole
        """
        nell = 0

        # TT,EE,TEET
        for m in ["TT", "EE", "TE"]:
            if self._is_mode[m]:
                nells = self._lmaxs[m] - self._lmins[m] + 1
                nell += np.sum([nells[self._xspec2xfreq.index(k)] for k in range(self._nxfreq)])

        return nell

    def _select_spectra(self, acl, mode):
        """
        Cut spectra given multipole ranges and flatten
        Return: list
        """
        xl = []
        for xf in range(self._nxfreq):
            lmin = self._lmins[mode][self._xspec2xfreq.index(xf)]
            lmax = self._lmaxs[mode][self._xspec2xfreq.index(xf)]
            xl.append(acl[xf, lmin : lmax + 1])
        return xl

    def _xspectra_to_xfreq(self, cl, weight, normed=True):
        """
        Average cross-spectra per cross-frequency
        """
        """
        xcl = jnp.zeros((self._nxfreq, self.lmax + 1))
        xw8 = jnp.zeros((self._nxfreq, self.lmax + 1))
        for xs in range(self._nxspec):
            xcl[self._xspec2xfreq[xs]] += weight[xs] * cl[xs]
            xw8[self._xspec2xfreq[xs]] += weight[xs]
        """
        xcl = self._xspec2xfreq_matrix.dot(weight * cl)
        xw8 = self._xspec2xfreq_matrix.dot(weight)

        xw8 = jnp.where(xw8 == 0, np.inf, xw8)
        if normed:
            return xcl / xw8
        else:
            return xcl, xw8

    def _compute_theory(self, pars, dlth, mode):
        # Nuisances
        cal = []
        for m1, m2 in itertools.combinations(self._mapnames, 2):
            if mode == "TT":
                cal1, cal2 = pars[f"cal{m1}"], pars[f"cal{m2}"]
            elif mode == "EE":
                cal1, cal2 = pars[f"cal{m1}"]*pars[f"pe{m1}"], pars[f"cal{m2}"]*pars[f"pe{m2}"]
            elif mode == "TE":
                cal1, cal2 = pars[f"cal{m1}"], pars[f"cal{m2}"]*pars[f"pe{m2}"]
            elif mode == "ET":
                cal1, cal2 = pars[f"cal{m1}"]*pars[f"pe{m1}"], pars[f"cal{m2}"]
            cal.append(cal1 * cal2 / pars["A_planck"] ** 2)

        # Model
        dlmodel = jnp.repeat(dlth[mode][None, ...], self._nxspec, axis=0)
        for fg in self.fgs[mode]:
            dlmodel += fg.compute_dl(pars)

        # Compute Rl = Dl - Dlth
        Rspec = jnp.array([cal[xs] * dlmodel[xs] for xs in range(self._nxspec)])

        return Rspec

    def dof(self):
        return len(self.precision)

    def compute_chi2(self, dlth, **params_values):
        """
        Compute likelihood from model out of Boltzmann code
        Units: Dl in muK^2

        Parameters
        ----------
        pars: dict
              parameter values
        dl: array or arr2d
              CMB power spectrum (Dl in muK^2)

        Returns
        -------
        lnL: float
            Log likelihood for the given parameters -2ln(L)
        """

        # cl_boltz from Boltzmann (Cl in muK^2)
#        lth = np.arange(self.lmax + 1)
#        dlth = np.asarray(dl)[:, lth][[0, 1, 3, 3]]  # select TT,EE,TE,TE

        # Create Data Vector
        Xl = []
        if self._is_mode["TT"]:
            # compute theory Dlth
            Rspec = self._compute_theory(params_values, dlth, "TT")
            # average to cross-spectra
            Rl = self._xspectra_to_xfreq(Rspec, self._dlweight["TT"])
            # select multipole range
            Xl += self._select_spectra(Rl, 'TT')

        if self._is_mode["EE"]:
            # compute theory Dlth
            Rspec = self._compute_theory(params_values, dlth, "EE")
            # average to cross-spectra
            Rl = self._xspectra_to_xfreq(Rspec, self._dlweight["EE"])
            # select multipole range
            Xl += self._select_spectra(Rl, 'EE')

        if self._is_mode["TE"] or self._is_mode["ET"]:
            Rl = 0
            Wl = 0
            # compute theory Dlth
            if self._is_mode["TE"]:
                Rspec = self._compute_theory(params_values, dlth, "TE")
                RlTE, WlTE = self._xspectra_to_xfreq(Rspec, self._dlweight["TE"], normed=False)
                Rl = Rl + RlTE
                Wl = Wl + WlTE
            if self._is_mode["ET"]:
                Rspec = self._compute_theory(params_values, dlth, "ET")
                RlET, WlET = self._xspectra_to_xfreq(Rspec, self._dlweight["ET"], normed=False)
                Rl = Rl + RlET
                Wl = Wl + WlET
            # select multipole range
            Xl += self._select_spectra(Rl / Wl, 'TE')

        self.flattheory = jnp.concatenate(Xl, axis=0)#.astype('float32')  # changes at 0.01 level
        self.flatdiff = self.flatdata - self.flattheory

        if self.proj_order:
            flattheory = self._proj.dot(self.flattheory)
            chi2 = self._chi2_dd + self._chi2_dt.dot(flattheory) + flattheory.dot(self._chi2_tt).dot(flattheory)
        else:
            #chi2 = jnp.sum(self.flatdiff * self.precision * self.flatdiff, axis=0)
            chi2 = self.flatdiff.dot(self.precision).dot(self.flatdiff)

        self.logger.debug(f"chi2/ndof = {chi2}/{len(self.flattheory)}")
        return chi2

    def loglike(self, dl, **params_values):
        """
        Compute likelihood from model out of Boltzmann code
        Units: Dl in muK^2

        Parameters
        ----------
        pars: dict
              parameter values
        dl: dict
              CMB power spectrum (Dl in µK^2)

        Returns
        -------
        lnL: float
            Log likelihood for the given parameters -2ln(L)
        """
        dlth = {k.upper(): dl[k][:self.lmax+1] for k in dl.keys()}
        dlth['ET'] = dlth['TE']
        chi2 = self.compute_chi2(dlth, **params_values)
        return -0.5 * chi2



class TTTEEEHighlPlanck2020HillipopLikelihood(BaseLikelihood):

    version = 'v4.2'
    config_fn = 'hillipop.yaml'
    installer_section = 'TTTEEEHighlPlanck2020HillipopLikelihood'
    name = 'TTTEEEHighlPlanck2020Hillipop'
    cls = ['tt', 'te', 'ee']

    def initialize(self, theory=None, cosmo=None, data_dir=None, proj_order=None, **kwargs):
        if data_dir is None:
            from desilike.install import Installer
            data_dir = os.path.join(Installer()[self.installer_section]['data_dir'], self.version, 'planck_2020', 'hillipop')
        self._like = HillipopLikelihood(data_dir, self.__class__.__name__.replace('Highl', '').replace('Planck2020HillipopLikelihood', ''), proj_order=proj_order)
        requested_cls = {cl: self._like.lmax for cl in self.cls}
        ells = np.arange(self._like.lmax + 1)
        self.factor = ells * (ells + 1) / 2 / np.pi
        if theory is None: theory = ClTheory()
        self.theory = theory
        self.theory.init.update(cls=requested_cls, lensing=True, unit='muK', T0=2.7255)
        if cosmo is not None: self.theory.init.update(cosmo=cosmo)
        super().initialize(**kwargs)
        self.flatdata = self._like.flatdata  # to get size

    """
    def initialize(self, theory=None, cosmo=None, data_dir=None):
        if data_dir is None:
            from desilike.install import Installer
            data_dir = os.path.join(Installer()[self.installer_section]['data_dir'], self.version)
        import planck_2020_hillipop
        from planck_2020_hillipop import TTTEEE
        data_folder = 'planck_2020/hillipop'
        multipoles_range_file = 'data/binning_v4.2.fits'
        xspectra_basename = 'data/dl_PR4_v4.2'
        covariance_matrix_file = 'data/invfll_PR4_v4.2_TTTEEE.fits'
        attrs = dict(path=data_dir, data_folder=data_folder, multipoles_range_file=multipoles_range_file, xspectra_basename=xspectra_basename, covariance_matrix_file=covariance_matrix_file)
        #for name, value in attrs.items(): setattr(TTTEEE, name, value)
        TTTEEE.is_installed = lambda *args, **kwargs: True
        self._like = TTTEEE.__new__(TTTEEE)
        for name, value in attrs.items(): setattr(self._like, name, value)
        self._like.__init__(packages_path=data_dir)
        requested_cls = {cl: self._like.lmax for cl in self.cls}
        ells = np.arange(self._like.lmax + 1)
        self.factor = ells * (ells + 1) / 2 / np.pi
        if theory is None: theory = ClTheory()
        self.theory = theory
        self.theory.init.update(cls=requested_cls, lensing=True, unit='muK', T0=2.7255)
        if cosmo is not None: self.theory.init.update(cosmo=cosmo)
    """

    def calculate(self, **params):
        dls = {name: self.theory.cls[name] * self.factor for name in self.cls}
        self.loglikelihood = self._like.loglike(dls, **params)

    @classmethod
    def install(cls, installer):
        #installer.pip('git+https://github.com/planck-npipe/hillipop')

        try:
            data_dir = installer[cls.installer_section]['data_dir']
        except KeyError:
            data_dir = installer.data_dir(cls.installer_section)

        from desilike.install import exists_path, download, extract

        if installer.reinstall or not exists_path(os.path.join(data_dir, cls.version)):
            names = cls.__name__.replace('Highl', '').replace('Planck2020HillipopLikelihood', '')
            tar_base = 'planck_2020_hillipop_{}_{}.tar.gz'.format(names, cls.version)
            url = 'https://portal.nersc.gov/cfs/cmb/planck2020/likelihoods/{}'.format(tar_base)
            tar_fn = os.path.join(data_dir, cls.version, tar_base)
            download(url, tar_fn)
            extract(tar_fn, os.path.dirname(tar_fn))

        installer.write({cls.installer_section: {'data_dir': data_dir}})


class TTHighlPlanck2020HillipopLikelihood(TTTEEEHighlPlanck2020HillipopLikelihood):

    version = 'v4.2'
    config_fn = 'hillipop.yaml'
    installer_section = 'TTHighlPlanck2020HillipopLikelihood'
    name = 'TTHighlPlanck2020Hillipop'