"""
JAX-adaptation of https://github.com/planck-npipe/lollipop/tree/master.
First class without desilike / cobaya dependency (would be good to have in the github above),
then desilike-specific wrapping.
"""

import os
import logging

import numpy as np
import fitsio

from desilike.likelihoods.base import BaseGaussianLikelihood
from desilike.jax import numpy as jnp
from desilike.jax import map
from .base import ClTheory


# Bin class
import numpy as np


class Bins(object):
    """
    lmins : list of integers
        Lower bound of the bins
    lmaxs : list of integers
        Upper bound of the bins
    """

    def __init__(self, lmins, lmaxs):
        if not (len(lmins) == len(lmaxs)):
            raise ValueError("Incoherent inputs")

        lmins = np.asarray(lmins)
        lmaxs = np.asarray(lmaxs)
        cutfirst = np.logical_and(lmaxs >= 2, lmins >= 2)
        self.lmins = lmins[cutfirst]
        self.lmaxs = lmaxs[cutfirst]

        self._derive_ext()

    @classmethod
    def fromdeltal(cls, lmin, lmax, delta_ell):
        nbins = (lmax - lmin + 1) // delta_ell
        lmins = lmin + np.arange(nbins) * delta_ell
        lmaxs = lmins + delta_ell - 1
        return cls(lmins, lmaxs)

    def _derive_ext(self):
        for l1, l2 in zip(self.lmins, self.lmaxs):
            if l1 > l2:
                raise ValueError("Incoherent inputs")
        self.lmin = np.min(self.lmins)
        self.lmax = np.max(self.lmaxs)
        if self.lmin < 1:
            raise ValueError("Input lmin is less than 1.")
        if self.lmax < self.lmin:
            raise ValueError("Input lmax is less than lmin.")

        self.nbins = len(self.lmins)
        self.lbin = (self.lmins + self.lmaxs) / 2.0
        self.dl = self.lmaxs - self.lmins + 1

    def bins(self):
        return (self.lmins, self.lmaxs)

    def cut_binning(self, lmin, lmax):
        sel = np.where((self.lmins >= lmin) & (self.lmaxs <= lmax))[0]
        self.lmins = self.lmins[sel]
        self.lmaxs = self.lmaxs[sel]
        self._derive_ext()

    def _bin_operators(self, Dl=False, cov=False):
        if Dl:
            ell2 = np.arange(self.lmax + 1)
            ell2 = ell2 * (ell2 + 1) / (2 * np.pi)
        else:
            ell2 = np.ones(self.lmax + 1)
        p = np.zeros((self.nbins, self.lmax + 1))
        q = np.zeros((self.lmax + 1, self.nbins))

        for b, (a, z) in enumerate(zip(self.lmins, self.lmaxs)):
            dl = z - a + 1
            p[b, a : z + 1] = ell2[a : z + 1] / dl
            if cov:
                q[a : z + 1, b] = 1 / ell2[a : z + 1] / dl
            else:
                q[a : z + 1, b] = 1 / ell2[a : z + 1]

        return p, q

    def bin_spectra(self, spectra):
        """
        Average spectra in bins specified by lmin, lmax and delta_ell,
        weighted by `l(l+1)/2pi`.
        Return Cb
        """
        spectra = jnp.asarray(spectra)
        minlmax = np.min([spectra.shape[-1] - 1, self.lmax])

        _p, _q = self._bin_operators()
        return jnp.dot(spectra[..., : minlmax + 1], _p.T[: minlmax + 1, ...])

    def bin_covariance(self, clcov):
        p, q = self._bin_operators(cov=True)
        return np.matmul(p, np.matmul(clcov, q))



def compute_offsets(ell, varcl, clref, fsky=1.0, iter=10):
    Nl = np.sqrt(np.abs(varcl - (2.0 / (2.0 * ell + 1) * clref ** 2) / fsky))
    for i in range(iter):
        Nl = np.sqrt(np.abs(varcl - 2.0 / (2.0 * ell + 1) / fsky * (clref ** 2 + 2.0 * Nl * clref)))
    return Nl * np.sqrt((2.0 * ell + 1) / 2.0)


def read_dl(datafile):
    data = np.loadtxt(datafile).T
    dl = np.zeros((3, int(max(data[0])) + 1))  # EE,BB,EB
    l = np.array(data[0], int)
    dl[0, l] = data[1]
    dl[1, l] = data[2]
    dl[2, l] = data[3]
    return dl


def get_binning(lmin,lmax):
    dl = 10
    if lmin < 2:
        raise ValueError( f"Lmin should be > 2: {lmin}")
    if lmax > 200:
        raise ValueError( f"Lmax should be < 200: {lmax}")

    if lmin >= 36:
        lmins = list(range(lmin, lmax - dl + 2, dl))
        lmaxs = list(range(lmin + dl - 1, lmax + 1, dl))
    elif lmax <= 35:
        lmins = list(range(lmin, lmax + 1))
        lmaxs = list(range(lmin, lmax + 1))
    else:
        llmin = lmin
        llmax = 35
        hlmin = 36
        hlmax = lmax
        lmins = list(range(llmin, llmax + 1)) + list(range(hlmin, hlmax - dl + 2, dl))
        lmaxs = list(range(llmin, llmax + 1)) + list(range(hlmin + dl - 1, hlmax + 1, dl))
    binc = Bins(lmins, lmaxs)
    return binc


def bin_covEB(clcov, binc):
    nell = len(clcov) // 3
    cbcov = np.zeros((3 * binc.nbins, 3 * binc.nbins))
    for t1 in range(3):
        for t2 in range(3):
            mymat = np.zeros((binc.lmax + 1, binc.lmax + 1))
            mymat[2:, 2:] = clcov[
                t1 * nell : t1 * nell + (binc.lmax - 1), t2 * nell : t2 * nell + (binc.lmax - 1)
            ]
            cbcov[
                t1 * binc.nbins : (t1 + 1) * binc.nbins, t2 * binc.nbins : (t2 + 1) * binc.nbins
            ] = binc.bin_covariance(mymat)
    return cbcov


def bin_covBB(clcov, binc):
    nell = len(clcov) // 3
    t1 = t2 = 1
    mymat = np.zeros((binc.lmax + 1, binc.lmax + 1))
    mymat[2:, 2:] = clcov[
        t1 * nell : t1 * nell + (binc.lmax - 1), t2 * nell : t2 * nell + (binc.lmax - 1)
    ]
    cbcov = binc.bin_covariance(mymat)
    return cbcov


def bin_covEE(clcov, binc):
    nell = len(clcov) // 3
    t1 = t2 = 0
    mymat = np.zeros((binc.lmax + 1, binc.lmax + 1))
    mymat[2:, 2:] = clcov[
        t1 * nell : t1 * nell + (binc.lmax - 1), t2 * nell : t2 * nell + (binc.lmax - 1)
    ]
    cbcov = binc.bin_covariance(mymat)
    return cbcov


def vec2mat(vect):
    """
    shape EE, BB and EB as a matrix
    input:
        vect: EE,BB,EB
    output:
        matrix: [[EE,EB],[EB,BB]]
    """
    off = vect[2] if len(vect) == 3 else 0.
    mat = jnp.array([[vect[0], off], [off, vect[1]]])
    """
    mat = np.zeros((2, 2))
    mat[0, 0] = vect[0]
    mat[1, 1] = vect[1]
    if len(vect) == 3:
        mat[1, 0] = mat[0, 1] = vect[2]
    """
    return mat


def mat2vec(mat):
    """
    shape polar matrix into polar vect
    input:
        matrix: [[EE,EB],[EB,BB]]
    output:
        vect: EE,BB,EB
    """
    vec = jnp.array([mat[0, 0], mat[1, 1], mat[0, 1]])
    return vec


def ghl(x):
    return jnp.sign(x - 1) * jnp.sqrt(2.0 * (x - jnp.log(x) - 1))



class LollipopLikelihood(object):

    logger = logging.getLogger('lollipop')
    data_folder = 'planck_2020/lollipop'
    cl_file = 'cl_lolEB_NPIPE.dat'
    fiducial_file = 'fiducial_lolEB_planck2018_tensor_lensedCls.dat'
    cl_cov_file = 'clcov_lolEB_NPIPE.fits'
    hartlap_factor = False
    marginalised_over_covariance = True
    Nsim = 400
    lmin = 2
    lmax = 30

    def __init__(self, data_folder=None, likelihood_name=None):
        # Set path to data
        # If no path specified, use the modules path
        if data_folder is not None:
            self.data_folder = str(data_folder)
        if not os.path.exists(self.data_folder):
            raise IOError(f"The 'data_folder' directory does not exist. Check the given path [{self.data_folder}].")

        # Get likelihood name and add the associated mode
        likelihood_name = likelihood_name or self.__class__.__name__
        self.mode = {'EE': 'lowlE', 'BB': 'lowlB', 'EB': 'lowlEB'}.get(likelihood_name, likelihood_name)
        self.logger.debug(f"mode = {self.mode}")
        if self.mode not in ["lowlE", "lowlB", "lowlEB"]:
            raise ValueError("The {} likelihood is not currently supported. Check your likelihood name.".format(self.mode))

        # Binning (fixed binning)
        self.bins = get_binning(self.lmin,self.lmax)
        self.logger.debug(f"lmax = {self.bins.lmax}")

        # Data (ell,ee,bb,eb)
        self.logger.debug("Reading cross-spectrum")
        filepath = os.path.join(self.data_folder, self.cl_file)
        data = read_dl(filepath)
        self.cldata = self.bins.bin_spectra(data)

        # Fiducial spectrum (ell,ee,bb,eb)
        self.logger.debug("Reading model")
        filepath = os.path.join(self.data_folder, self.fiducial_file)
        data = read_dl(filepath)
        self.clfid = self.bins.bin_spectra(data)

        # covmat (ee,bb,eb)
        self.logger.debug("Reading covariance")
        filepath = os.path.join(self.data_folder, self.cl_cov_file)
        clcov = fitsio.read(filepath)
        if self.mode == "lowlEB":
            cbcov = bin_covEB(clcov, self.bins)
        elif self.mode == "lowlE":
            cbcov = bin_covEE(clcov, self.bins)
        elif self.mode == "lowlB":
            cbcov = bin_covBB(clcov, self.bins)
        clvar = np.diag(cbcov).reshape(-1, self.bins.nbins)

        if self.mode == "lowlEB":
            rcond = getattr(self, "rcond", 1e-9)
            self.invclcov = np.linalg.pinv(cbcov, rcond)
        else:
            self.invclcov = np.linalg.inv(cbcov)

        # Hartlap et al. 2008
        if self.hartlap_factor:
            if self.Nsim != 0:
                self.invclcov *= (self.Nsim - len(cbcov) - 2) / (self.Nsim - 1)

        if self.marginalised_over_covariance:
            if self.Nsim <= 1:
                raise ValueError("Need the number of MC simulations used to compute the covariance in order to marginalise over (Nsim>1).")

        # compute offsets
        self.logger.debug("Compute offsets")
        fsky = getattr(self, "fsky", 0.52)
        self.cloff = compute_offsets(self.bins.lbin, clvar, self.clfid, fsky=fsky)
        self.cloff[2:] = 0.0  # force NO offsets EB

        self.logger.debug("Initialized!")
        for name in ['cloff', 'cldata', 'clfid']:
            setattr(self, name, jnp.asarray(getattr(self, name)))

    def _compute_chi2_2fields(self, cl, **params_values):
        """
        Compute offset-Hamimeche&Lewis likelihood
        Input: Cl in muK^2
        """
        # get model in Cl, muK^2
        clth = jnp.array(
            [self.bins.bin_spectra(cl[mode]) for mode in ["ee", "bb", "eb"] if mode in cl]
        )

        cal = params_values["A_planck"] ** 2

        nell = self.cldata.shape[1]

        def _get_x(ell):
            O = vec2mat(self.cloff[:, ell])
            D = vec2mat(self.cldata[:, ell]*cal) + O
            M = vec2mat(clth[:, ell]) + O
            F = vec2mat(self.clfid[:, ell]) + O

            # compute P = C_model^{-1/2}.C_data.C_model^{-1/2}
            w, V = jnp.linalg.eigh(M)
            #            if prod( sign(w)) <= 0:
            #                print( "WARNING: negative eigenvalue for l=%d" %l)
            L = V @ jnp.diag(1.0 / jnp.sqrt(w)) @ V.transpose()
            P = L.transpose() @ D @ L

            # apply HL transformation
            w, V = jnp.linalg.eigh(P)
            g = jnp.sign(w) * ghl(jnp.abs(w))
            G = V @ jnp.diag(g) @ V.transpose()

            # cholesky fiducial
            w, V = jnp.linalg.eigh(F)
            L = V @ jnp.diag(jnp.sqrt(w)) @ V.transpose()

            # compute C_fid^1/2 * G * C_fid^1/2
            X = L.transpose() @ G @ L
            return mat2vec(X)

        x = map(_get_x, np.arange(nell))

        # compute chi2
        x = x.T.flatten()
        if self.marginalised_over_covariance:
            chi2 = self.Nsim * jnp.log(1 + (x @ self.invclcov @ x) / (self.Nsim - 1))
        else:
            chi2 = x @ self.invclcov @ x

        self.logger.debug(f"chi2/ndof = {chi2}/{len(x)}")
        return chi2

    def _compute_chi2_1field(self, cl, **params_values):
        """
        Compute offset-Hamimeche&Lewis likelihood
        Input: Cl in muK^2
        """
        # model in Cl, muK^2
        m = 0 if self.mode == "lowlE" else 1
        clth = self.bins.bin_spectra(cl["ee" if self.mode == "lowlE" else "bb"])

        cal = params_values["A_planck"] ** 2

        x = (self.cldata[m]*cal + self.cloff[m]) / (clth + self.cloff[m])
        g = jnp.sign(x) * ghl(jnp.abs(x))

        X = (jnp.sqrt(self.clfid[m] + self.cloff[m])) * g * (jnp.sqrt(self.clfid[m] + self.cloff[m]))

        if self.marginalised_over_covariance:
            # marginalised over S = Ceff
            chi2 = self.Nsim * jnp.log(1 + (X @ self.invclcov @ X) / (self.Nsim - 1))
        else:
            chi2 = X @ self.invclcov @ X

        self.logger.debug(f"chi2/ndof = {chi2}/{len(X)}")
        return chi2

    def loglike(self, cl, **params_values):
        if self.mode == "lowlEB":
            chi2 = self._compute_chi2_2fields(cl, **params_values)
        elif self.mode in ["lowlE", "lowlB"]:
            chi2 = self._compute_chi2_1field(cl, **params_values)

        return -0.5 * chi2


class EELowlPlanck2020LollipopLikelihood(BaseGaussianLikelihood):

    config_fn = 'lollipop.yaml'
    installer_section = 'EELowlPlanck2020lollipopLikelihood'
    cls = ['ee', 'bb']
    name = 'EELowlPlanck2020lollipop'

    def initialize(self, theory=None, cosmo=None, data_dir=None):
        if data_dir is None:
            from desilike.install import Installer
            data_dir = os.path.join(Installer()[self.installer_section]['data_dir'], 'planck_2020', 'lollipop')

        self._like = LollipopLikelihood(data_dir, self.__class__.__name__.replace('Lowl', '').replace('Planck2020LollipopLikelihood', ''))
        requested_cls = {cl: self._like.lmax for cl in self.cls}
        ells = np.arange(self._like.lmax + 1)
        self.factor = ells * (ells + 1) / 2 / np.pi
        if theory is None: theory = ClTheory()
        self.theory = theory
        self.theory.init.update(cls=requested_cls, lensing=True, unit='muK', T0=2.7255)
        if cosmo is not None: self.theory.init.update(cosmo=cosmo)
    """
    def initialize(self, theory=None, cosmo=None, data_dir=None):
        if data_dir is None:
            from desilike.install import Installer
            data_dir = os.path.join(Installer()[self.installer_section]['data_dir'])
        import planck_2020_lollipop
        from planck_2020_lollipop import lowlE as EE
        data_folder = 'planck_2020/lollipop'
        cl_file = 'cl_lolEB_NPIPE.dat'
        fiducial_file = 'fiducial_lolEB_planck2018_tensor_lensedCls.dat'
        cl_cov_file = 'clcov_lolEB_NPIPE.fits'
        hartlap_factor = False
        marginalised_over_covariance = True
        Nsim = 400
        lmin = 2
        lmax = 30
        attrs = dict(path=data_dir, data_folder=data_folder, cl_file=cl_file, fiducial_file=fiducial_file, cl_cov_file=cl_cov_file, hartlap_factor=hartlap_factor,
        marginalised_over_covariance=marginalised_over_covariance, Nsim=Nsim, lmin=lmin, lmax=lmax)
        #for name, value in attrs.items(): setattr(EE, name, value)
        EE.is_installed = lambda *args, **kwargs: True
        self._like = EE.__new__(EE)
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
        #installer.pip('git+https://github.com/planck-npipe/lollipop')

        try:
            data_dir = installer[cls.installer_section]['data_dir']
        except KeyError:
            data_dir = installer.data_dir(cls.installer_section)

        from desilike.install import exists_path, download, extract

        if installer.reinstall or not exists_path(data_dir):
            tar_base = 'planck_2020_lollipop.tar.gz'
            url = 'https://portal.nersc.gov/cfs/cmb/planck2020/likelihoods/{}'.format(tar_base)
            tar_fn = os.path.join(data_dir, tar_base)
            download(url, tar_fn)
            extract(tar_fn, data_dir)

        installer.write({cls.installer_section: {'data_dir': data_dir}})


class EBLowlPlanck2020LollipopLikelihood(EELowlPlanck2020LollipopLikelihood):

    name = 'EBLowlPlanck2020lollipop'


class BBLowlPlanck2020LollipopLikelihood(EELowlPlanck2020LollipopLikelihood):

    name = 'BBLowlPlanck2020lollipop'