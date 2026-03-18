import os

import numpy as np
from getdist import IniFile
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline
try:
    from scipy.integrate import simpson
except ImportError:
    from scipy.integrate import simps as simpson
import astropy.units as u
import astropy.constants as const
import pickle
import copy

from desilike.likelihoods.base import BaseGaussianLikelihood
from desilike.jax import numpy as jnp
from desilike.theories.weak_lensing import DESWeakLensing3x2pt


class BaseDESY3Likelihood(BaseGaussianLikelihood):

    def initialize(self, use_sr=False, theory=None, cosmo=None, data_dir=None, **kwargs):
        if data_dir is None:
            from desilike.install import Installer
            data_dir = os.path.join(Installer()[self.installer_section]['data_dir'], 'des_y3')
        if theory is None: theory = DESWeakLensing3x2pt()
        self.theory = theory
        if cosmo is not None: self.theory.init.update(cosmo=cosmo)
        self.use_sr=use_sr
        self.load_data(data_dir, **kwargs)
        self.data_vector = self.make_vector(self.data_arrays)

    def load_data(self, data_dir):
        ini = IniFile(os.path.join(data_dir, 'DES_3YR_final.dataset'))
        self.indices = []
        self.used_indices = []
        self.used_items = []
        self.fullcov = np.loadtxt(ini.relativeFileName('cov_file'))
        ntheta = ini.int('num_theta_bins')
        theta = np.loadtxt(ini.relativeFileName('theta_bins_file'))
        self.theta_edges = theta[:,0]
        self.theta_edges = np.append(self.theta_edges, theta[-1,1])
        self.theta_bins = theta[:,2]
        self.intrinsic_alignment_model = ini.string('intrinsic_alignment_model')
        self.data_types = ini.string('data_types').split()
        self.used_types = ini.list('used_data_types', self.data_types)
        with open(ini.relativeFileName('data_selection'), encoding="utf-8") as f:
            header = f.readline()
            assert ('#  type bin1 bin2 theta_min theta_max' == header.strip())
            lines = f.readlines()
        ranges = {}
        for tp in self.data_types:
            ranges[tp] = np.empty((6, 6), dtype=object)
        for line in lines:
            items = line.split()
            if items[0] in self.used_types:
                bin1, bin2 = [int(x) - 1 for x in items[1:3]]
                ranges[items[0]][bin1][bin2] = [np.float64(x) for x in items[3:]]
        self.ranges = ranges
        self.nzbins = ini.int('num_z_bins')  # for lensing sources
        self.nwbins = ini.int('num_gal_bins', 0)  # for galaxies
        maxbin = max(self.nzbins, self.nwbins)
        cov_ix = 0
        self.bin_pairs = []
        self.data_arrays = []
        self.thetas = []
        for i, tp in enumerate(self.data_types):
            xi = np.loadtxt(ini.relativeFileName('measurements[%s]' % tp))
            bin1 = xi[:, 0].astype(int) - 1
            bin2 = xi[:, 1].astype(int) - 1
            tbin = xi[:, 2].astype(int) - 1
            corr = np.empty((maxbin, maxbin), dtype=object)
            corr[:, :] = None
            self.data_arrays.append(corr)
            self.bin_pairs.append([])
            for f1, f2, ix, dat in zip(bin1, bin2, tbin, xi[:, 3]):
                self.indices.append((i, f1, f2, ix))
                if not (f1, f2) in self.bin_pairs[i]:
                    self.bin_pairs[i].append((f1, f2))
                    corr[f1, f2] = np.zeros(ntheta)
                corr[f1, f2][ix] = dat
                if ranges[tp][f1, f2] is not None:
                    mn, mx = ranges[tp][f1, f2]
                    if mn < self.theta_bins[ix] < mx:
                        self.thetas.append(self.theta_bins[ix])
                        self.used_indices.append(cov_ix)
                        self.used_items.append(self.indices[-1])
                cov_ix += 1
        nz_source = np.loadtxt(ini.relativeFileName('nz_file'))
        self.zmid = nz_source[:, 1]
        nz_lens = np.loadtxt(ini.relativeFileName('nz_gal_file'))
        assert (np.array_equal(nz_lens[:, 1], self.zmid))
        self.std_z = []
        for b in range(self.nwbins):
            mean_z = np.trapz(self.zmid * nz_lens[:, b + 3], self.zmid) / np.trapz(nz_lens[:, b + 3], self.zmid)
            variance_z = np.trapz(nz_lens[:, b + 3] * (self.zmid - mean_z)**2, self.zmid) / np.trapz(nz_lens[:, b + 3], self.zmid)
            self.std_z.append(np.sqrt(variance_z))
        self.zmax = self.zmid[-1]

        #shear-ratio data
        if self.use_sr:
            sr_file = ini.relativeFileName('sr_file')
            with open(sr_file, "rb") as f:
                ratio_data = pickle.load(f)
            self.sr_nbin_source = ratio_data['nbin_source']  # 4, we use all the source bins
            self.sr_nbin_lens = ratio_data['nbin_lens'] # 3, because we don't use all the lens bins
            self.sr_nratios_per_lens = ratio_data['nratios_per_lens'] # 3, because there are 3 independent ratios we can construct given 4 source bins, per each lens bin.

            self.sr_data = ratio_data['measured_ratios']
            sr_cov = ratio_data['ratio_cov']
            self.sr_covinv = np.linalg.inv(sr_cov)
            sr_theta = ratio_data['theta_data']
            sr_theta_max = [25.4, 18.26, 13.03, 10.87, 9.66, 9.04]
            sr_theta_min = [[8.47, 6.07, 4.34, 2.5, 2.5, 2.5], [8.47, 6.07, 4.34, 2.5, 2.5, 2.5], [2.5, 2.5, 4.34, 2.5, 2.5, 2.5]]
            ind_cov_data = ratio_data['inv_cov_individual_ratios']

            # Generate scale masks for each bin pair.
            # These are used in the calculation of each ratio from the range of points
            self.sr_masks = {}
            for sc in range(0, self.sr_nratios_per_lens):
                for l in range(0, self.sr_nbin_lens):
                    t_min = sr_theta_min[sc][l]
                    t_max = sr_theta_max[l]
                    self.sr_masks[(sc, l)] = (sr_theta > t_min) & (sr_theta <= t_max)

            self.inv_cov_individual_ratios = {}
            sr_n = 0
            for l in range(0, self.sr_nbin_lens):
                for sc in range(0, self.sr_nratios_per_lens):
                    mask = self.sr_masks[sc, l]
                    srP = ind_cov_data[sr_n][mask][:, mask]
                    self.inv_cov_individual_ratios[sc, l] = srP
                    sr_n += 1
    
    def initialize_postload(self):
        self.covmat = self.fullcov[np.ix_(self.used_indices, self.used_indices)]
        self.covinv_orig = np.linalg.inv(self.covmat)
        self.errors = copy.deepcopy(self.data_arrays)
        cov_ix = 0
        for i, (type_ix, f1, f2, ix) in enumerate(self.indices):
            self.errors[type_ix][f1, f2][ix] = np.sqrt(self.fullcov[cov_ix, cov_ix])
            cov_ix += 1
        self.theta_bins_radians = self.theta_bins / 60 * np.pi / 180
        self.theta_edges_radians = self.theta_edges / 60 * np.pi / 180

        self.zs = self.zmid[self.zmid <= self.zmax]

        #add point_mass to \gamma_t
        sigcrit_inv_fac = (4 * np.pi * const.G)/(const.c**2)
        self.sigcrit_inv_fac_Mpc_Msun = (sigcrit_inv_fac.to(u.Mpc/u.M_sun)).value

        def get_Dcom_array(zarray, Omega_m, H0, Omega_L=None):
            if Omega_L is None:
                Omega_L = 1. - Omega_m
            c = 299792.458
            Dcom_array = np.zeros(len(zarray))
            for j in range(len(zarray)):
                zf = zarray[j]
                res1 = scipy.integrate.quad(lambda z: (c / H0) * (1 / (np.sqrt(Omega_L + Omega_m * ((1 + z) ** 3)))), 0, zf)
                Dcom = res1[0]
                Dcom_array[j] = Dcom
            return Dcom_array

        z_distance = np.linspace(0.0,6.2,1000)
        chi_distance = get_Dcom_array(z_distance, 0.3, 69.0)
        chi_of_z = InterpolatedUnivariateSpline(z_distance, chi_distance * 0.69 )
        # Setup the variables needed for sigma_crit_inverse
        self.chi_lens = chi_of_z(self.zs)
        self.chi_source = chi_of_z(self.zs)
        self.chi_lmat = np.tile(self.chi_lens.reshape(len(self.zs), 1), (1, len(self.zs)))
        self.chi_smat = np.tile(self.chi_source.reshape(1, len(self.zs)), (len(self.zs), 1))
        self.num = self.chi_smat - self.chi_lmat
        ind_lzero = np.where(self.num <= 0)
        self.num[ind_lzero] = 0

        #add point_mass to \gamma_t
        self.sigma_crit_inv = np.empty((self.nwbins, self.nzbins))
        nz_lens = self.theory.nz_lens
        nz_source = self.theory.nz_source
        for j1 in range(self.nwbins):
            for j2 in range(self.nzbins):
                nz_lens_ = nz_lens[j1]
                nz_source_ = nz_source[j2]

                ng_array_source_rep = np.tile(nz_source_.reshape(1, len(self.zs)), (len(self.zs), 1))
                int_sourcez = simpson(ng_array_source_rep * (self.num / self.chi_smat), x=self.zs)

                coeff_ints = self.sigcrit_inv_fac_Mpc_Msun
                Is = coeff_ints * self.chi_lens * (1. + self.zs) * int_sourcez
                # Evaluate the Cij of Eq 24 written in the comoving coordinates
                # Since gamma_t is a scalar it should be same in both physical and comoving coordinates
                # It is just easier to match the expressions in comoving coordinates to the ones on methods paper.
                betaj1j2_pm = simpson(nz_lens_ * Is * (1./self.chi_lens**2), x=self.zs)
                self.sigma_crit_inv[j1, j2] = 1e13 * betaj1j2_pm

        n_datavector = len(self.data_vector)
        if any(t in self.used_types for t in ["gammat"]):
            gammat_start_ind = 0
            for item in self.used_items:
                if item[0] != 2:
                    gammat_start_ind += 1
                else:
                    break
            gammat_spec = [t for t in self.used_items if t[0] == 2]
            bin_pairs = [(t[1], t[2]) for t in gammat_spec]
            bin1 = [p[0] for p in bin_pairs]
            bin2 = [p[1] for p in bin_pairs]
            theta_rad = np.array([self.theta_bins_radians[t[3]] for t in gammat_spec])
            lens_bin_ids = list(set([p[0] for p in bin_pairs]))
            source_bin_ids = list(set([p[1] for p in bin_pairs]))
            bin_pair_dv_inds = {}
            template_matrix_pm = np.zeros((len(lens_bin_ids), n_datavector))
            for i,lens_bin in enumerate(lens_bin_ids):
                template_vector = np.zeros(n_datavector)
                lens_use = (bin1==lens_bin)
                lens_spec_inds, = np.where(lens_use)
                lens_dv_inds = lens_spec_inds + gammat_start_ind
                theta_m2 = theta_rad[lens_use]**-2  #rad
                template_vector[lens_dv_inds] = theta_m2
                for source_bin in source_bin_ids:
                    bin_pair_use = lens_use * (bin2==source_bin)
                    bin_pair_spec_inds, = np.where(bin_pair_use)
                    bin_pair_dv_inds[lens_bin, source_bin] = bin_pair_spec_inds + gammat_start_ind
                template_matrix_pm[i] = template_vector

            for i, lens_bin in enumerate(lens_bin_ids):
                for j, source_bin in enumerate(source_bin_ids):
                    sig_crit_inv = self.sigma_crit_inv[lens_bin,source_bin]
                    inds = bin_pair_dv_inds[lens_bin, source_bin]
                    template_matrix_pm[i][inds] *= sig_crit_inv
            template_matrix_with_sig = template_matrix_pm
            sigma_a = 1e4
            #Now construct inverse covariance
            U, V = template_matrix_with_sig.T, template_matrix_with_sig
            UCinvV = np.matmul(V, np.matmul(self.covinv_orig, U))
            if sigma_a > 0.:
                X = sigma_a**-2 * np.identity(UCinvV.shape[0]) + UCinvV 
            else:
                #infinite prior case
                X = UCinvV
            Xinv = np.linalg.inv(X)
            Y = np.matmul(Xinv, np.matmul(V, self.covinv_orig) )
            sub_from_inv = np.matmul(self.covinv_orig, np.matmul(U, Y))
            self.covinv = self.covinv_orig - sub_from_inv
        else:
            self.covinv = self.covinv_orig

    #shear-ratio calculation
    def get_ratio_from_gammat(self, gammat1, gammat2, inv_cov):
        #Given two gammats, calculate the ratio
        s2 = 1./float(np.ones(len(gammat1)) @ inv_cov @ np.ones(len(gammat1)))
        ratio = s2*float(gammat1/gammat2 @ inv_cov @ np.ones(len(gammat1)))
        return ratio
    
    def make_vector(self, arrays):
        nused = len(self.used_items)
        data = np.empty(nused)
        for i, (type_ix, f1, f2, theta_ix) in enumerate(self.used_items):
            data[i] = arrays[type_ix][f1, f2][theta_ix]
        return data
    
    def chi_squared(self, theory):
        theory_vector = self.make_vector([theory.xip, theory.xim, theory.gammat, theory.wtheta])
        delta = self.data_vector - theory_vector
        chi2 = self.covinv.dot(delta).dot(delta)
        return chi2

    def calculate(self):
        self.initialize_postload()
        if self.use_sr:
            s_ref = self.sr_nbin_source
            corrs_th_t = self.theory.gammat
            theory_ratios = []

            for l in range(0, self.sr_nbin_lens):
                gammat_ref = corrs_th_t[l, s_ref - 1]

                for s in range(0, self.sr_nbin_source - 1):
                    mask = self.sr_masks[s, l]
                    srP = self.inv_cov_individual_ratios[s, l]
                    gamma_t = corrs_th_t[l, s]
                    ratio = self.get_ratio_from_gammat(gamma_t[mask], gammat_ref[mask], srP)
                    theory_ratios.append(ratio)
            d = theory_ratios - self.sr_data
            sr_chi2 = d @ self.sr_covinv @ d
            self.loglikelihood = -0.5 * (self.chi_squared(self.theory) + sr_chi2)
        else:
            self.loglikelihood = -0.5 * self.chi_squared(self.theory)