import numpy as np
from desilike.jax import numpy as jnp
from desilike.jax import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import special
from fastpt import FASTPT
from desilike.theories.primordial_cosmology import constants as const

_spline = InterpolatedUnivariateSpline

try:
    from desilike.jax.numpy import trapezoid as trapz
except:
    from numpy import trapz
jnp.trapz = trapz

def TwoPTCalculator(params, cosmo, PKdelta_NL, PKdelta_L, PKWeyl, PKMW,
                   zbin_sp, zbin_w_sp, zs, nzbins, nwbins, bin_pairs,
                   Limber=None, Weyl=False, ia_model=None, fourier=None):
    acc = 1
    l_max = 40000
    h = params['h'].value
    h2 = h ** 2
    h3 = h ** 3
    omegam = (params['omega_b'].value + params['omega_cdm'].value + params['m_ncdm'].value/93.14) / h2
    chis = cosmo.comoving_radial_distance(zs)/h
    Hs = cosmo.hubble_function(zs)/(const.c / 1e3)
    dchis = jnp.hstack(
        ((chis[1] + chis[0]) / 2, (chis[2:] - chis[:-2]) / 2, (chis[-1] - chis[-2])))

    D_growth = PKdelta_L(0.001)
    D_growth = jnp.sqrt(D_growth / PKdelta_L(0.001)[0])
    c = const.c # m/s
    Ci = jnp.array([0.43, 0.30, 1.75, 1.94, 1.56, 2.96])

    bin_bias = jnp.array([params['DES_b{}'.format(i + 1)].value for i in range(nwbins)])
    shear_calibration_parameters = jnp.array([params['DES_m{}'.format(i + 1)].value for i in range(nzbins)])
    lens_photoz_errors = jnp.array([params['DES_DzL{}'.format(i + 1)].value for i in range(nwbins)])
    wl_photoz_errors = jnp.array([params['DES_DzS{}'.format(i + 1)].value for i in range(nzbins)])
    lens_photoz_width = jnp.array([params['DES_szL{}'.format(i + 1)].value for i in range(nwbins)])
    intrinsic_alignment_A1 = params['DES_AIA1'].value
    intrinsic_alignment_A2 = params['DES_AIA2'].value
    intrinsic_alignment_bTA = params['DES_bTA'].value
    intrinsic_alignment_alpha1 = params['DES_alphaIA1'].value
    intrinsic_alignment_alpha2 = params['DES_alphaIA2'].value
    intrinsic_alignment_z0 = params['DES_z0IA'].value

    a_s = 1/(1 + zs)
    lnD_of_lna_spline = InterpolatedUnivariateSpline(np.log(a_s)[::-1], np.log(D_growth)[::-1])
    f = (lnD_of_lna_spline.derivative())(np.log(a_s))
    f_of_chi_spline = InterpolatedUnivariateSpline(chis, f)

    qgalm = None
    nz_lens = None

    for b in range(nwbins):
        zshift = zs - lens_photoz_errors[b]
        zbin = interp1d(zshift, zs, zbin_w_sp[b], method='cubic')
        zbin = jnp.where(zshift < 0, 0, zbin)
        zbin = jnp.nan_to_num(zbin, nan=0.0)
        zbin /= (jnp.sum(zbin) - 0.5 * (zbin[0] + zbin[-1])) * (zs[1]-zs[0])
        zmean = jnp.average(zs, weights=zbin)
        z_bias = lens_photoz_width[b]*(zs-zmean)+zmean
        zbin = jnp.where(z_bias < 0, 0, zbin)
        nz_lens_width = interp1d(zs, z_bias, zbin, method='cubic')
        nz_lens_width = jnp.where(z_bias < 0, 0, nz_lens_width)
        nz_lens_width = jnp.nan_to_num(nz_lens_width, nan=0.0)
        nz_lens_width /= (jnp.sum(nz_lens_width) - 0.5 * (nz_lens_width[0] + nz_lens_width[-1])) * (zs[1]-zs[0])
        nz_lens = jnp.vstack([nz_lens, nz_lens_width]) if nz_lens is not None else nz_lens_width
        n_chi = Hs * nz_lens_width
        chi_i = chis[:, None]
        chi_j = chis[None, :]
        mask = chi_j >= chi_i
        kernel = (1 - chi_i / chi_j) * dchis[None, :]
        kernel = jnp.where(mask, kernel, 0.0)
        qgalm_b = jnp.sum(n_chi[None, :] * kernel, axis=1)
        if qgalm is None:
            qgalm = 3 * omegam * h2 * (1e5 / c) ** 2 * chis * (1 + zs) / 2 * qgalm_b * Ci[b]
            qgal = n_chi * bin_bias[b] + qgalm
        else:
            qgalm = jnp.vstack([qgalm, 3 * omegam * h2 * (1e5 / c) ** 2 * chis * (1 + zs) / 2 * qgalm_b * Ci[b]])
            qgal = jnp.vstack([qgal, n_chi * bin_bias[b] + qgalm[b]])

    Alignment_z = (intrinsic_alignment_A1 *
                    (((1 + zs) / (1 + intrinsic_alignment_z0)) **
                    intrinsic_alignment_alpha1) * 0.0134 / D_growth)
    Alignment_z /= (chis * (1 + zs) * 3 * h2 * (1e5 / c) ** 2 / 2)

    wq = None
    nz_source = None

    for b in range(nzbins):
        zshift = zs - wl_photoz_errors[b]
        zbin = interp1d(zshift, zs, zbin_sp[b], method='cubic')
        zbin = jnp.where(zshift < 0, 0, zbin)
        zbin = jnp.nan_to_num(zbin, nan=0.0)
        zbin /= (jnp.sum(zbin) - 0.5 * (zbin[0] + zbin[-1])) * (zs[1]-zs[0])
        nz_source = jnp.vstack([nz_source, zbin]) if nz_source is not None else zbin
        n_chi = Hs * zbin
        chi_i = chis[:, None]
        chi_j = chis[None, :]
        mask = chi_j >= chi_i
        kernel = (1 - chi_i / chi_j) * dchis[None, :]
        kernel = jnp.where(mask, kernel, 0.0)
        wq_b = jnp.sum(n_chi[None, :] * kernel, axis=1)
        wq = jnp.vstack([wq, wq_b - Alignment_z * n_chi]) if wq is not None else wq_b - Alignment_z * n_chi

    if Weyl:
        qsw = chis * wq
        qs = 3 * omegam * h2 * (1e5 / c) ** 2 * chis * (1 + zs) / 2 * wq
    else:
        qs = 3 * omegam * h2 * (1e5 / c) ** 2 * chis * (1 + zs) / 2 * wq
        qsw = qs

    if fourier == 'legendre':
        ls_cl = jnp.concatenate((jnp.linspace(1, 4, 5), jnp.logspace(jnp.log10(5), jnp.log10(1e5), 80)))
    elif fourier == 'binned_bessels':
        ls_cl = jnp.hstack((jnp.arange(2., 100 - 4 / acc, 4 / acc),
                        jnp.exp(jnp.linspace(jnp.log(100.), jnp.log(l_max),
                                            int(50 * acc)))))
    #use nonLimber where ell < 200
    ls_exact = ls_cl[ls_cl<200]
    # Get the angular power spectra and transform back
    dchifac = dchis / chis ** 2
    tmpnonlimber = None
    tmp = None
    weight = jnp.empty(chis.shape)
    for ix, l in enumerate(ls_cl):
        k = (l + 0.5) / chis
        kh = k/h
        mask = (k >= 1e-4) & (k < 100)
        weight = jnp.where(mask, dchifac, 0.0)
        #non-Limber
        tmpnonlimberp = weight * jnp.diag(PKdelta_NL(kh) - PKdelta_L(kh))/h3
        tmpnonlimber = jnp.vstack([tmpnonlimber, jnp.nan_to_num(tmpnonlimberp, nan=0.0)]) if tmpnonlimber is not None else jnp.nan_to_num(tmpnonlimberp, nan=0.0)
        #Limber
        tmpp = weight * jnp.diag(PKdelta_NL(kh))/h3
        tmp = jnp.vstack([tmp, jnp.nan_to_num(tmpp, nan=0.0)]) if tmp is not None else jnp.nan_to_num(tmpp, nan=0.0)
    if Weyl:
        tmplens = None
        tmpmw = None
        for ix, l in enumerate(ls_cl):
            k = (l + 0.5) / chis
            kh = k/h
            mask = (k >= 1e-4) & (k < 100)
            weight = jnp.where(mask, dchifac, 0.0)
            #Limber
            tmplensp = weight * k**4 * jnp.diag((PKdelta_NL(kh)/PKdelta_L(kh)) * PKWeyl(kh))/h3/4
            tmplens = jnp.vstack([tmplens, jnp.nan_to_num(tmplensp, nan=0.0)]) if tmplens is not None else jnp.nan_to_num(tmplensp, nan=0.0)
        for ix, l in enumerate(ls_cl):
            k = (l + 0.5) / chis
            kh = k/h
            mask = (k >= 1e-4) & (k < 100)
            weight = jnp.where(mask, dchifac, 0.0)
            #Limber
            tmpmwp = weight * k**2 * jnp.diag((PKdelta_NL(kh)/PKdelta_L(kh)) * PKMW(kh))/h3/2
            tmpmw = jnp.vstack([tmpmw, jnp.nan_to_num(tmpmwp, nan=0.0)]) if tmpmw is not None else jnp.nan_to_num(tmpmwp, nan=0.0)
    else:
        tmplens = tmp
        tmpmw = tmp

    if ia_model == 'TATT':
        #TATT model calculation begins
        Alignment_C1 = -(intrinsic_alignment_A1 *(((1 + zs) / (1 + intrinsic_alignment_z0)) **intrinsic_alignment_alpha1) * 0.0134 / D_growth) * omegam
        Alignment_Cdel = intrinsic_alignment_bTA * Alignment_C1
        Alignment_C2 = 5 * (intrinsic_alignment_A2 *(((1 + zs) / (1 + intrinsic_alignment_z0)) **intrinsic_alignment_alpha2) * 0.0134 / D_growth ** 2) * omegam

        kmin = 1e-4
        kIA = np.logspace(np.log10(kmin), np.log10(100 * acc), 1000)
        P_tmp = PKdelta_L(kIA).T[0]/h3
        P = np.nan_to_num(P_tmp, nan=0.0)
        P_IA_dict = fastpt_power(kIA*h, P)
        Growth4 = D_growth ** 4

        P_II_EE = np.empty((ls_cl.shape[0], chis.shape[0]))
        P_II_BB = np.empty((ls_cl.shape[0], chis.shape[0]))
        P_GI_E = np.empty((ls_cl.shape[0], chis.shape[0]))
        weight = np.empty(chis.shape)
        for ix, l in enumerate(ls_cl):
            k = (l + 0.5) / chis
            weight[:] = dchifac
            weight[k < 1e-4] = 0
            weight[k >= 100] = 0

            C1Cdel = 2 * Alignment_C1 * Alignment_Cdel
            C1C2 = 2 * Alignment_C1 * Alignment_C2
            C2Cdel = 2 * Alignment_C2 * Alignment_Cdel
            Cdel2 = Alignment_Cdel ** 2
            C22 = Alignment_C2 ** 2

            P_ta_EE = _spline(kIA*h, P_IA_dict["P_ta_EE"])(k) * Growth4
            P_ta_BB = _spline(kIA*h, P_IA_dict["P_ta_BB"])(k) * Growth4
            P_ta_dE1 = _spline(kIA*h, P_IA_dict["P_ta_dE1"])(k) * Growth4
            P_ta_dE2 = _spline(kIA*h, P_IA_dict["P_ta_dE2"])(k) * Growth4
            P_mix_A = _spline(kIA*h, P_IA_dict["P_mix_A"])(k) * Growth4
            P_mix_B = _spline(kIA*h, P_IA_dict["P_mix_B"])(k) * Growth4
            P_mix_D_EE = _spline(kIA*h, P_IA_dict["P_mix_D_EE"])(k) * Growth4
            P_mix_D_BB = _spline(kIA*h, P_IA_dict["P_mix_D_BB"])(k) * Growth4
            P_tt_EE = _spline(kIA*h, P_IA_dict["P_tt_EE"])(k) * Growth4
            P_tt_BB = _spline(kIA*h, P_IA_dict["P_tt_BB"])(k) * Growth4

            #TA
            P_ta_II_EE = P_ta_EE * Cdel2 + (P_ta_dE1 + P_ta_dE2) * C1Cdel
            P_ta_II_BB = P_ta_BB * Cdel2
            P_ta_GI = (P_ta_dE1 + P_ta_dE2) * Alignment_Cdel

            # TT
            P_tt_II_EE = P_tt_EE * C22
            P_tt_II_BB = P_tt_BB * C22
            P_tt_GI = (P_mix_A + P_mix_B) * Alignment_C2

            # mixed
            P_mix_II_EE = (P_mix_A + P_mix_B) * C1C2 + P_mix_D_EE * C2Cdel
            P_mix_II_BB = P_mix_D_BB * C2Cdel

            #IA power spectrum
            P_II_EE[ix, :] = weight * (P_ta_II_EE + P_tt_II_EE + P_mix_II_EE)
            P_II_BB[ix, :] = weight * (P_ta_II_BB + P_tt_II_BB + P_mix_II_BB)
            P_GI_E[ix, :] = weight * (P_ta_GI + P_tt_GI)
        #TATT model calculation ends

    corrs_th_p = None
    corrs_th_m = None
    corrs_th_w = None
    corrs_th_t = None
    power_th_p = None
    power_th_m = None
    power_th_w = None
    power_th_t = None
    if fourier == 'legendre':
        ls_legender, legendre_cache = get_fourier(fourier)
        P_l, P_l_2, Gp_l, Gm_l = legendre_cache
        for f1 in range(nzbins):
            power_t_p = None
            power_t_m = None
            corrs_t_p = None
            corrs_t_m = None
            for f2 in range(nzbins):
                if (f1, f2) in bin_pairs['xip']:
                    if ia_model == 'TATT':
                        n_chi_1 = Hs * nz_source[f1]
                        n_chi_2 = Hs * nz_source[f2]

                        P_xi_EE = jnp.dot(tmplens, qsw[f1] * qsw[f2]) + jnp.dot(P_II_EE, n_chi_1 * n_chi_2) + jnp.dot(P_GI_E, n_chi_1 * qs[f2] + n_chi_2 * qs[f1])
                        cl_EE = interp1d(ls_legender, ls_cl, P_xi_EE, method='cubic')
                        cl_EE = jnp.nan_to_num(cl_EE, nan=0.0)
                        P_xi_BB = jnp.dot(P_II_BB, n_chi_1 * n_chi_2)
                        cl_BB = interp1d(ls_legender, ls_cl, P_xi_BB, method='cubic')
                        cl_BB = jnp.nan_to_num(cl_BB, nan=0.0)
                        fac = (1 + shear_calibration_parameters[f1]) * (1 + shear_calibration_parameters[f2])
                        power_p = cl_EE + cl_BB
                        power_m = cl_EE - cl_BB
                        corrs_p = jnp.dot(cl_EE + cl_BB, Gp_l) * fac
                        corrs_m = jnp.dot(cl_EE - cl_BB, Gm_l) * fac
                    else:
                        cl = interp1d(ls_legender, ls_cl, jnp.dot(tmplens, qsw[f1] * qsw[f2]), method='cubic')
                        cl = jnp.nan_to_num(cl, nan=0.0)
                        fac = (1 + shear_calibration_parameters[f1]) * (1 + shear_calibration_parameters[f2])
                        power_p = cl
                        power_m = cl
                        corrs_p = jnp.dot(cl, Gp_l) * fac
                        corrs_m = jnp.dot(cl, Gm_l) * fac
                else:
                    power_p = power_m = jnp.empty(Gp_l.shape[0])
                    corrs_p = corrs_m = jnp.empty(Gp_l.shape[1])
                power_t_p = jnp.vstack([power_t_p, power_p]) if power_t_p is not None else power_p
                power_t_m = jnp.vstack([power_t_m, power_m]) if power_t_m is not None else power_m
                corrs_t_p = jnp.vstack([corrs_t_p, corrs_p]) if corrs_t_p is not None else corrs_p
                corrs_t_m = jnp.vstack([corrs_t_m, corrs_m]) if corrs_t_m is not None else corrs_m
            power_th_p = jnp.vstack([power_th_p, power_t_p[None, :, :]]) if power_th_p is not None else power_t_p[None, :, :]
            power_th_m = jnp.vstack([power_th_m, power_t_m[None, :, :]]) if power_th_m is not None else power_t_m[None, :, :]
            corrs_th_p = jnp.vstack([corrs_th_p, corrs_t_p[None, :, :]]) if corrs_th_p is not None else corrs_t_p[None, :, :]
            corrs_th_m = jnp.vstack([corrs_th_m, corrs_t_m[None, :, :]]) if corrs_th_m is not None else corrs_t_m[None, :, :]
        for f1 in range(nwbins):
            power_t_t = None
            corrs_t_t = None
            for f2 in range(nzbins):
                if (f1, f2) in bin_pairs['gammat']:
                    if ia_model == 'TATT':
                        n_chi_2 = Hs * nz_source[f2]

                        cl_limberall = jnp.dot(tmpmw, qsw[f2] * qgal[f1]) + jnp.dot(P_GI_E, n_chi_2 * qgal[f1])
                        cl = interp1d(ls_legender, ls_cl, cl_limberall, method='cubic')
                        cl = jnp.nan_to_num(cl, nan=0.0)
                    else:
                        cl = interp1d(ls_legender, ls_cl, jnp.dot(tmpmw, qgal[f1] * qsw[f2], axis=1), method='cubic')
                        cl = jnp.nan_to_num(cl, nan=0.0)
                    power_t = cl
                    corrs_t = jnp.dot(cl, P_l_2) * (1 + shear_calibration_parameters[f2])
                else:
                    power_t = jnp.empty(P_l_2.shape[0])
                    corrs_t = jnp.empty(P_l_2.shape[1])
                power_t_t = jnp.vstack([power_t_t, power_t]) if power_t_t is not None else power_t
                corrs_t_t = jnp.vstack([corrs_t_t, corrs_t]) if corrs_t_t is not None else corrs_t
            power_th_t = jnp.vstack([power_th_t, power_t_t[None, :, :]]) if power_th_t is not None else power_t_t[None, :, :]
            corrs_th_t = jnp.vstack([corrs_th_t, corrs_t_t[None, :, :]]) if corrs_th_t is not None else corrs_t_t[None, :, :]
        for f1 in range(nwbins):
            power_t_w = None
            corrs_t_w = None
            for f2 in range(nwbins):
                if (f1, f2) in bin_pairs['wtheta']:
                    if not Limber:
                        cl_limberall = jnp.dot(tmp, qgal[f1] * qgal[f2])
                        cl_limberfront = jnp.dot(tmpnonlimber, qgal[f1] * qgal[f2])
                        cl_limber = cl_limberall[ls_cl>=200]
                        cl_front = cl_limberfront[ls_cl<200]
                        assert f1==f2
                        mean_z = np.trapz(zs * nz_lens[f1], zs) / np.trapz(nz_lens[f1], zs)
                        variance_z = np.trapz(nz_lens[f1] * (zs - mean_z)**2, zs) / np.trapz(nz_lens[f1], zs)
                        std_z = jnp.sqrt(variance_z)
                        cl_exact = cl_front + nonLimber(ls_exact, qgal[f1], qgal[f2], chis, D_growth, PKdelta_L, h, std_z, f_of_chi_spline, bin_bias[f1], bin_bias[f2])
                        cl_p = jnp.concatenate((cl_exact, cl_limber))
                        cl = interp1d(ls_legender, ls_cl, cl_p, method='cubic')
                        cl = jnp.nan_to_num(cl, nan=0.0)
                    else:
                        cl = interp1d(ls_legender, ls_cl, jnp.dot(tmp, qgal[f1] * qgal[f2]), method='cubic')
                    power_w = cl
                    corrs_w = jnp.dot(cl, P_l)
                else:
                    power_w = jnp.empty(P_l.shape[0])
                    corrs_w = jnp.empty(P_l.shape[1])
                power_t_w = jnp.vstack([power_t_w, power_w]) if power_t_w is not None else power_w
                corrs_t_w = jnp.vstack([corrs_t_w, corrs_w]) if corrs_t_w is not None else corrs_w
            power_th_w = jnp.vstack([power_th_w, power_t_w[None, :, :]]) if power_th_w is not None else power_t_w[None, :, :]
            corrs_th_w = jnp.vstack([corrs_th_w, corrs_t_w[None, :, :]]) if corrs_th_w is not None else corrs_t_w[None, :, :]
    elif fourier == 'binned_bessels':
        ls_bessel, bessel_cache = get_fourier(fourier)
        j0s, j2s, j4s = bessel_cache
        ls_bessel = ls_bessel
        for f1 in range(nzbins):
            power_t_p = None
            power_t_m = None
            corrs_t_p = None
            corrs_t_m = None
            for f2 in range(nzbins):
                if (f1, f2) in bin_pairs['xip']:
                    if ia_model == 'TATT':
                        n_chi_1 = Hs * nz_source[f1]
                        n_chi_2 = Hs * nz_source[f2]

                        P_xi_EE = jnp.dot(tmplens, qsw[f1] * qsw[f2]) + jnp.dot(P_II_EE, n_chi_1 * n_chi_2) + jnp.dot(P_GI_E, n_chi_1 * qs[f2] + n_chi_2 * qs[f1])
                        cl_EE = interp1d(ls_bessel, ls_cl, P_xi_EE, method='cubic')
                        cl_EE = jnp.nan_to_num(cl_EE, nan=0.0)
                        P_xi_BB = jnp.dot(P_II_BB, n_chi_1 * n_chi_2)
                        cl_BB = interp1d(ls_bessel, ls_cl, P_xi_BB, method='cubic')
                        cl_BB = jnp.nan_to_num(cl_BB, nan=0.0)
                        fac = (1 + shear_calibration_parameters[f1]) * (1 + shear_calibration_parameters[f2])
                        power_p = cl_EE + cl_BB
                        power_m = cl_EE - cl_BB
                        corrs_p = jnp.dot(cl_EE + cl_BB, j0s) * fac
                        corrs_m = jnp.dot(cl_EE - cl_BB, j4s) * fac
                    else:
                        cl = interp1d(ls_bessel, ls_cl, jnp.dot(tmplens, qsw[f1] * qsw[f2]), method='cubic')
                        cl = jnp.nan_to_num(cl, nan=0.0)
                        fac = (1 + shear_calibration_parameters[f1]) * (1 + shear_calibration_parameters[f2])
                        power_p = cl
                        power_m = cl
                        corrs_p = jnp.dot(cl, j0s) * fac
                        corrs_m = jnp.dot(cl, j4s) * fac
                else:
                    power_p = jnp.empty(Gp_l.shape[0])
                    power_m = jnp.empty(Gp_l.shape[0])
                    corrs_p = jnp.empty(Gp_l.shape[1])
                    corrs_m = jnp.empty(Gp_l.shape[1])
                power_t_p = jnp.vstack([power_t_p, power_p]) if power_t_p is not None else power_p
                power_t_m = jnp.vstack([power_t_m, power_m]) if power_t_m is not None else power_m
                corrs_t_p = jnp.vstack([corrs_t_p, corrs_p]) if corrs_t_p is not None else corrs_p
                corrs_t_m = jnp.vstack([corrs_t_m, corrs_m]) if corrs_t_m is not None else corrs_m
            power_th_p = jnp.vstack([power_th_p, power_t_p[None, :, :]]) if power_th_p is not None else power_t_p[None, :, :]
            power_th_m = jnp.vstack([power_th_m, power_t_m[None, :, :]]) if power_th_m is not None else power_t_m[None, :, :]
            corrs_th_p = jnp.vstack([corrs_th_p, corrs_t_p[None, :, :]]) if corrs_th_p is not None else corrs_t_p[None, :, :]
            corrs_th_m = jnp.vstack([corrs_th_m, corrs_t_m[None, :, :]]) if corrs_th_m is not None else corrs_t_m[None, :, :]
        for f1 in range(nwbins):
            power_t_t = None
            corrs_t_t = None
            for f2 in range(nzbins):
                if (f1, f2) in bin_pairs['gammat']:
                    if ia_model == 'TATT':
                        n_chi_2 = Hs * nz_source[f2]

                        cl_limberall = jnp.dot(tmpmw, qsw[f2] * qgal[f1]) + jnp.dot(P_GI_E, n_chi_2 * qgal[f1])
                        cl = interp1d(ls_bessel, ls_cl, cl_limberall, method='cubic')
                        cl = jnp.nan_to_num(cl, nan=0.0)
                    else:
                        cl = interp1d(ls_bessel, ls_cl, jnp.dot(tmpmw, qgal[f1] * qsw[f2], axis=1), method='cubic')
                        cl = jnp.nan_to_num(cl, nan=0.0)
                    power_t = cl
                    corrs_t = jnp.dot(cl, j2s) * (1 + shear_calibration_parameters[f2])
                else:
                    power_t = jnp.empty(P_l_2.shape[0])
                    corrs_t = jnp.empty(P_l_2.shape[1])
                power_t_t = jnp.vstack([power_t_t, power_t]) if power_t_t is not None else power_t
                corrs_t_t = jnp.vstack([corrs_t_t, corrs_t]) if corrs_t_t is not None else corrs_t
            power_th_t = jnp.vstack([power_th_t, power_t_t[None, :, :]]) if power_th_t is not None else power_t_t[None, :, :]
            corrs_th_t = jnp.vstack([corrs_th_t, corrs_t_t[None, :, :]]) if corrs_th_t is not None else corrs_t_t[None, :, :]
        for f1 in range(nwbins):
            power_t_w = None
            corrs_t_w = None
            for f2 in range(nwbins):
                if (f1, f2) in bin_pairs['wtheta']:
                    if not Limber:
                        cl_limberall = jnp.dot(tmp, qgal[f1] * qgal[f2])
                        cl_limberfront = jnp.dot(tmpnonlimber, qgal[f1] * qgal[f2])
                        cl_limber = cl_limberall[ls_cl>=200]
                        cl_front = cl_limberfront[ls_cl<200]
                        assert f1==f2
                        mean_z = np.trapz(zs * nz_lens[f1], zs) / np.trapz(nz_lens[f1], zs)
                        variance_z = np.trapz(nz_lens[f1] * (zs - mean_z)**2, zs) / np.trapz(nz_lens[f1], zs)
                        std_z = jnp.sqrt(variance_z)
                        cl_exact = cl_front + nonLimber(ls_exact, qgal[f1], qgal[f2], chis, D_growth, PKdelta_L, h, std_z, f_of_chi_spline, bin_bias[f1], bin_bias[f2])
                        cl_p = jnp.concatenate((cl_exact, cl_limber))
                        cl = interp1d(ls_bessel, ls_cl, cl_p, method='cubic')
                        cl = jnp.nan_to_num(cl, nan=0.0)
                    else:
                        cl = interp1d(ls_bessel, ls_cl, jnp.dot(tmp, qgal[f1] * qgal[f2]), method='cubic')
                        cl = jnp.nan_to_num(cl, nan=0.0)
                    power_w = cl
                    corrs_w = jnp.dot(cl, j0s)
                else:
                    power_w = jnp.empty(P_l.shape[0])
                    corrs_w = jnp.empty(P_l.shape[1])
                power_t_w = jnp.vstack([power_t_w, power_w]) if power_t_w is not None else power_w
                corrs_t_w = jnp.vstack([corrs_t_w, corrs_w]) if corrs_t_w is not None else corrs_w
            power_th_w = jnp.vstack([power_th_w, power_t_w[None, :, :]]) if power_th_w is not None else power_t_w[None, :, :]
            corrs_th_w = jnp.vstack([corrs_th_w, corrs_t_w[None, :, :]]) if corrs_th_w is not None else corrs_t_w[None, :, :]
    TwoPT = {}
    if fourier == 'legendre':
        TwoPT['ell'] = np.arange(P_l.shape[0])
    elif fourier == 'binned_bessels':
        TwoPT['ell'] = ls_bessel
    TwoPT['xip'] = corrs_th_p
    TwoPT['xim'] = corrs_th_m
    TwoPT['gammat'] = corrs_th_t
    TwoPT['wtheta'] = corrs_th_w
    TwoPT['cl_xip'] = power_th_p
    TwoPT['cl_xim'] = power_th_m
    TwoPT['cl_gammat'] = power_th_t
    TwoPT['cl_wtheta'] = power_th_w
    TwoPT['chis'] = chis
    TwoPT['Hs'] = Hs
    TwoPT['nz_lens'] = nz_lens
    TwoPT['nz_source'] = nz_source
    return TwoPT

#nonLimber calculation depend on 1911.11947
def nonLimber(ells, kernel1_vals, kernel2_vals, chis, D_growth, PK, h, sigma, f_interp=None, bias1=None, bias2=None):
    chi_pad_upper=1.
    chi_pad_lower=1.
    chi_extrap_upper=1.
    chi_extrap_lower=1.
    
    if np.isnan(sigma):
        dlogchi = np.log(chis[-1]/chis[-2])
    else:
        dlogchi = sigma/3.5
    log_chimin, log_chimax = np.log(chis[0]), np.log(chis[-1])
    nchi = np.ceil((log_chimax-log_chimin)/dlogchi).astype(int)
    log_chi_vals = jnp.linspace(log_chimin, log_chimax, nchi)
    chi_vals = jnp.exp(log_chi_vals)

    if chi_pad_upper>0.:
        assert chi_pad_lower==chi_pad_upper
        N_pad = (np.ceil(float(chi_pad_upper)/dlogchi)).astype(int)
    else:
        N_pad = 0
    if chi_extrap_upper>0.:
        N_extrap_upper = (np.ceil(float(chi_extrap_upper)/dlogchi)).astype(int)
    else:
        N_extrap_upper = 0
    if chi_extrap_lower>0.:
        N_extrap_lower = (np.ceil(float(chi_extrap_lower)/dlogchi)).astype(int)
    else:
        N_extrap_lower = 0

    if np.all(kernel1_vals == kernel2_vals):
        w1p = kernel1_vals * D_growth * chis
        #w1 = _spline(chis, w1p)(chi_vals)
        w1 = interp1d(chi_vals, chis, w1p, method='cubic')
        w1 = jnp.nan_to_num(w1, nan=0.0)
        w2 = w1
    else:
        w1p = kernel1_vals * D_growth * chis
        w2p = kernel2_vals * D_growth * chis
        #w1 = _spline(chis, w1p)(chi_vals)
        #w2 = _spline(chis, w2p)(chi_vals)
        w1 = interp1d(chi_vals, chis, w1p, method='cubic')
        w1 = jnp.nan_to_num(w1, nan=0.0)
        w2 = interp1d(chi_vals, chis, w2p, method='cubic')
        w2 = jnp.nan_to_num(w2, nan=0.0)
    if f_interp is not None:
        if bias1 !=0 and bias2 != 0:
            f_vals = f_interp(chi_vals)
            w1_rsd = w1 * f_vals/bias1
            w2_rsd = w2 * f_vals/bias2
        else:
            f_vals = f_interp(chi_vals)
            w1_rsd = w1 * f_vals
            w2_rsd = w2 * f_vals
    
    cell = []
    for i_ell, ell in enumerate(ells):
        if np.all(kernel1_vals == kernel2_vals):
            k_vals, I_1 = FFT(chi_vals, w1, ell, N_extrap_upper, N_extrap_lower, N_pad)
            I_2 = I_1
            if f_interp is not None:
                k_vals_check, I_1_rsd = FFT(chi_vals, w1_rsd, ell, N_extrap_upper, N_extrap_lower, N_pad, nu=1.1)
                assert jnp.allclose(k_vals_check, k_vals)
                I_1 = I_1 - I_1_rsd
                I_2 = I_1
        else:
            k_vals, I_1 = FFT(chi_vals, w1, ell, N_extrap_upper, N_extrap_lower, N_pad)
            k_vals, I_2 = FFT(chi_vals, w2, ell, N_extrap_upper, N_extrap_lower, N_pad)
            if f_interp is not None:
                k_vals_check, I_1_rsd = FFT(chi_vals, w1_rsd, ell, N_extrap_upper, N_extrap_lower, N_pad, nu=1.1)
                k_vals_check, I_2_rsd = FFT(chi_vals, w2_rsd, ell, N_extrap_upper, N_extrap_lower, N_pad, nu=1.1)
                assert jnp.allclose(k_vals_check, k_vals)
                I_1 = I_1 - I_1_rsd
                I_2 = I_2 - I_2_rsd
        logk_vals = jnp.log(k_vals)
        pk_vals = PK(k_vals/h).T[0]/h**3
        integrand_vals = k_vals * k_vals * k_vals * pk_vals * I_1 * I_2
        #Spline and integrate the integrand.
        integrand_interp = InterpolatedUnivariateSpline(logk_vals, integrand_vals)
        integral = integrand_interp.integral(logk_vals.min(), logk_vals.max())
        integral *= 2./np.pi
        cell.append(integral)
    return np.array(cell)

def FFT(x, fx, ell, N_extrap_high = 0, N_extrap_low = 0, N_pad = 0, nu = 1.0):
    dlnx = jnp.log(x[1]/x[0])
    c_window_width = 0.25

    # extrapolate x and f(x) linearly in log(x), and log(f(x))
    x = log_extrap(x, N_extrap_low, N_extrap_high)
    fx = log_extrap(fx, N_extrap_low, N_extrap_high)
    N = jnp.size(x)

    # zero-padding
    if(N_pad):
        pad = jnp.zeros(N_pad)
        x = log_extrap(x, N_pad, N_pad)
        fx = jnp.hstack((pad, fx, pad))
        N += 2*N_pad
        N_extrap_high += N_pad
        N_extrap_low += N_pad

    if(N%2==1): # Make sure the array sizes are even
        x= x[:-1]
        fx=fx[:-1]
        N -= 1
        if(N_pad):
            N_extrap_high -=1

    f_b=fx * x**(-nu)
    c_m=jnp.fft.rfft(f_b)
    m=jnp.arange(0,N//2+1) 
    c_m = c_m * c_window(m, int(c_window_width*N//2.) )
    eta_m = 2*jnp.pi/(float(N)*dlnx) * m
    
    z_ar = nu + 1j*eta_m
    y = (ell+1.) / x[::-1]
    if nu ==1:
        h_m = c_m * (x[0]*y[0])**(-1j*eta_m) * g_l(ell, z_ar)
    else:
        h_m = c_m * (x[0]*y[0])**(-1j*eta_m) * g_l_2(ell, z_ar)

    Fy = jnp.fft.irfft(jnp.conj(h_m)) * y**(-nu) * jnp.sqrt(jnp.pi)/4.
    if jnp.any(jnp.isnan(Fy)):
        print("found nans in Fy for ell=%s"%ell)
    return (y[N_extrap_high:N-N_extrap_low], Fy[N_extrap_high:N-N_extrap_low])

def c_window(n, n_cut):
    n_right = n[-1] - n_cut
    mask = (n > n_right)
    theta_right=(n[-1]-n)/float(n[-1]-n_right-1) 
    taper = theta_right - 1/(2*jnp.pi) * jnp.sin(2*jnp.pi * theta_right)
    W = jnp.where(mask, taper, 1.0)
    return W
    
def log_extrap(x, N_extrap_low, N_extrap_high):
    low_x = jnp.array([])
    high_x = jnp.array([])
    if(N_extrap_low):
        if x[1] <= 1e-30 or x[0] <= 1e-30:
            low_x = jnp.zeros(N_extrap_low)
        else:
            dlnx_low = jnp.log(x[1]/x[0])
            low_x = x[0] * jnp.exp(dlnx_low * jnp.arange(-N_extrap_low, 0) )
    if(N_extrap_high):
        if x[-1] <= 1e-30 or x[-2] <= 1e-30:
            high_x = jnp.zeros(N_extrap_high)
        else:
            dlnx_high= jnp.log(x[-1]/x[-2])
            high_x = x[-1] * jnp.exp(dlnx_high * jnp.arange(1, N_extrap_high+1) )
    x_extrap = jnp.hstack((low_x, x, high_x))
    return x_extrap

def g_l(l, z_array):
    gl = 2.**z_array * g_m_vals(l+0.5,z_array-1.5)
    return gl

def g_l_2(l,z_array):
    gl2 = 2.**(z_array-2) *(z_array -1)*(z_array -2)* g_m_vals(l+0.5,z_array-3.5)
    return gl2

def g_m_vals(mu, q):
    imag_q = jnp.imag(q)
    cut = 200

    cond_asym = (jnp.abs(imag_q) + jnp.abs(mu) > cut)
    cond_good = (jnp.abs(imag_q) + jnp.abs(mu) <= cut) & (q != mu + 1 + 0j)
    cond_zero = (q == mu + 1 + 0j)

    # asym variables
    asym_q = q
    asym_plus  = (mu + 1 + asym_q) / 2
    asym_minus = (mu + 1 - asym_q) / 2

    # exact variables
    alpha_plus  = (mu + 1 + q) / 2
    alpha_minus = (mu + 1 - q) / 2

    g_exact = special.gamma(alpha_plus) / special.gamma(alpha_minus)

    g_asym = jnp.exp(
        (asym_plus - 0.5) * jnp.log(asym_plus)
        - (asym_minus - 0.5) * jnp.log(asym_minus)
        - asym_q
        + 1/12 * (1/asym_plus - 1/asym_minus)
        + 1/360 * (1/asym_minus**3 - 1/asym_plus**3)
        + 1/1260 * (1/asym_plus**5 - 1/asym_minus**5)
    )

    g_zero = 0.0 + 0.0j

    # final combine using jnp.where (JAX-friendly)
    g_m = jnp.where(cond_zero, g_zero,
        jnp.where(cond_asym, g_asym,
            jnp.where(cond_good, g_exact, 0.0 + 0.0j)
        )
    )
    return g_m
#nonLimber calculation ends

#Legendre function
def Pl_rec_binav(ells, cost_min, cost_max):
    """Calculate average Pl"""
    Pl_binav = np.zeros(len(ells))
    Pl_binav[0] = 1.
    # coefficients that are a function of ell only
    ell = ells[1:]
    coeff = 1./(2.*ell+1.)
    # computation of legendre polynomials
    # --- this computes all polynomials of order 0 to ell_max+1 and for all ell's
    lpns_min = special.lpn(ell[-1]+1, cost_min)[0]
    lpns_max = special.lpn(ell[-1]+1, cost_max)[0]
    # terms in the numerator of average Pl
    term_lm1 = lpns_max[:-2] - lpns_min[:-2]
    term_lp1 = lpns_max[2:] - lpns_min[2:]
    # denominator in average Pl
    dcost = cost_max-cost_min
    # computation of bin-averaged Pl(ell)
    Pl_binav[ell] = coeff * (term_lp1 - term_lm1) / dcost
    return Pl_binav

def get_legfactors_00_binav(ells, theta_edges):
    n_ell, n_theta = len(ells), len(theta_edges)-1
    #theta_edges = theta_bin_means_to_edges(thetas) # this does geometric mean
    legfacs = np.zeros((n_theta, n_ell))
    ell_factor = np.zeros(len(ells))
    ell_factor[1:] = (2 * ells[1:] + 1) / 4. / np.pi
    for it, t in enumerate(theta_edges[1:]):
        t_min = theta_edges[it]
        t_max = t
        cost_min = np.cos(t_min) # thetas are already converted to radians
        cost_max = np.cos(t_max)
        Pl = Pl_rec_binav(ells, cost_min, cost_max)
        legfacs[it] = Pl * ell_factor
    return legfacs

def P2l_rec_binav(ells, cost_min, cost_max):
    """Calculate P2l using recurrence relation for normalised P2l"""
    P2l_binav = np.zeros(len(ells))
    P2l_binav[0] = 0.
    P2l_binav[1] = 0.
    # coefficients that are a function of ell only
    ell = ells[2:]
    coeff_lm1 = ell+2./(2.*ell+1.)
    coeff_lp1 = 2./(2.*ell+1.)
    coeff_l   = 2.-ell
    # computation of legendre polynomials
    # --- this computes all polynomials of order 0 to ell_max+1 and for all ell's
    lpns_min = special.lpn(ell[-1]+1, cost_min)[0][1:]
    lpns_max = special.lpn(ell[-1]+1, cost_max)[0][1:]
    # terms in the numerator of average P2l
    term_lm1 = coeff_lm1 * (lpns_max[:-2]-lpns_min[:-2])
    term_lp1 = coeff_lp1 * (lpns_max[2:]-lpns_min[2:])
    term_l   = coeff_l   * (cost_max*lpns_max[1:-1]-cost_min*lpns_min[1:-1])
    # denominator in average P2l
    dcost = cost_max-cost_min
    # computation of bin-averaged P2l(ell)
    P2l_binav[ell] = (term_lm1 + term_l - term_lp1) / dcost
    return P2l_binav

def get_legfactors_02_binav(ells, theta_edges):
    n_ell, n_theta = len(ells), len(theta_edges)-1
    #theta_edges = theta_bin_means_to_edges(thetas) # this does geometric mean
    legfacs = np.zeros((n_theta, n_ell))
    ell_factor = np.zeros(len(ells))
    ell_factor[1:] = (2 * ells[1:] + 1) / 4. / np.pi / ells[1:] / (ells[1:] + 1)
    for it, t in enumerate(theta_edges[1:]):
        t_min = theta_edges[it]
        t_max = t
        cost_min = np.cos(t_min) # thetas are already converted to radians
        cost_max = np.cos(t_max)
        P2l = P2l_rec_binav(ells, cost_min, cost_max)
        legfacs[it] = P2l * ell_factor
    return legfacs

def Gp_plus_minus_Gm_binav(ells, cost_min, cost_max):
    """Calculate bin-averaged G_{l,2}^{+/-}"""
    Gp_plus_Gm  = np.zeros(len(ells))
    Gp_minus_Gm = np.zeros(len(ells))

    # for ell=0,1 it is 0
    Gp_plus_Gm[0:1]  = 0.
    Gp_minus_Gm[0:1] = 0.

    # for the rest of ell's compute equation (5.8) in https://arxiv.org/abs/1911.11947
    ell = ells[2:]
    #---coefficients including only P_l
    coeff_lm1    = -ell*(ell-1.)/2. * (ell+2./(2.*ell+1)) - (ell+2.)
    coeff_lp1    =  ell*(ell-1.)/(2.*ell+1.)
    coeff_l      = -ell*(ell-1.)*(2.-ell)/2.
    coeff_l_plus = -2.*(ell-1.)
    #---coefficients including dP_l/dx
    coeff_dlm1_plus = -2.*(ell+2.)
    coeff_xdl_plus  = 2.*(ell-1.)
    coeff_xdlm1     = ell+2.
    coeff_dl        = 4.-ell

    # computation of legendre polynomials
    #---this computes all polynomials of order 0 to ell_max+1 and for all ell's
    lpns_min  = special.lpn(ell[-1]+1, cost_min)[0][1:]
    lpns_max  = special.lpn(ell[-1]+1, cost_max)[0][1:]
    dlpns_min = special.lpn(ell[-1]+1, cost_min)[1][1:]
    dlpns_max = special.lpn(ell[-1]+1, cost_max)[1][1:]

    # denominator in average
    dcost = cost_max-cost_min

    # numerator in average
    #---common part in both plus and minus
    common_part  = coeff_lm1*(lpns_max[:-2]-lpns_min[:-2])
    common_part += coeff_l*(cost_max*lpns_max[1:-1] - cost_min*lpns_min[1:-1])
    common_part += coeff_lp1*(lpns_max[2:]-lpns_min[2:])
    common_part += coeff_dl*(dlpns_max[1:-1]-dlpns_min[1:-1])
    common_part += coeff_xdlm1*(cost_max*dlpns_max[:-2]-cost_min*dlpns_min[:-2])
    #---plus
    Gp_plus_Gm_extra  = coeff_xdl_plus*(cost_max*dlpns_max[1:-1]-cost_min*dlpns_min[1:-1])
    Gp_plus_Gm_extra += coeff_l_plus*(lpns_max[1:-1] - lpns_min[1:-1])
    Gp_plus_Gm_extra += coeff_dlm1_plus*(dlpns_max[:-2]-dlpns_min[:-2])
    Gp_plus_Gm[2:] = common_part + Gp_plus_Gm_extra
    Gp_plus_Gm /= dcost
    #---minus
    Gp_minus_Gm_extra = -Gp_plus_Gm_extra
    Gp_minus_Gm[2:] = common_part + Gp_minus_Gm_extra
    Gp_minus_Gm /= dcost

    return Gp_plus_Gm, Gp_minus_Gm

def get_legfactors_22_binav(ells, theta_edges):
    n_ell, n_theta = len(ells), len(theta_edges)-1
    #theta_edges = theta_bin_means_to_edges(thetas) # this does geometric mean
    leg_factors_p  = np.zeros((n_theta, n_ell))
    leg_factors_m = np.zeros((n_theta, n_ell))
    ell_factor = np.zeros(len(ells))
    ell_factor[2:] = (2 * ells[2:] + 1) / 2. / np.pi / ells[2:] / ells[2:] / (ells[2:]+1.) / (ells[2:]+1.)
    for it, t in enumerate(theta_edges[1:]):
        t_min = theta_edges[it]
        t_max = t
        cost_min = np.cos(t_min) # thetas are already converted to radians
        cost_max = np.cos(t_max)
        gp, gm = Gp_plus_minus_Gm_binav(ells, cost_min, cost_max)
        leg_factors_p[it] = gp * ell_factor
        leg_factors_m[it] = gm * ell_factor
    return [leg_factors_p, leg_factors_m]

def apply_filter(ell_max, high_l_filter, legfacs):
    f = sin_filter( ell_max, ell_max*high_l_filter)
    return legfacs * np.tile(f, (legfacs.shape[0],1))

def sin_filter(ell_max, ell_right):
    ells = np.arange(ell_max+1)
    y = (ell_max-ells)/(ell_max-ell_right)
    return np.where( ells>ell_right, y - np.sin(2*np.pi*y)/2/np.pi, np.ones(len(ells)) )

#fastpt to calulate TATT IA power spectrum
def fastpt_power(k, P):
    fastpt = FASTPT(k, to_do=['IA_all'], low_extrap=-5, high_extrap=3, n_pad=int(len(k))) 
    C_window = 0.75

    P_IA_dict = {}
    IA_tt = fastpt.IA_tt(P, C_window=C_window)
    P_IA_dict["P_tt_EE"] = IA_tt[0]##
    P_IA_dict["P_tt_BB"] = IA_tt[1]##
    IA_ta = fastpt.IA_ta(P, C_window=C_window)
    P_IA_dict["P_ta_dE1"] = IA_ta[0]
    P_IA_dict["P_ta_dE2"] = IA_ta[1]
    P_IA_dict["P_ta_EE"] = IA_ta[2]##
    P_IA_dict["P_ta_BB"] = IA_ta[3]##
    IA_mix = fastpt.IA_mix(P, C_window=C_window)
    P_IA_dict["P_mix_A"] = IA_mix[0]
    P_IA_dict["P_mix_B"] = IA_mix[1]
    P_IA_dict["P_mix_D_EE"] = IA_mix[2]##
    P_IA_dict["P_mix_D_BB"] = IA_mix[3]##

    return P_IA_dict

def get_fourier(fourier, acc=1, l_max=40000):
    if fourier == 'legendre':
        theta = np.loadtxt('/Users/illyasviel/Desktop/Cosmology/DES/des-y3/des-y3/des3/DES_3YR_final_theta_bins.dat')
        theta_edges = theta[:,0]
        theta_edges = np.append(theta_edges, theta[-1,1])
        theta_bins = theta[:,2]
        theta_bins_radians = theta_bins / 60 * np.pi / 180
        theta_edges_radians = theta_edges / 60 * np.pi / 180
    elif fourier == 'binned_bessels':
        theta_bins = np.loadtxt('/Users/illyasviel/Desktop/Cosmology/DES/des-y3/des-y3/des3/DES_3YR_final_theta_bins.dat')
        theta_bins_radians = theta_bins / 60 * np.pi / 180

    # Note hankel assumes integral starts at ell=0
    # (though could change spline to zero at zero).
    # At percent level it matters what is assumed
    if fourier == 'binned_bessels':
        # Approximate bessel integral as binned smooth C_L against integrals of
        # bessel in each bin. Here we crudely precompute an approximation to the
        # bessel integral by brute force
        dls = np.diff(np.unique((np.exp(np.linspace(
            np.log(1.), np.log(l_max), int(500 * acc)))).astype(int)))
        groups = []
        ell = 2  # ell_min
        ls_bessel = np.zeros(dls.size)
        for i, dlx in enumerate(dls):
            ls_bessel[i] = (2 * ell + dlx - 1) / 2.
            groups.append(np.arange(ell, ell + dlx))
            ell += dlx
        js = np.empty((3, ls_bessel.size, len(theta_bins_radians)))
        bigell = np.arange(0, l_max + 1, dtype=np.float64)
        for i, theta in enumerate(theta_bins_radians):
            bigx = bigell * theta
            for ix, nu in enumerate([0, 2, 4]):
                bigj = special.jn(nu, bigx) * bigell / (2 * np.pi)
                for j, g in enumerate(groups):
                    js[ix, j, i] = np.sum(bigj[g])
        bessel_cache = js[0, :, :], js[1, :, :], js[2, :, :]
        return ls_bessel, bessel_cache
    elif fourier == 'legendre':
        high_l_filter = 0.75
        ls_legender = np.arange(l_max + 1)
        P_l = get_legfactors_00_binav(ls_legender, theta_edges_radians)
        P_l = apply_filter(l_max, high_l_filter, P_l)
        P_l_2 = get_legfactors_02_binav(ls_legender, theta_edges_radians)
        P_l_2 = apply_filter(l_max, high_l_filter, P_l_2)
        Gp_l, Gm_l = get_legfactors_22_binav(ls_legender, theta_edges_radians)
        Gp_l = apply_filter(l_max, high_l_filter, Gp_l)
        Gm_l = apply_filter(l_max, high_l_filter, Gm_l)
        legendre_cache = P_l.T, P_l_2.T, Gp_l.T, Gm_l.T
        return ls_legender, legendre_cache