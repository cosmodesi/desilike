import os
import numpy as np
from desilike import plotting, utils
from .base import BaseSNLikelihood

class PantheonPlusSNLikelihood(BaseSNLikelihood):
    """
    Likelihood for Pantheon+ (without SH0ES) type Ia supernovae sample.

    Reference
    ---------
    https://arxiv.org/abs/2202.04077 

    Parameters
    ----------
    data_dir : str, Path, default=None
        Data directory. Defaults to path saved in desilike's configuration,
        as provided by :class:`Installer` if likelihood has been installed.
    """
    config_fn = 'pantheonplus.yaml'
    installer_section = 'PantheonPlusSNLikelihood'

    def initialize(self,*args, **kwargs):
        super(PantheonPlusSNLikelihood, self).initialize(*args, **kwargs)
        self.precision = utils.inv(self.covariance)
        self.std = np.diag(self.covariance)**0.5


    def calculate(self, Mb=0):
        z = self.light_curve_params['zcmb']
        self.flattheory = 5 * np.log10(self.cosmo.luminosity_distance(z)/self.cosmo['h']) + 25
        self.flatdata = self.light_curve_params['mb'] - Mb - 5 * np.log10((1 + self.light_curve_params['zhel']) / (1 + z))
        super(PantheonPlusSNLikelihood, self).calculate()
    
    def read_light_curve_params(self, fn):
        from pandas import read_csv # Uses pandas for faster reading of the datafile
        
        data=read_csv(fn,delim_whitespace=True)
        
        self.N=len(data) # Number of datapoints in the full dataset
        self.filter=(data['zHD']>0.01) # Only those SNe at z>0.01 are used for cosmology
        data=data[self.filter]
        
        return {'zcmb':data['zHD'].to_numpy(),'zhel':data['zHEL'].to_numpy(),'mb':data['m_b_corr'].to_numpy()}
    
    
    def read_covariance(self,fn):
        """
        Run once at the start to build the covariance matrix for the data.
        Borrowed from the cosmosis likelihood implementation
        """
        filename = fn
        print("Loading covariance from {}".format(filename))

        # The file format for the covariance has the first line as an integer
        # indicating the number of covariance elements, and the the subsequent
        # lines being the elements.
        # This function reads in the file and the nasty for loops trim down the covariance
        # to match the only rows of data that are used for cosmology
        with open(filename) as f:
            line = f.readline()
            n = int(len(self.light_curve_params['zcmb']))
            C = np.zeros((n,n))
            ii = -1
            jj = -1
            mine = 999
            maxe = -999
            for i in range(self.N):
                jj = -1
                if self.filter[i]:
                    ii += 1
                for j in range(self.N):
                    if self.filter[j]:
                        jj += 1
                    val = float(f.readline())
                    if self.filter[i]:
                        if self.filter[j]:
                            C[ii,jj] = val
        print('Done')
        return C
    
    @plotting.plotter
    def plot(self):
        """
        Plot Hubble diagram: Hubble residuals as a function of distance.

        fn : str, Path, default=None
        Optionally, path where to save figure.
        If not provided, figure is not saved.

        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

        show : bool, default=False
            If ``True``, show figure.
        """
        from matplotlib import pyplot as plt
        fig, lax = plt.subplots(2, sharex=True, sharey=False, gridspec_kw={'height_ratios': (3, 1)}, figsize=(6, 6), squeeze=True)
        fig.subplots_adjust(hspace=0)
        alpha = 0.3
        argsort = np.argsort(self.light_curve_params['zcmb'])
        zdata = self.light_curve_params['zcmb'][argsort]
        flatdata, flattheory, std = self.flatdata[argsort], self.flattheory[argsort], self.std[argsort]
        lax[0].plot(zdata, flatdata, marker='o', markeredgewidth=0., linestyle='none', alpha=alpha, color='b')
        lax[0].plot(zdata, flattheory, linestyle='-', marker=None, color='k')
        lax[0].set_xscale('log')
        lax[1].errorbar(zdata, flatdata - flattheory, yerr=std, linestyle='none', marker='o', alpha=alpha, color='b')
        lax[0].set_ylabel(r'distance modulus [$\mathrm{mag}$]')
        lax[1].set_ylabel(r'Hubble res. [$\mathrm{mag}$]')
        lax[1].set_xlabel('$z$')
        return lax

    @classmethod
    def install(cls, installer):
        try:
            data_dir = installer[cls.installer_section]['data_dir']
        except KeyError:
            data_dir = installer.data_dir(cls.installer_section)

        from desilike.install import exists_path, download

        data_fn = os.path.join(data_dir, 'Pantheon+SH0ES.dat')

        if installer.reinstall or not exists_path(data_fn):
            github = 'https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/'
            for fn in ['Pantheon+SH0ES.dat', 'Pantheon+SH0ES_STAT+SYS.cov']:
                download(os.path.join(github, fn), os.path.join(data_dir, fn))
            
            #Creates config file to ensure compatibility with Base class            
            config_fn=os.path.join(data_dir, 'config.dataset') 
            try:
                with open(config_fn,'r') as file:
                    txt = file.read()
                # txt = txt.replace('/your-path/', '')
            except FileNotFoundError:
                with open(config_fn, 'w') as file:
                    for text in ['name = PantheonPlus\n','data_file = Pantheon+SH0ES.dat\n','mag_covmat_file = Pantheon+SH0ES_STAT+SYS.cov\n']:
                        file.write(text)
            installer.write({cls.__name__: {'data_dir': data_dir}})
