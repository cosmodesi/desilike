# Installation Instructions

Before running any of the notebooks, make sure to set up the environment by following these steps:

1. **Create the conda environment** using the provided `environment.yml`:
   ```bash
   mamba env create -f environment.yml
   conda activate desilike_env
2. **Run the installation script to install the required Python packages from git:**
   ```bash
   bash install_desilike.sh
  
Once these steps are complete, all the notebooks below should run correctly as mentioned below.   

# Notebooks

- basic_examples.ipynb: desilike basics, how to create a calculator (theory or likelihood), define parameters, fit and sample likelihood
- kaiser_implementation_examples.ipynb: implementation of a Kaiser theory, down to the likelihood
- bao_examples.ipynb: BAO fit, plot BAO wiggles, estimate detection level, sample the BAO likelihood, inference of Omega_m
- fs_shapefit_examples.ipynb: full shape fits with the shapefit parameterization
- fs_direct_examples.ipynb: full shape fits with the direct parameterization (i.e. base cosmological parameters)
- compression_examples.ipynb: constraints on cosmological parameters from BAO and shapefit constraints, comparison to direct constraints
- png_examples.ipynb: local primordial non-gaussianity fits
- turnover_examples.ipynb: turnover scale fits, inference of H0
- fisher_examples.ipynb: BAO Fisher forecasts, combination with Planck CMB
- fisher_desi.ipynb: BAO and RSD Fisher forecasts for DESI
- fisher_planck_examples.ipynb: Fisher forecasts with Planck CMB, comparing Fisher to Planck covariance matrices, and MCMC with emulation of theory Cl's
- window_standard_compression_examples.ipynb: interpretation standard parameters (qpar, qper, df) as compressed statistics, estimation of the associated window matrix
- window_shapefit_compression_examples.ipynb: interpretation shapefit parameters (qpar, qper, df, dm) as compressed statistics, estimation of the associated window matrix
- window_wigglesplit_compression_examples.ipynb: interpretation shapefit parameters (qbao, qap, df, dm) as compressed statistics, estimation of the associated window matrix
- flexible_bao_examples.ipynb: tests of new, flexible broadband parameterization for BAO
