---
title: Tutorials
subject: Tutorials
short_title: Overview
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - CSIC
      - UCM
      - IGEO
    orcid: 0000-0002-6739-0053
    email: juanjohn@ucm.es
license: CC-BY-4.0
keywords: notation
---


## Datasets

* AEMET
* E-OBS
* ERA5 (TODO)

## Block Maximas


---
## Walk-Throughs I

The first set of walkthroughs use the library.

---
### Stationary IID

This first tutorial showcases how we can learn a stationary model for a group of stations
We use yearly block maximum for 2m Max Temperature based on a dataset from AEMET.

- Part I: EDA [Link](./tutorials_stationary_iid/1_spain_t2m_gevd_eda.ipynb)
- Part II: MCMC Inference [Link](./tutorials_stationary_iid/2a_spain_t2m_gevd_model_mcmc.ipynb)
- Part III: Laplacian Inference [Link](./tutorials_stationary_iid/2a_spain_t2m_gevd_model_lap.ipynb)
- Part IV: Station Analysis + Attribution [Link](./tutorials_stationary_iid/2b_spain_t2m_gevd_analysis_station.ipynb)
- Part V: Regional Analysis [Link](./tutorials_stationary_iid/2c_spain_t2m_gevd_analysis_region.ipynb)

---
### NonStationary IID

This next tutorial showcases how we can learn a non-stationary model for a group of stations
We use yearly block maximum for 2m Max Temperature based on a dataset from AEMET.

-  Part I: EDA [Link](./tutorials_nonstationary_iid/2a_t2m_gevd_mcmc.ipynb)
-  Part II: MCMC Inference [Link](./tutorials_nonstationary_iid/2b_t2m_gevd_mcmc_station.ipynb)
- Part III: Station Analysis [Link](./tutorials_nonstationary_iid/2c_t2m_gevd_mcmc_station_pos.ipynb)
- Part IV: Station Attribution and Predictions [Link](./tutorials_nonstationary_iid/2d_t2m_gevd_mcmc_region.ipynb)

---
## Walk-Throughs I

This set of tutorials are independent and are designed to understand some of the underlying bits and pieces to customizing models. Due to the complexity, I have tried to isolate the model code to make it more didactic and understandable and should only be run from these notebooks.

See:
- [vgp](./tutorials_spatiotemporal_gp/vgp.py)
- [utils](./tutorials_spatiotemporal_gp/utils.py)

---
### Stationary + Spatial Dependencies

This next tutorial showcases how we can learn a stationary model with spatial dependencies. 
We use yearly block maximum for precipitation based on a dataset from AEMET.

-  Part I: Laplacian Inference [Link](./tutorials_spatiotemporal_gp/2.2.1_pr_sf_vgp_laplacian.ipynb)
-  Part II: Station Predictions and Attribution [Link](./tutorials_spatiotemporal_gp/2.2.2_pr_sf_vgp_laplacian_station_post.ipynb)

---
### NonStationary + Spatial Dependencies

This tutorial showcases how we can learn a non-stationary model with spatial dependencies.
We use yearly block maximum for 2m Max Temperature based on a dataset from AEMET.

-  Part I: Laplacian Inference [Link](./tutorials_spatiotemporal_gp/3.1.1_t2m_st_vgp_laplacian.ipynb)
-  Part II: Station Analysis [Link](./tutorials_spatiotemporal_gp/3.1.2_t2m_st_vgp_laplacian_station_post.ipynb)
-  Part III: Station Predictions and Attribution [Link](./tutorials_spatiotemporal_gp/3.1.3_t2m_st_vgp_laplacian_station_postpred.ipynb)
-  Part IV: Region Analysis and Attribution [Link](./tutorials_spatiotemporal_gp/3.1.4_t2m_st_vgp_laplacian_region.ipynb)