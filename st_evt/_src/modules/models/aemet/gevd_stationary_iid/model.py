import autoroot
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" # first gpu
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'FALSE'

import jax
jax.config.update('jax_platform_name', 'cpu')

import numpyro
from omegaconf import OmegaConf
import multiprocessing
from pathlib import Path
import xarray as xr
import numpy as np
from loguru import logger
from st_evt._src.models.gevd import StationaryUnPooledGEVD
from st_evt.viz import (
    plot_scatter_ts,
    plot_histogram,
    plot_density,
    plot_qq_plot_gevd_manual,
    plot_return_level_gevd_manual_unc_multiple,
    plot_periods,
    plot_periods_diff,
    plot_spain
)
from st_evt.viz import plot_return_level_gevd_manual_unc, plot_return_level_hist_manual_unc
from st_evt.extremes import estimate_return_level_gevd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

import arviz as az
import typer
num_devices = multiprocessing.cpu_count()
numpyro.set_platform("cpu")
# numpyro.set_host_device_count(4)
# num_chains = 5
numpyro.set_host_device_count(num_devices)

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import seaborn as sns
sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)


app = typer.Typer()

import numpyro.distributions as dist
import jax.random as jrandom
from jaxtyping import Array
from st_evt._src.models.inference import SVILearner, MCMCLearner


UNITS = {
    "t2max": "[°C]",
    "pr": "[mm/day]",
    "gmst": "[°C]",
}
VARIABLE_LABEL = {
    "t2max": "2m Max Temperature [°C]",
    "pr": "Cumulative Daily Precipitation [mm/day]",
    "gmst": "Global Mean Surface Temperature [°C]"
}

def init_t2m_model(
    y_values: Array,
    spatial_dim_name: str = "space",
    time_dim_name: str = "time",
    variable_name: str = "obs"
) -> StationaryUnPooledGEVD:
    
    # LOCATION PARAMETER
    loc_init = np.mean(y_values)
    scale_init = np.std(y_values)
    logger.debug(f"Initial Location: Normal({loc_init:.2f}, {scale_init:.2f})")
    location_prior = dist.Normal(float(loc_init), float(scale_init))
    
    # Scale Parameter is always positive
    loc_init = np.log(scale_init)
    logger.debug(f"Initial Scale: LogNormal({loc_init:.2f}, 0.5)")
    scale_prior = dist.LogNormal(loc_init, 0.5)
    
    # TEMPERATURE has a negative shape
    concentration_prior = dist.TruncatedNormal(-0.3, 0.1, low=-1.0, high=-1e-5)

    # initialize model
    return StationaryUnPooledGEVD(
        location_prior=location_prior,
        scale_prior=scale_prior,
        concentration_prior=concentration_prior,
        spatial_dim_name=spatial_dim_name,
        time_dim_name=time_dim_name,
        variable_name=variable_name
        
    )


def init_pr_model(
    y_values: Array,
    spatial_dim_name: str = "space",
    time_dim_name: str = "time",
    variable_name: str = "obs"
) -> StationaryUnPooledGEVD:
    
    # LOCATION PARAMETER
    loc_log_init = np.log(np.mean(y_values))
    scale_log_init = np.log(np.std(y_values))
    logger.debug(f"Initial Location: Normal({loc_log_init:.2f}, {scale_log_init:.2f})")
    location_prior = dist.LogNormal(float(loc_log_init), float(scale_log_init))
    
    # Scale Parameter is always positive
    loc_init = scale_log_init
    logger.debug(f"Initial Scale: LogNormal({loc_init:.2f}, 0.5)")
    scale_prior = dist.LogNormal(loc_init, 0.5)
    
    # TEMPERATURE has a negative shape
    concentration_prior = dist.TruncatedNormal(0.3, 0.1, low=1e-5, high=1.0)

    # initialize model
    return StationaryUnPooledGEVD(
        location_prior=location_prior,
        scale_prior=scale_prior,
        concentration_prior=concentration_prior,
        spatial_dim_name=spatial_dim_name,
        time_dim_name=time_dim_name,
        variable_name=variable_name
        
    )



@app.command()
def train_model_gevd_mcmc(
    dataset_path: str = "",
    variable: str = "t2max",
    covariate: str = "gmst",
    save_path: str = "data/results",
    num_map_warmup: int = 1_000,
    num_mcmc_samples: int = 1_000,
    num_chains: int = 4,
    num_mcmc_warmup: int = 10_000,
    seed: int = 123,
):
    logger.debug(f"VARIABLE: {variable}")
    logger.debug(f"COVARIATE: {covariate}")
    logger.debug(f"Load Path: {dataset_path}")
    logger.debug(f"Save Path: {save_path}")
    SPATIAL_DIM_NAME = "station_id"
    rng_key = jrandom.PRNGKey(seed)
    DATA_URL = Path(dataset_path)
    
    # LOAD DATA
    with xr.open_dataset(DATA_URL, engine="zarr") as f:
        ds_bm = f.load()
        ds_bm = ds_bm.where(ds_bm.red_feten_mask == 1, drop=True)


    y = ds_bm[variable].values.squeeze()

    logger.info(f"Initializing Model")
    if variable == "t2max":
        model = init_t2m_model(
            y_values=ds_bm[variable].values,
            spatial_dim_name=SPATIAL_DIM_NAME,
            time_dim_name=covariate,
            variable_name=variable,
            
        )
    elif variable == "pr":
        model = init_pr_model(
            y_values=ds_bm[variable].values,
            spatial_dim_name=SPATIAL_DIM_NAME,
            time_dim_name=covariate,
            variable_name=variable,
            
        )
    else:
        raise NotImplementedError()
    
    logger.info("Running MAP Initialization...")
    num_steps = num_map_warmup
    num_warmup_steps = int(0.1 * num_steps)

    init_lr = 1e-7
    peak_lr = 1e-3
    end_lr = 1e-4
    method = "map"
    svi_learner = SVILearner(model, peak_lr=peak_lr, end_lr=end_lr, init_lr=init_lr, num_steps=num_steps, num_warmup_steps=num_warmup_steps, method=method)

    svi_posterior = svi_learner(y=y)
    
    logger.info("Grabbing initial parameters...")
    init_params = svi_posterior.median_params
    
    
    logger.info("Running MCMC...")
    
    mcmc_learner = MCMCLearner(
        model=model, 
        num_warmup=num_mcmc_warmup,
        num_samples=num_mcmc_samples,
        num_chains=num_chains,
        init_params=init_params,
    )
    
    
    mcmc_posterior = mcmc_learner(y=y)
    
    # Grabbing Posterior Samples
    
    
    logger.info("Constructing Posterior")
    az_ds = mcmc_posterior.init_arviz_summary()

    # correct coordinates
    az_ds = az_ds.assign_coords({covariate: ds_bm[covariate]})
    az_ds = az_ds.assign_coords({SPATIAL_DIM_NAME: ds_bm[SPATIAL_DIM_NAME]})
    
    
    # Posterior predictive samples
    logger.info("Calculating Predictive Posterior...")
    rng_key, rng_subkey = jrandom.split(rng_key)


    posterior_predictive_samples = mcmc_posterior.posterior_predictive_samples(rng_subkey, num_timesteps=y.shape[0])
        
    
    logger.info("Creating Dataset...")
    az_ds_postpred = az.from_numpyro(
        posterior_predictive=posterior_predictive_samples,
        dims=model.dimensions,
        num_chains=num_chains,
    )
    # correct coordinates
    az_ds_postpred = az_ds_postpred.assign_coords({covariate: ds_bm[covariate]})
    az_ds_postpred = az_ds_postpred.assign_coords({SPATIAL_DIM_NAME: ds_bm[SPATIAL_DIM_NAME]})
    
    az_ds.add_groups(az_ds_postpred)
    
    logger.info("Calculating Log-Likelihood Stats...")
    stats = az.waic(az_ds)
    az_ds.log_likelihood.attrs["elpd_waic"] = stats.elpd_waic
    az_ds.log_likelihood.attrs["se"] = stats.se
    az_ds.log_likelihood.attrs["p_waic"] = stats.p_waic
    
    logger.info("Saving Data Structure...")
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    az_ds.to_netcdf(filename=save_path.joinpath("az_nonstationary.zarr"))
    
    logger.info(f"Grabbing Posterior Samples...")
    posterior_samples = mcmc_posterior.posterior_samples
    logger.info(f"Converting Posterior Samples to serializable format...")
    posterior_samples = {k: np.asarray(v).tolist() for k, v in posterior_samples.items()}
    
    logger.info("Saving Posterior Samples...")
    OmegaConf.save(posterior_samples, save_path.joinpath("az_posterior_params.yaml"))
    
    
@app.command()
def train_model_gevd_lap(
    dataset_path: str = "",
    variable: str = "t2max",
    covariate: str = "gmst",
    save_path: str = "data/results",
    num_map_warmup: int = 1_000,
    num_mcmc_samples: int = 1_000,
    num_chains: int = 4,
    num_mcmc_warmup: int = 10_000,
    seed: int = 123,
):
    raise NotImplementedError()
  
if __name__ == '__main__':
    app()