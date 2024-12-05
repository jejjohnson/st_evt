from typing import Dict
import autoroot
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" # first gpu
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'FALSE'

import jax
jax.config.update('jax_platform_name', 'cpu')
from dvclive import Live
import numpyro
import multiprocessing

num_devices = multiprocessing.cpu_count()
numpyro.set_platform("cpu")
# numpyro.set_host_device_count(4)
# num_chains = 5
numpyro.set_host_device_count(num_devices)

import numpyro
import numpy as np
from pathlib import Path
import xarray as xr
from jaxtyping import Array
import numpyro.distributions as dist
from tensorflow_probability.substrates.jax import distributions as tfd
from loguru import logger
from numpyro.infer import Predictive
import jax.random as jrandom
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, init_to_value, init_to_median
import typer
import arviz as az

app = typer.Typer()

@app.command()
def train_model_station(
    load_path: str="data/ml_ready/",
    num_mcmc_samples: int=1_000,
    num_chains: int=4,
    num_warmup: int=10_000,
    seed: int=123,
    station_id: str='3129A',
    gmst_min: float=0.0,
    gmst_max: float=2.5,
):
    rng_key = jrandom.PRNGKey(seed)
    DATA_URL = Path(load_path).joinpath("t2max_stations_bm_year.zarr")
    
    
    # LOAD DATA
    with xr.open_zarr(DATA_URL) as f:
        ds_bm = f.load()
        ds_bm = ds_bm.sel(station_id=station_id, drop=True)

    variable_bm = "t2max_bm_year"
    
    # initialize empirical values
    logger.info("Getting Initial Values")
    empirical_loc = ds_bm[variable_bm].mean().values.squeeze()
    empirical_std = ds_bm[variable_bm].std().values.squeeze()

    logger.debug(f"Loc: {empirical_loc.squeeze():.2f}")
    logger.debug(f"Scale: {empirical_std.squeeze():.2f}")

    empirical_loc_log = np.log(empirical_loc)
    empirical_std_log = np.log(empirical_std)

    logger.debug(f"Loc (log): {empirical_loc_log:.2f}")
    logger.debug(f"Scale (log): {empirical_std_log:.2f}")
    
    logger.info(f"Initializing Coordinates")
    y = ds_bm[variable_bm].values.squeeze()
    t = ds_bm.gmst.values.squeeze()
    t_pred = np.linspace(gmst_min, gmst_max, 100)
    logger.debug(f"t: {t.shape} | y: {y.shape}")
        



    def model(t: Array, y=None, *args, **kwargs):

        num_timesteps = t.shape[0]

        plate_time = numpyro.plate("time", num_timesteps, dim=-1)


        # LOCATION PARAMETERS
        weight_prior = dist.Normal(0.0, 1.0)
        location_weight = numpyro.sample("location_weight", fn=weight_prior)

        bias_prior = dist.Normal(empirical_loc, empirical_std)
        location_bias = numpyro.sample("location_bias", fn=bias_prior)
        location = numpyro.deterministic("location", location_weight * t + location_bias)

        # SCALE PARAMETER
        scale_prior = dist.LogNormal(1.0, 0.5)
        scale = numpyro.sample("scale", fn=scale_prior)

        # SHAPE Parameter
        concentration_prior = dist.TruncatedNormal(-0.3, 0.1, low=-1.0, high=-1e-5)
        concentration = numpyro.sample("concentration", fn=concentration_prior)

        # create likelihood distribution
        with plate_time:
            y_dist = tfd.GeneralizedExtremeValue(loc=location, scale=scale, concentration=concentration)
            y = numpyro.sample("obs", y_dist, obs=y)

        return y
    
    logger.info("Prior Predictions...")
    

    sampler = NUTS(
        model,
        init_strategy=init_to_median(num_samples=10)
    )

    mcmc = MCMC(
        sampler=sampler,
        num_warmup=num_warmup,
        num_samples=num_mcmc_samples,
        num_chains=num_chains,
        
    )

    # create key
    rng_key_train, rng_key = jrandom.split(rng_key, num=2)

    # RUN MCMC SAMPLER
    mcmc.run(rng_key_train, t=t, y=y)
    
    mcmc.print_summary(exclude_deterministic=True)
    
    logger.info("Posterior Samples...")
    posterior_samples = mcmc.get_samples()
    az_ds = az.from_numpyro(
        posterior=mcmc,
        dims={
            "location": ["gmst"],
            "scale": [],
            "concentration": [],
            "obs": ["gmst"]
        },
        pred_dims={
            "obs": ["gmst"],
            "location": ["gmst"],
        },
        # coords={"time": t_pred}
    )
    # correct coordinates
    az_ds = az_ds.assign_coords({"gmst": ds_bm.gmst})
    
    logger.info(f"Posterior Predictive")
    # Get Posterior Samples
    predictive = Predictive(
        model=model, posterior_samples=posterior_samples, parallel=True, 
        return_sites=[
            "location", "scale", "concentration", "obs"
        ]
    )
    # Posterior predictive samples
    rng_key, rng_subkey = jrandom.split(rng_key)

    posterior_predictive_samples = predictive(rng_subkey, t=t)
    
    az_ds_postpred = az.from_numpyro(
        # posterior=mcmc,
        posterior_predictive=posterior_predictive_samples,
        dims={
            "location": ["gmst"],
            "scale": [],
            "concentration": [],
            "obs": ["gmst"]
        },
        pred_dims={
            "obs": ["gmst"],
            "location": ["gmst"],
        },
        # coords={"time": t_pred}
    )
    # correct coordinates
    az_ds_postpred = az_ds_postpred.assign_coords({"gmst": t})
    
    # Get Posterior Samples
    predictive = Predictive(
        model=model, posterior_samples=posterior_samples, parallel=True, 
        return_sites=[
            "location", "scale", "concentration", "obs"
        ]
    )
    # Posterior predictive samples
    rng_key, rng_subkey = jrandom.split(rng_key)

    prediction_samples = predictive(rng_subkey, t=t_pred)
    
    az_ds_preds = az.from_numpyro(
        # posterior=mcmc,
        predictions=prediction_samples,
        pred_dims={
            "obs": ["gmst"],
            "location": ["gmst"],
        },
        # coords={"time": t_pred}
    )
    # correct coordinates
    az_ds_preds = az_ds_preds.assign_coords({"gmst": t_pred})
    
    az_ds.add_groups(az_ds_postpred)
    az_ds.add_groups(az_ds_preds)
    
    stats = az.waic(az_ds)
    
    with Live() as live:
        live.log_metric("elpd_waic", stats.elpd_waic)
        live.log_metric("elpd_waic_se", stats.se)
        live.log_metric("p_waic", stats.p_waic)


    pass


@app.command()
def evaluate_model():
    raise NotImplementedError()


@app.command()
def predict_model():
    raise NotImplementedError()


if __name__ == '__main__':
    app()