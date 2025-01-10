import os

import autoroot

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # first gpu
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "FALSE"

import jax

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
import numpyro

numpyro.set_host_device_count(16)
numpyro.set_platform("cpu")

import multiprocessing
from pathlib import Path

import arviz as az
import jax.numpy as jnp
import numpy as np
import numpyro
import typer
import xarray as xr
from loguru import logger
from numpyro.infer import log_likelihood
from omegaconf import OmegaConf
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from st_evt._src.models.gp import SpatialGP, SpatioTemporalGEVD, SpatioTemporalModel
from st_evt._src.models.scalar import ScalarModel
from st_evt._src.modules.models.aemet.gevd_nonstationary_gp.model_zoo import (
    init_feature_transformer,
    init_t2m_nonstationary_gp_model,
)
from st_evt.extremes import estimate_return_level_gevd
from st_evt.viz import (
    plot_density,
    plot_histogram,
    plot_periods,
    plot_periods_diff,
    plot_qq_plot_gevd_manual,
    plot_return_level_gevd_manual_unc,
    plot_return_level_gevd_manual_unc_multiple,
    plot_return_level_hist_manual_unc,
    plot_scatter_ts,
    plot_spain,
)

num_devices = multiprocessing.cpu_count()
numpyro.set_platform("cpu")
# numpyro.set_host_device_count(4)
# num_chains = 5
numpyro.set_host_device_count(num_devices)

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter, ScalarFormatter

sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)


app = typer.Typer()

import jax.random as jrandom
import numpyro.distributions as dist
import pyproj
from jaxtyping import Array

from st_evt._src.models.inference import MCMCLearner, SVILearner


@app.command()
def train_model_laplace(
    dataset_path: str = "",
    variable: str = "t2max",
    covariate: str = "gmst",
    save_path: str = "data/results",
    num_iterations: int = 100_000,
    num_posterior_samples: int = 1_000,
    include_train_noise: bool = False,
    include_pred_noise: bool = False,
    seed: int = 123,
    red_feten: bool = False,
):
    """
    Train a non-stationary Gaussian Process model using Laplace approximation.

    Parameters:
    -----------
    dataset_path : str, optional
        Path to the dataset in Zarr format. Default is an empty string.
    variable : str, optional
        The target variable to model. Default is "t2max".
    covariate : str, optional
        The covariate to use in the model. Default is "gmst".
    save_path : str, optional
        Path to save the results. Default is "data/results".
    num_iterations : int, optional
        Number of iterations for the training. Default is 100,000.
    num_posterior_samples : int, optional
        Number of posterior samples to generate. Default is 1,000.
    include_train_noise : bool, optional
        Whether to include noise in the training data. Default is False.
    include_pred_noise : bool, optional
        Whether to include noise in the prediction data. Default is False.
    seed : int, optional
        Random seed for reproducibility. Default is 123.
    red_feten : bool, optional
        Whether to use the red_feten mask for training. Default is False.

    Returns:
    --------
    None
    """
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

    y = ds_bm[variable].values.squeeze()
    t = ds_bm[covariate].values.squeeze()
    assert len(y.shape) == 2
    assert len(t.shape) == 1

    # STANDARDIZATION
    logger.info(f"Featurizing Coordinates...")
    ds_bm, spatial_transformer = init_feature_transformer(ds_bm)

    logger.info("Saving Transformer...")
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"spatial": spatial_transformer},
        save_path.joinpath("feature_transforms.joblib"),
    )

    # DATASET TRAIN-VALID-TEST SPLIT
    logger.info("Train Split...")
    if red_feten:
        ds_bm_train = ds_bm.where(ds_bm.red_feten_mask == 1, drop=True)
    else:
        ds_bm_train = ds_bm

    y_train = ds_bm_train[variable].values.squeeze()
    # t = ds_bm.time.dt.year.values.squeeze()
    t_train = ds_bm_train[covariate].values.squeeze()
    s_train = ds_bm_train.coords_norm.values

    t0 = t_train.min()
    logger.debug(f"T-Train Shape: {t_train.shape}")
    logger.debug(f"S-Train Shape: {s_train.shape}")
    logger.debug(f"Y-Train Shape: {y_train.shape}")

    logger.info("Test Split...")
    ds_bm_test = ds_bm.where(ds_bm.red_feten_mask == 0, drop=True)

    s_test = ds_bm_test.coords_norm.values
    t_test = ds_bm_train[covariate].values.squeeze()
    y_test = ds_bm_test[variable].values.squeeze()
    logger.debug(f"T-Test Shape: {t_test.shape}")
    logger.debug(f"S-Test Shape: {s_test.shape}")
    logger.debug(f"Y-Test Shape: {y_test.shape}")

    logger.info("Prediction Split...")
    s_pred = ds_bm.coords_norm.values
    t_pred = np.linspace(0.0, 2.5, 100)
    y_pred = ds_bm[variable].values.squeeze()
    logger.debug(f"T-Prediction Shape: {t_pred.shape}")
    logger.debug(f"S-Prediction Shape: {s_pred.shape}")
    logger.debug(f"Y-Prediction Shape: {y_pred.shape}")

    logger.info("Initializing GP Model...")
    model = init_t2m_nonstationary_gp_model(
        spatial_coords=s_train,
        y_values=y_train,
        t0=t0,
        spatial_dim_name=SPATIAL_DIM_NAME,
        time_dim_name=covariate,
        variable_name=variable,
    )

    logger.info("Initializing SVI Training/Inference Procedure...")
    num_warmup_steps = int(0.1 * num_iterations)

    init_lr = 1e-7
    peak_lr = 1e-3
    end_lr = 1e-4
    method = "laplace"
    svi_learner = SVILearner(
        model,
        peak_lr=peak_lr,
        end_lr=end_lr,
        init_lr=init_lr,
        num_steps=num_iterations,
        num_warmup_steps=num_warmup_steps,
        method=method,
    )

    logger.info("Training Model...")
    svi_posterior = svi_learner(
        t=t_train, s=s_train, y=y_train, train=True, noise=include_train_noise
    )

    logger.info("Generating Posterior Samples...")

    # POSTERIOR
    rng_key, rng_subkey = jrandom.split(rng_key)
    posterior_samples = svi_posterior.variational_samples(
        rng_subkey, num_samples=num_posterior_samples
    )

    # LOG LIKELIHOOD (Posterior)
    logger.info("Calculating Log-Likelihood...")
    nll_post_samples = svi_posterior.log_likelihood_samples(
        rng_subkey,
        num_samples=num_posterior_samples,
        # MY MODEL ARGUMENTS
        s=s_train,
        t=t_train,
        y=y_train,
        train=True,
        noise=include_train_noise,
    )

    logger.info("Constructing Posterior Datastructure...")
    az_ds = az.from_dict(
        posterior={k: np.expand_dims(v, 0) for k, v in posterior_samples.items()},
        log_likelihood={k: np.expand_dims(v, 0) for k, v in nll_post_samples.items()},
        observed_data={variable: y_train},
        dims=model.dimensions,
    )

    # correct coordinates
    logger.info("Correcting Coordinates...")
    az_ds = az_ds.assign_coords({covariate: ds_bm_train[covariate]})
    az_ds = az_ds.assign_coords({SPATIAL_DIM_NAME: ds_bm_train[SPATIAL_DIM_NAME]})
    az_ds = az_ds.rename({"spatial_dims": "spherical"})
    az_ds = az_ds.assign_coords({"spherical": ds_bm.spherical})
    nll = az_ds.log_likelihood[variable]
    az_ds.log_likelihood[variable] = nll.where(np.isfinite(nll), np.finfo(float).eps)

    # add extra data
    logger.info("Adding Extra Information...")
    az_ds.observed_data["s_coords"] = ds_bm_train["coords_norm"]
    az_ds.observed_data["t0"] = float(t0)

    # PREDICTIVE POSTERIOR
    logger.info("Calculating Posterior-Predictive...")
    return_sites = [
        "location",
        "location_slope",
        "location_intercept",
        "scale",
        "concentration",
        variable,
    ]
    posterior_predictive_samples = svi_posterior.posterior_predictive_samples(
        rng_subkey,
        num_samples=num_posterior_samples,
        return_sites=return_sites,
        s=s_pred,
        t=t_train,
        train=False,
        noise=include_pred_noise,
    )
    # PREDICTIVE POSTERIOR
    posterior_predictive_samples_all = svi_posterior.posterior_predictive_samples(
        rng_subkey,
        num_samples=num_posterior_samples,
        return_sites=model.variables,
        s=s_pred,
        t=t_train,
        train=False,
        noise=include_pred_noise,
    )

    logger.info("Calculating Log-Likelihood for Posterior Predictive...")
    nll_postpred_samples = log_likelihood(
        model=model,
        posterior_samples=posterior_predictive_samples_all,
        parallel=False,
        batch_ndim=1,
        # My Function Arguments
        s=s_pred,
        t=t_train,
        y=y_pred,
        train=False,
        noise=include_pred_noise,
    )

    logger.info("Constructing Posterior Predictive Datastructure...")
    az_ds_postpred = az.from_dict(
        posterior_predictive={
            k: np.expand_dims(v, 0) for k, v in posterior_predictive_samples.items()
        },
        dims=model.dimensions,
    )

    # correct coordinates
    logger.info("Correcting Coordinates...")
    az_ds_postpred = az_ds_postpred.assign_coords({covariate: ds_bm_train[covariate]})
    az_ds_postpred = az_ds_postpred.assign_coords(
        {SPATIAL_DIM_NAME: ds_bm[SPATIAL_DIM_NAME]}
    )
    az_ds_postpred = az_ds_postpred.rename({"spatial_dims": "spherical"})
    az_ds_postpred = az_ds_postpred.assign_coords({"spherical": ds_bm.spherical})

    # ADDING EXTRA DATA
    logger.info("Adding Extra Coordinates...")
    az_ds_postpred.posterior_predictive["nll"] = (
        ("chain", "draw", covariate, SPATIAL_DIM_NAME),
        np.expand_dims(nll_postpred_samples[variable], 0),
    )
    nll = az_ds_postpred.posterior_predictive["nll"]
    az_ds_postpred.posterior_predictive["nll"] = nll.where(np.isfinite(nll), np.finfo(float).eps)
    az_ds_postpred.posterior_predictive[f"{variable}_true"] = (
        (covariate, SPATIAL_DIM_NAME),
        np.asarray(y_pred),
    )

    # PREDICTIONS
    logger.info("Calculating Predictions (Forecasting)...")
    return_sites = [
        "location",
        "location_slope",
        "location_intercept",
        "scale",
        "concentration",
        variable,
    ]
    prediction_samples = svi_posterior.posterior_predictive_samples(
        rng_subkey,
        num_samples=num_posterior_samples,
        return_sites=return_sites,
        s=s_pred,
        t=t_pred,
        train=False,
        noise=include_pred_noise,
    )

    logger.info("Constructing Predictions Datastructure...")
    az_ds_preds = az.from_dict(
        predictions={k: np.expand_dims(v, 0) for k, v in prediction_samples.items()},
        pred_dims=model.dimensions,
    )

    # correct coordinates
    logger.info("Correcting Coordinates...")
    az_ds_preds = az_ds_preds.assign_coords({covariate: t_pred})
    az_ds_preds = az_ds_preds.assign_coords({SPATIAL_DIM_NAME: ds_bm[SPATIAL_DIM_NAME]})
    az_ds_preds = az_ds_preds.rename({"spatial_dims": "spherical"})
    az_ds_preds = az_ds_preds.assign_coords({"spherical": ds_bm.spherical})

    logger.info("Combining all datastructures...")
    az_ds.add_groups(az_ds_postpred)
    az_ds.add_groups(az_ds_preds)

    logger.info("Calculating Quick Stats...")
    stats = az.waic(az_ds)

    logger.debug(f"ELPD WAIC: {stats.elpd_waic:.2f}")
    logger.debug(f"ELPD WAIC SE: {stats.se:.2f}")
    logger.debug(f"P WAIC: {stats.p_waic:.2f}")

    logger.info("Saving stats to NLL Group")
    az_ds.log_likelihood.attrs["elpd_waic"] = stats.elpd_waic
    az_ds.log_likelihood.attrs["se"] = stats.se
    az_ds.log_likelihood.attrs["p_waic"] = stats.p_waic

    logger.info("Saving Data Structure...")
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    save_name = (
        "az_nonstationary_gp_lap_redfeten.zarr"
        if red_feten
        else "az_nonstationary_gp_lap.zarr"
    )
    logger.debug(f"Filename: {save_name}")
    az_ds.to_zarr(store=str(save_path.joinpath(save_name)))
    logger.debug(f"Full Save Path: {save_path.joinpath(save_name)}")

    logger.info("Finished Script!")


@app.command()
def train_model_mcmc(
    dataset_path: str = "",
    variable: str = "t2max",
    covariate: str = "gmst",
    save_path: str = "data/results",
    include_train_noise: bool = False,
    include_pred_noise: bool = False,
    num_map_warmup: int = 100_000,
    num_mcmc_samples: int = 1_000,
    num_chains: int = 4,
    num_mcmc_warmup: int = 10_000,
    seed: int = 123,
    red_feten: bool = False,
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

    y = ds_bm[variable].values.squeeze()
    t = ds_bm[covariate].values.squeeze()
    assert len(y.shape) == 2
    assert len(t.shape) == 1

    # STANDARDIZATION
    logger.info(f"Featurizing Coordinates...")
    ds_bm, spatial_transformer = init_feature_transformer(ds_bm)

    logger.info("Saving Transformer...")
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"spatial": spatial_transformer},
        save_path.joinpath("feature_transforms.joblib"),
    )

    # DATASET TRAIN-VALID-TEST SPLIT
    logger.info("Train Split...")
    if red_feten:
        ds_bm_train = ds_bm.where(ds_bm.red_feten_mask == 1, drop=True)
    else:
        ds_bm_train = ds_bm

    y_train = ds_bm_train[variable].values.squeeze()
    # t = ds_bm.time.dt.year.values.squeeze()
    t_train = ds_bm_train[covariate].values.squeeze()
    s_train = ds_bm_train.coords_norm.values

    t0 = t_train.min()
    logger.debug(f"T-Train Shape: {t_train.shape}")
    logger.debug(f"S-Train Shape: {s_train.shape}")
    logger.debug(f"Y-Train Shape: {y_train.shape}")

    logger.info("Test Split...")
    ds_bm_test = ds_bm.where(ds_bm.red_feten_mask == 0, drop=True)

    s_test = ds_bm_test.coords_norm.values
    t_test = ds_bm_train[covariate].values.squeeze()
    y_test = ds_bm_test[variable].values.squeeze()
    logger.debug(f"T-Test Shape: {t_test.shape}")
    logger.debug(f"S-Test Shape: {s_test.shape}")
    logger.debug(f"Y-Test Shape: {y_test.shape}")

    logger.info("Prediction Split...")
    s_pred = ds_bm.coords_norm.values
    t_pred = np.linspace(0.0, 2.5, 100)
    y_pred = ds_bm[variable].values.squeeze()
    logger.debug(f"T-Prediction Shape: {t_pred.shape}")
    logger.debug(f"S-Prediction Shape: {s_pred.shape}")
    logger.debug(f"Y-Prediction Shape: {y_pred.shape}")
    
    logger.info("Initializing GP Model...")
    model = init_t2m_nonstationary_gp_model(
        spatial_coords=s_train,
        y_values=y_train,
        t0=t0,
        spatial_dim_name=SPATIAL_DIM_NAME,
        time_dim_name=covariate,
        variable_name=variable,
    )

    logger.info("Initializing SVI Training/Inference Procedure...")
    num_warmup_steps = int(0.1 * num_map_warmup)

    init_lr = 1e-7
    peak_lr = 1e-3
    end_lr = 1e-4
    method = "map"
    svi_learner = SVILearner(
        model,
        peak_lr=peak_lr,
        end_lr=end_lr,
        init_lr=init_lr,
        num_steps=num_map_warmup,
        num_warmup_steps=num_warmup_steps,
        method=method,
    )

    logger.info("Training Model SVI Warmup...")
    svi_posterior = svi_learner(
        t=t_train, s=s_train, y=y_train, train=True, noise=include_train_noise
    )

    logger.info("Initializing MCMC Inferencer...")

    mcmc_learner = MCMCLearner(
        model=model,
        num_warmup=num_mcmc_warmup,
        num_samples=num_mcmc_samples,
        num_chains=num_chains,
        init_params=svi_posterior.median_params,
    )

    logger.info("Running MCMC...")

    mcmc_posterior = mcmc_learner(
        t=t_train, s=s_train, y=y_train, train=True, noise=include_train_noise
    )

    logger.info("Generating Posterior Samples...")

    # POSTERIOR
    logger.info("Constructing Posterior Datastructure")
    az_ds = mcmc_posterior.init_arviz_summary()

    # correct coordinates
    logger.info("Correcting Coordinates...")
    az_ds = az_ds.assign_coords({covariate: ds_bm_train[covariate]})
    az_ds = az_ds.assign_coords({SPATIAL_DIM_NAME: ds_bm_train[SPATIAL_DIM_NAME]})
    az_ds = az_ds.rename({"spatial_dims": "spherical"})
    az_ds = az_ds.assign_coords({"spherical": ds_bm.spherical})

    # add extra data
    logger.info("Adding Extra Information...")
    az_ds.observed_data["s_coords"] = ds_bm_train["coords_norm"]
    az_ds.observed_data["t0"] = float(t0)

    # PREDICTIVE POSTERIOR
    return_sites = [
        "location",
        "location_slope",
        "location_intercept",
        "scale",
        "concentration",
        variable,
    ]

    rng_key, rng_subkey = jrandom.split(rng_key)

    posterior_predictive_samples = mcmc_posterior.posterior_predictive_samples(
        rng_subkey,
        return_sites=return_sites,
        s=s_pred,
        t=t_train,
        train=False,
        noise=include_pred_noise,
    )

    # PREDICTIVE POSTERIOR
    posterior_predictive_samples_all = mcmc_posterior.posterior_predictive_samples(
        rng_subkey,
        return_sites=None,
        s=s_pred,
        t=t_train,
        train=False,
        noise=include_pred_noise,
    )

    logger.info("Calculating Log-Likelihood for Posterior Predictive...")
    nll_postpred_samples = log_likelihood(
        model=model,
        posterior_samples=posterior_predictive_samples_all,
        parallel=False,
        batch_ndim=1,
        # My Function Arguments
        s=s_pred,
        t=t_train,
        y=y_pred,
        train=False,
        noise=include_pred_noise,
    )

    logger.info("Constructing Posterior Predictive Datastructure...")
    az_ds_postpred = az.from_dict(
        posterior_predictive={
            k: np.expand_dims(v, 0) for k, v in posterior_predictive_samples.items()
        },
        dims=model.dimensions,
    )

    # correct coordinates
    logger.info("Correcting Coordinates...")
    az_ds_postpred = az_ds_postpred.assign_coords({covariate: ds_bm_train[covariate]})
    az_ds_postpred = az_ds_postpred.assign_coords(
        {SPATIAL_DIM_NAME: ds_bm[SPATIAL_DIM_NAME]}
    )
    az_ds_postpred = az_ds_postpred.rename({"spatial_dims": "spherical"})
    az_ds_postpred = az_ds_postpred.assign_coords({"spherical": ds_bm.spherical})

    # ADDING EXTRA DATA
    logger.info("Adding Extra Coordinates...")
    az_ds_postpred.posterior_predictive["nll"] = (
        ("chain", "draw", covariate, SPATIAL_DIM_NAME),
        np.expand_dims(nll_postpred_samples[variable], 0),
    )
    az_ds_postpred.posterior_predictive[f"{variable}_true"] = (
        (covariate, SPATIAL_DIM_NAME),
        np.asarray(y_pred),
    )

    # PREDICTIONS
    logger.info("Calculating Predictions (Forecasting)...")
    return_sites = [
        "location",
        "location_slope",
        "location_intercept",
        "scale",
        "concentration",
        variable,
    ]
    prediction_samples = mcmc_posterior.posterior_predictive_samples(
        rng_subkey,
        return_sites=return_sites,
        s=s_pred,
        t=t_pred,
        train=False,
        noise=include_pred_noise,
    )

    logger.info("Constructing Predictions Datastructure...")
    az_ds_preds = az.from_dict(
        predictions={k: np.expand_dims(v, 0) for k, v in prediction_samples.items()},
        pred_dims=model.dimensions,
    )

    # correct coordinates
    logger.info("Correcting Coordinates...")
    az_ds_preds = az_ds_preds.assign_coords({covariate: t_pred})
    az_ds_preds = az_ds_preds.assign_coords({SPATIAL_DIM_NAME: ds_bm[SPATIAL_DIM_NAME]})
    az_ds_preds = az_ds_preds.rename({"spatial_dims": "spherical"})
    az_ds_preds = az_ds_preds.assign_coords({"spherical": ds_bm.spherical})

    logger.info("Combining all datastructures...")
    az_ds.add_groups(az_ds_postpred)
    az_ds.add_groups(az_ds_preds)

    logger.info("Calculating Quick Stats...")
    stats = az.waic(az_ds)

    logger.debug(f"ELPD WAIC: {stats.elpd_waic:.2f}")
    logger.debug(f"ELPD WAIC SE: {stats.se:.2f}")
    logger.debug(f"P WAIC: {stats.p_waic:.2f}")

    logger.info("Saving stats to NLL Group")
    az_ds.log_likelihood.attrs["elpd_waic"] = stats.elpd_waic
    az_ds.log_likelihood.attrs["se"] = stats.se
    az_ds.log_likelihood.attrs["p_waic"] = stats.p_waic

    logger.info("Saving Data Structure...")
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    save_name = (
        "az_nonstationary_gp_mcmc_redfeten.zarr"
        if red_feten
        else "az_nonstationary_gp_mcmc.zarr"
    )
    logger.debug(f"Filename: {save_name}")
    az_ds.to_zarr(store=str(save_path.joinpath(save_name)))
    logger.debug(f"Full Save Path: {save_path.joinpath(save_name)}")

    logger.info("Finished Script!")


if __name__ == "__main__":
    app()
