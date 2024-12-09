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
import matplotlib
sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)

plt.style.use(
    "https://raw.githubusercontent.com/ClimateMatchAcademy/course-content/main/cma.mplstyle"
)


app = typer.Typer()

import numpyro.distributions as dist
import jax.random as jrandom
from jaxtyping import Array
from st_evt._src.models.inference import SVILearner, MCMCLearner

SHAPE_BOUNDS = {
    "t2max": (-1.0, 7, 0.0),
    "pr": (0.0, 5, 0.5)
}

SCALE_BOUNDS = {
    "t2max": (0.0, 7, 3.5),
    "pr": (5.0, 5, 50.0)
}

LOCATION_BOUNDS = {
    "t2max": (20.0, 6, 50.0),
    "pr": (10, 8, 50.0)
}

CMAP = {
    "t2max": "Reds",
    "pr": "Blues"
}


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


def calculate_ds_return_periods(az_ds):

    from st_evt.extremes import estimate_return_level_gevd, calculate_exceedence_probs


    RETURN_PERIODS_GEVD = np.logspace(0.001, 4, 100)
    
    fn_gevd = jax.jit(estimate_return_level_gevd)
    
    def calculate_return_period(return_periods, location, scale, shape):
        rl = jax.vmap(fn_gevd, in_axes=(0,None,None,None))(return_periods, location, scale, shape)
        return rl

    az_ds.posterior["return_level_100"] = xr.apply_ufunc(
        calculate_return_period,
        [100],
        az_ds.posterior.location,
        az_ds.posterior.scale,
        az_ds.posterior.concentration,
        input_core_dims=[[""], ["draw"], ["draw"], ["draw"]],
        output_core_dims=[["draw"]],
        vectorize=True
    )
    
    az_ds.posterior["return_level"] = xr.apply_ufunc(
        calculate_return_period,
        RETURN_PERIODS_GEVD,
        az_ds.posterior.location,
        az_ds.posterior.scale,
        az_ds.posterior.concentration,
        input_core_dims=[["return_period"], ["draw"], ["draw"], ["draw"]],
        output_core_dims=[["return_period", "draw"]],
        vectorize=True
    )
    
    
    az_ds.posterior_predictive["return_level_100"] = xr.apply_ufunc(
        calculate_return_period,
        [100],
        az_ds.posterior_predictive.location,
        az_ds.posterior_predictive.scale,
        az_ds.posterior_predictive.concentration,
        input_core_dims=[[""], ["draw"], ["draw"], ["draw"]],
        output_core_dims=[["draw"]],
        vectorize=True
    )
    
    az_ds.posterior_predictive["return_level"] = xr.apply_ufunc(
        calculate_return_period,
        RETURN_PERIODS_GEVD,
        az_ds.posterior_predictive.location,
        az_ds.posterior_predictive.scale,
        az_ds.posterior_predictive.concentration,
        input_core_dims=[["return_period"], ["draw"], ["draw"], ["draw"]],
        output_core_dims=[["return_period", "draw"]],
        vectorize=True
    )
    az_ds = az_ds.assign_coords({"return_period": RETURN_PERIODS_GEVD})

    return az_ds


def plot_model_params_critique(
    az_ds,
    variables: list = [
        "concentration",
        "scale",
        "location",
        "return_level_100", 
        ],
    save_path: str = "",
):
    
    logger.info(f"Plotting Parameter Traces...")
    fig = az.plot_trace(
        az_ds.posterior, 
        var_names=variables,
        figsize=(10, 7)
    );
    plt.gcf().set_dpi(300)
    plt.tight_layout()
    sub_figures_path = save_path.joinpath("params")
    sub_figures_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(sub_figures_path.joinpath("trace.png"))
    plt.close()
    
    logger.info(f"Plotting Parameter Jonts...")
    fig = az.plot_pair(
        az_ds.posterior,
        # group="posterior",
        var_names=variables,
        kind=["scatter", "kde"],
        kde_kwargs={"fill_last": False},
        marginals=True,
        # coords=coords,
        point_estimate="median",
        figsize=(10, 8),
    )
    plt.gcf().set_dpi(300)
    plt.tight_layout()
    sub_figures_path = save_path.joinpath("params")
    sub_figures_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(sub_figures_path.joinpath("joint.png"))
    plt.close()

    
    return None


def plot_model_metrics(
    az_ds,
    variable: str = "t2max",
    covariate: str = "gmst",
    save_path: str = "",
    units: str = "[°C]"
):
    
    # extract parameters
    idata = az.extract(az_ds, group="posterior", num_samples=None)

    # extract observations
    y_obs = az_ds.observed_data[variable].dropna(dim=covariate)

    logger.info(f"Plotting qq plot...")
    fig, ax = plot_qq_plot_gevd_manual(
        location=idata.location, 
        scale=idata.scale, 
        concentration=idata.concentration, 
        observations=y_obs.values.ravel(),
        figures_save_dir=None)
    fig.set(dpi=300)
    sub_figures_path = save_path.joinpath("metrics")
    sub_figures_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(sub_figures_path.joinpath("qq_plot.png"))
    plt.close()
    
    
    logger.info(f"Plotting RMSE...")
    idata = az.extract(az_ds, group="posterior_predictive", num_samples=None)

    y_obs = az_ds.observed_data[variable]
    y_pred = idata[variable]
    
    error = xr.apply_ufunc(
        root_mean_squared_error,
        y_obs,
        y_pred,
        input_core_dims=[[covariate], [covariate]],
        output_core_dims=[[]],
        vectorize=True
    )
        
    fig, ax = plot_density(error)
    ax.set(
        title="",
        xlabel=f"Root Mean Squared Error {units}"
    )
    fig.set(dpi=300)
    plt.tight_layout()
    sub_figures_path = save_path.joinpath("metrics")
    sub_figures_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(sub_figures_path.joinpath("rmse_density.png"))
    plt.close()
    
    error = xr.apply_ufunc(
        mean_absolute_error,
        y_obs,
        y_pred,
        input_core_dims=[[covariate], [covariate]],
        output_core_dims=[[]],
        vectorize=True
    )
    
    fig, ax = plot_density(error)
    ax.set(
        title="",
        xlabel=f"Mean Absolute Error {units}"
    )
    fig.set(dpi=300)
    plt.tight_layout()
    sub_figures_path = save_path.joinpath("metrics")
    sub_figures_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(sub_figures_path.joinpath("mae_density.png"))
    plt.close()
    
    return None


def plot_return_period_metrics(
    az_ds,
    variable: str = "t2max",
    covariate: str = "gmst",
    save_path: str = "",
    variable_label: str = "",
):
    idata = az.extract(az_ds, group="posterior_predictive", num_samples=None)
        
    from functools import partial
    return_periods = np.logspace(0.001, 3, 100)


    return_level = idata["return_level"].T
    return_periods = idata.return_period
    y = az_ds.observed_data[variable].dropna(dim=covariate).squeeze()
    
    
    logger.info(f"Plotting Return Period...")
    fig, ax = plot_return_level_gevd_manual_unc(
        return_level.squeeze(), return_periods, y.squeeze(), None
    )
    ax.set(
        ylabel=variable_label,
    )
    fig.set(dpi=300)
    sub_figures_path = Path(save_path).joinpath("returns")
    sub_figures_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(sub_figures_path.joinpath("rp_vs_obs.png"))
    plt.close()
    
    
    logger.info(f"Plotting 100-Year Return Period Histogram...")
    return_level_100 = idata.return_level_100
    
    hist, bins = np.histogram(return_level_100.values.ravel(), bins=20)
    bins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

    fig, ax = plot_return_level_hist_manual_unc(return_level_100, None, bins=bins)

    ax.set(
        xlabel=variable_label,
        title="100-Year Return Level"
    )
    fig.set(dpi=300)
    sub_figures_path = Path(save_path).joinpath("returns")
    sub_figures_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(sub_figures_path.joinpath("rl100_density.png"))
    plt.close()
        
    return None


@app.command()
def evaluate_model_gevd_mcmc_station(
    dataset_path: str = "",
    variable: str = "t2max",
    covariate: str = "gmst",
    station_id: str = "8414A",
    save_path: str = "",
):
    logger.info(f"Starting script...")
    # NONE
    
    dataset_path = Path(dataset_path)
    save_path = Path(save_path)
    logger.debug(f"Dataset path {dataset_path}")
    logger.debug(f"Save path {save_path}")
    logger.debug(f"Variable {variable}")
    logger.debug(f"Covariate {covariate}")
    logger.debug(f"Station ID {station_id}")
    
    logger.info(f"Load data...")
    logger.info(f"Creating figures directory...")
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Figures Path: {save_path}")
    
    logger.info(f"Loading dataset")
    az_ds = az.from_netcdf(dataset_path)
    
    logger.info(f"Selecting station")
    az_ds = az_ds.sel(station_id = station_id)
    
    logger.info(f"Calculating RP")
    az_ds = calculate_ds_return_periods(az_ds)
    
    plot_model_params_critique(
        az_ds=az_ds,
        save_path=Path(save_path)
    )
    
    plot_model_metrics(
        az_ds=az_ds,
        variable=variable,
        covariate=covariate,
        save_path=save_path,
        units=UNITS[variable]
    )
    
    plot_return_period_metrics(
        az_ds=az_ds,
        variable=variable,
        covariate=covariate,
        save_path=save_path,
        variable_label=VARIABLE_LABEL[variable],
    )

 
@app.command()
def evaluate_model_gevd_mcmc_spain(
    dataset_path: str = "",
    variable: str = "t2max",
    covariate: str = "gmst",
    save_path: str = "",
):
    logger.info(f"Starting script...")
    
    dataset_path = Path(dataset_path)
    save_path = Path(save_path)
    logger.debug(f"Dataset path {dataset_path}")
    logger.debug(f"Save path {save_path}")
    logger.debug(f"Variable {variable}")
    logger.debug(f"Covariate {covariate}")
    
    logger.info(f"Load data...")
    logger.info(f"Creating figures directory...")
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Figures Path: {save_path}")
    
    logger.info(f"Loading dataset")
    az_ds = az.from_netcdf(dataset_path)
    
    # logger.info(f"Calculating RP")
    # az_ds = calculate_ds_return_periods(az_ds)
    
    az_ds = az.from_netcdf(dataset_path)
    
    plot_spain_nll(az_ds=az_ds, variable=variable, covariate=covariate, save_path=save_path)
    
    plot_spain_scale(
        az_ds=az_ds,
        variable=variable,
        covariate=covariate,
        units=UNITS[variable],
        save_path=save_path,
        bounds=SCALE_BOUNDS[variable],
        )
    
    plot_spain_concentration(
        az_ds=az_ds,
        variable=variable,
        covariate=covariate,
        units=UNITS[variable],
        bounds=SHAPE_BOUNDS[variable],
        save_path=save_path,
    )
    plot_spain_location(
        az_ds=az_ds,
        variable=variable,
        covariate=covariate,
        cmap=CMAP[variable],
        units=UNITS[variable],
        bounds=LOCATION_BOUNDS[variable],
        save_path=save_path,
    )

    
def plot_spain_nll(
    az_ds,
    variable: str="t2max",
    covariate: str="gmst",
    save_path: str=""):
    
    logger.info(f"Plotting NLL Density ...")
    
    idata = az.extract(az_ds, group="log_likelihood", num_samples=None).median(dim=["sample"]).load()
    idata = idata.sortby(covariate)
    
    isub = idata[variable].rename("nll").sum(dim=covariate)

    subfigures_path = save_path.joinpath("nll")
    subfigures_path.mkdir(parents=True, exist_ok=True)
    cbar_label = r"ELBO Loss, $\mathcal{L}(\mathcal{D})$"
        
    fig, ax = plot_density(isub)
    ax.set(
        title="",
        xlabel=cbar_label
    )
    fig.set(dpi=300)
    plt.tight_layout()
    fig.savefig(subfigures_path.joinpath("nll_density.png"))
    plt.close()
    
    import matplotlib.colors as colors

    # colormap
    cmap = plt.get_cmap('viridis', 6)
    vmin = isub.min()
    vmax = isub.max()
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    logger.info(f"Plotting NLL Map...")
    fig, ax, cbar = plot_spain(isub, s=75.0, norm=norm, vmin=vmin, vmax=vmax, cmap=cmap, region="mainland")
    cbar.set_label(cbar_label)
    fig.set(dpi=300)
    plt.tight_layout()
    fig.savefig(subfigures_path.joinpath("nll_map.png"))
    plt.close()
    
    logger.info(f"Plotting covariates")
    isub_mainland = isub.where(isub.lon > -10, drop=True )
    
    # PLOT COVARIATES
    plot_covariates(ds=isub_mainland, parameter="nll", cbar_label=cbar_label, save_path=subfigures_path)
    
    return None


def plot_spain_scale(
    az_ds,
    variable: str="t2max",
    covariate: str="gmst",
    units: str = "[°C]",
    bounds: tuple = (20.0, 5, 45.0),
    save_path: str=""):
    
    logger.info(f"Plotting Scale Parameter ...")
    
    parameter = "scale"
    
    idata = az.extract(az_ds, group="posterior", num_samples=None)
    idata = idata.sortby(covariate)
    isub = idata[parameter].quantile(q=0.5, dim=["sample"]).load()

    subfigures_path = save_path.joinpath(f"posterior/{parameter}")
    subfigures_path.mkdir(parents=True, exist_ok=True)
    cbar_label =r"Scale, $\boldsymbol{\sigma}_0(\mathbf{s})$" + f" {units}"
        
    fig, ax = plot_density(isub)
    ax.set(
        title="",
        xlabel=cbar_label
    )
    fig.set(dpi=300)
    plt.tight_layout()
    fig.savefig(subfigures_path.joinpath(f"{parameter}_density.png"))
    plt.close()
    
    import matplotlib.colors as colors

    # colormap
    cmap = plt.get_cmap('viridis', bounds[1])
    vmin = bounds[0]
    vmax = bounds[2]
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    logger.info(f"Plotting Scale Map...")
    fig, ax, cbar = plot_spain(isub, s=75.0, norm=norm, vmin=vmin, vmax=vmax, cmap=cmap, region="mainland")
    cbar.set_label(cbar_label)
    fig.set(dpi=300)
    plt.tight_layout()
    fig.savefig(subfigures_path.joinpath(f"{parameter}_map.png"))
    plt.close()
    
    logger.info(f"Plotting Scale vs. covariates")
    isub_mainland = isub.where(isub.lon > -10, drop=True )
    
    plot_covariates(ds=isub_mainland, parameter="scale", cbar_label=cbar_label, save_path=subfigures_path)
    
    return None


def plot_covariates(ds, parameter: str="scale", cbar_label="", save_path: str=""):
    isub_mainland = ds.where(ds.lon > -10, drop=True )
    
    # LONGITUDE
    fig, ax = plt.subplots()
    isub_mainland.to_dataset().plot.scatter(ax=ax,  x=parameter, y="lon", s=10.0, color="black")
    ax.set(
        xlabel=cbar_label
    )
    ax.grid(True, which="major", linestyle='--', linewidth='0.5', color='gray')
    ax.minorticks_on()
    fig.set(dpi=300)
    plt.tight_layout()
    fig.savefig(save_path.joinpath(f"{parameter}_covariates_lon.png"))
    plt.close()
    
    # LONGITUDE
    fig, ax = plt.subplots()
    isub_mainland.to_dataset().plot.scatter(ax=ax,  x=parameter, y="lat", s=10.0, color="black")
    ax.set(
        xlabel=cbar_label
    )
    ax.grid(True, which="major", linestyle='--', linewidth='0.5', color='gray')
    ax.minorticks_on()
    fig.set(dpi=300)
    plt.tight_layout()
    fig.savefig(save_path.joinpath(f"{parameter}_covariates_lat.png"))
    plt.close()
    
    # LONGITUDE
    fig, ax = plt.subplots()
    isub_mainland.to_dataset().plot.scatter(ax=ax,  x=parameter, y="lon", s=10.0, color="black")
    ax.set(
        xlabel=cbar_label
    )
    ax.grid(True, which="major", linestyle='--', linewidth='0.5', color='gray')
    ax.minorticks_on()
    fig.set(dpi=300)
    plt.tight_layout()
    fig.savefig(save_path.joinpath(f"{parameter}_covariates_alt.png"))
    plt.close()
    return None


def plot_spain_concentration(
    az_ds,
    variable: str="t2max",
    covariate: str="gmst",
    units: str = "[°C]",
    bounds: tuple = (-1.0, 5, 1.0),
    save_path: str=""):
    
    logger.info(f"Plotting Scale Parameter ...")
    
    parameter = "concentration"
    
    idata = az.extract(az_ds, group="posterior", num_samples=None)
    idata = idata.sortby(covariate)
    isub = idata[parameter].quantile(q=0.5, dim=["sample"]).load()

    subfigures_path = save_path.joinpath(f"posterior/{parameter}")
    subfigures_path.mkdir(parents=True, exist_ok=True)
    cbar_label =r"Concentration, $\boldsymbol{\kappa}_1(\mathbf{s})$" + f" {units}"
        
    fig, ax = plot_density(isub)
    ax.set(
        title="",
        xlabel=cbar_label
    )
    fig.set(dpi=300)
    plt.tight_layout()
    fig.savefig(subfigures_path.joinpath(f"{parameter}_density.png"))
    plt.close()
    
    import matplotlib.colors as colors

    # colormap
    cmap = plt.get_cmap('viridis', bounds[1])
    vmin = bounds[0]
    vmax = bounds[2]
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    logger.info(f"Plotting Concnetration Map...")
    fig, ax, cbar = plot_spain(isub, s=75.0, norm=norm, vmin=vmin, vmax=vmax, cmap=cmap, region="mainland")
    cbar.set_label(cbar_label)
    fig.set(dpi=300)
    plt.tight_layout()
    fig.savefig(subfigures_path.joinpath(f"{parameter}_map.png"))
    plt.close()
    
    plot_covariates(ds=isub, parameter="concentration", cbar_label=cbar_label, save_path=subfigures_path)

    
    return None


def plot_spain_location(
    az_ds,
    variable: str="t2max",
    covariate: str="gmst",
    cmap: str = "Reds",
    units: str = "[°C]",
    bounds: tuple = (20.0, 5, 45.0),
    save_path: str=""):
    
    logger.info(f"Plotting Location Parameter ...")
    
    parameter = "location"
    
    idata = az.extract(az_ds, group="posterior", num_samples=None)
    idata = idata.sortby(covariate)
    isub = idata[parameter].quantile(q=0.5, dim=["sample"]).load()

    subfigures_path = save_path.joinpath(f"posterior/{parameter}")
    subfigures_path.mkdir(parents=True, exist_ok=True)
    cbar_label = r"Location, $\boldsymbol{\mu}_1(\mathbf{s})$" + f" {units}"
        
    fig, ax = plot_density(isub)
    ax.set(
        title="",
        xlabel=cbar_label
    )
    fig.set(dpi=300)
    plt.tight_layout()
    fig.savefig(subfigures_path.joinpath(f"{parameter}_density.png"))
    plt.close()
    
    import matplotlib.colors as colors

    # colormap
    cmap = plt.get_cmap(cmap, bounds[1])
    vmin = bounds[0]
    vmax = bounds[2]
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    logger.info(f"Plotting Location Map...")
    fig, ax, cbar = plot_spain(isub, s=75.0, norm=norm, vmin=vmin, vmax=vmax, cmap=cmap, region="mainland")
    cbar.set_label(cbar_label)
    fig.set(dpi=300)
    plt.tight_layout()
    fig.savefig(subfigures_path.joinpath(f"{parameter}_map.png"))
    plt.close()
    
    plot_covariates(ds=isub, parameter="location", cbar_label=cbar_label, save_path=subfigures_path)

    
    return None




if __name__ == '__main__':
    app()   


