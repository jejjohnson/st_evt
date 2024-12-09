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

VARIABLE_BOUNDS = {
    "t2max": (20.0, 6, 50.0),
    "t2max_mean": (20, 10, 45.0),
    "t2max_std": (0.0, 8, 4.0),
    "pr": (10, 8, 50.0),
    "pr_mean": (10, 10, 100.0),
    "pr_std": (10.0, 9, 100.0),
}

CMAP = {
    "t2max": "Reds",
    "t2max_mean": "Reds",
    "t2max_std": "viridis",
    "pr": "Blues",
    "pr_mean": "Blues",
    "pr_std": "Blues",
}


UNITS = {
    "t2max": "[°C]",
    "pr": "[mm/day]",
    "gmst": "[°C]",
}
VARIABLE_LABEL = {
    "t2max": "2m Max Temperature [°C]",
    "t2max_mean": "Mean 2m Max Temperature [°C]",
    "t2max_std": "Stddev 2m Max Temperature [°C]",
    "pr": "Cumulative Daily Precipitation [mm/day]",
    "pr_mean": "Cumulative Daily Precipitation [mm/day]",
    "pr_std": "Cumulative Daily Precipitation [mm/day]",
    "gmst": "Global Mean Surface Temperature [°C]"
}


@app.command()
def eda_station(
    dataset_path: str = "",
    extremes_path: str = "",
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
    

 
@app.command()
def eda_spain(
    dataset_path: str = "",
    extremes_path: str = "",
    variable: str = "t2max",
    covariate: str = "gmst",
    save_path: str = "",
):
    logger.info(f"Starting script...")
    #
    print("hi!")
    
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
    ds = xr.open_dataset(dataset_path)
    
    # logger.info(f"Calculating RP")
    # az_ds = calculate_ds_return_periods(az_ds)
    
    # statistics
    new_variable = f"{variable}_mean"
    da = ds[variable].mean(dim=covariate).rename(new_variable)
    
    plot_spain_variable(da=da, cmap=CMAP[new_variable], bounds=VARIABLE_BOUNDS[new_variable], save_path=save_path)
    
    # statistics
    new_variable = f"{variable}_std"
    da = ds[variable].std(dim=covariate).rename(new_variable)
    
    plot_spain_variable(da=da, cmap=CMAP[new_variable], bounds=VARIABLE_BOUNDS[new_variable], save_path=save_path)

    
def plot_spain_variable(
    da,
    cmap: str="viridis",
    bounds: tuple = (20.0, 5, 45.0),
    save_path: str=""):
    
    logger.info(f"Plotting Variable ...")
    
    name = da.name
    
    isub = da

    subfigures_path = save_path.joinpath(name)
    subfigures_path.mkdir(parents=True, exist_ok=True)
    cbar_label = VARIABLE_LABEL[name]
        
    fig, ax = plot_density(isub)
    ax.set(
        title="",
        xlabel=cbar_label
    )
    fig.set(dpi=300)
    plt.tight_layout()
    fig.savefig(subfigures_path.joinpath(f"{name}_density.png"))
    plt.close()
    
    fig, ax = plot_histogram(isub)
    ax.set(
        title="",
        xlabel=cbar_label
    )
    fig.set(dpi=300)
    plt.tight_layout()
    fig.savefig(subfigures_path.joinpath(f"{name}_hist.png"))
    plt.close()
    
    import matplotlib.colors as colors

    # colormap
    cmap = plt.get_cmap(cmap, bounds[1])
    vmin = bounds[0]
    vmax = bounds[2]
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    logger.info(f"Plotting {name} Map...")
    fig, ax, cbar = plot_spain(isub, s=75.0, norm=norm, vmin=vmin, vmax=vmax, cmap=cmap, region="mainland")
    cbar.set_label(cbar_label)
    fig.set(dpi=300)
    plt.tight_layout()
    fig.savefig(subfigures_path.joinpath(f"{name}_map.png"))
    plt.close()
    
    logger.info(f"Plotting covariates")
    isub_mainland = isub.where(isub.lon > -10, drop=True )
    
    # PLOT COVARIATES
    plot_covariates(ds=isub_mainland, parameter=name, cbar_label=cbar_label, save_path=subfigures_path)
    
    return None


def plot_covariates(ds, parameter: str="scale", cbar_label="", save_path: str=""):
    isub_mainland = ds.where(ds.lon > -10, drop=True )
    
    # LONGITUDE
    fig, ax = plt.subplots()
    isub_mainland.to_dataset().plot.scatter(ax=ax,  x=parameter, y="lon", s=10.0, color="black")
    ax.set(
        xlabel=cbar_label,
        title="",
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
        xlabel=cbar_label,
        title="",
    )
    ax.grid(True, which="major", linestyle='--', linewidth='0.5', color='gray')
    ax.minorticks_on()
    fig.set(dpi=300)
    plt.tight_layout()
    fig.savefig(save_path.joinpath(f"{parameter}_covariates_lat.png"))
    plt.close()
    
    # LONGITUDE
    fig, ax = plt.subplots()
    isub_mainland.to_dataset().plot.scatter(ax=ax,  x=parameter, y="alt", s=10.0, color="black")
    ax.set(
        xlabel=cbar_label,
        title="",
    )
    ax.grid(True, which="major", linestyle='--', linewidth='0.5', color='gray')
    ax.minorticks_on()
    fig.set(dpi=300)
    plt.tight_layout()
    fig.savefig(save_path.joinpath(f"{parameter}_covariates_alt.png"))
    plt.close()
    return None


if __name__ == '__main__':
    app()   


