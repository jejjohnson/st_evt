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
import matplotlib.colors as colors
from st_evt._src.models.gevd import StationaryUnPooledGEVD
from st_evt.viz import (
    plot_scatter_ts,
    plot_histogram,
    plot_density,
    plot_density_multiple,
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

VARIABLE_LABEL_RETURNS = {
    "t2max": r"2m Max Temperature, $R_a$ [°C]",
    "pr": r"Cumulative Daily Precipitation, $R_a$ [mm/day]",
    "gmst": r"Global Mean Surface Temperature, $R_a$ [°C]"
}

VARIABLE_LABEL_RETURNS_100 = {
    "t2max": r"2m Max Temperature, $R_{100}$ [°C]",
    "pr": r"Cumulative Daily Precipitation, $R_{100}$ [mm/day]",
    "gmst": r"Global Mean Surface Temperature, $R_{100}$ [°C]"
}

VARIABLE_LABEL_RETURNS_DIFFERENCE_100 = {
    "t2max": r"2m Max Temperature, $\Delta R_{100}$ [°C]",
    "pr": r"Cumulative Daily Precipitation, $\Delta R_{100}$ [mm/day]",
    "gmst": r"Global Mean Surface Temperature, $\Delta R_{100}$ [°C]"
}

VARIABLE_LABELS = {
    "nll": (
        "Negative Log Predictive Density\n" +
        r"$\boldsymbol{L}(\mathbf{y};\boldsymbol{\theta},\boldsymbol{\phi},\mathcal{D})$"
    ),
    "residuals": "Residual [°C]",
    "residuals_abs": "Absolute Residual [°C]",
    "return_level_100": r"100-Year Return Level, $\boldsymbol{R}_{100}(\mathbf{s},t)$ [°C]",
    "return_level_100_spatial": r"100-Year Return Level, $\boldsymbol{R}_{100}(\mathbf{s})$ [°C]",
    "return_level_100_spatiotemporal": r"100-Year Return Level, $\boldsymbol{R}_{100}(\mathbf{s},t)$ [°C]",
    "return_level_100_difference": r"100-Year Return Level Difference, $\Delta\boldsymbol{R}_{100}(\mathbf{s},t)$ [°C]",
    "concentration": r"Shape, $\kappa_0$",
    "location": r"Location, $\boldsymbol{\mu}(\mathbf{s},t)$ [°C]",
    "location_mean_intercept": r"Mean Intercept, $\alpha_{\mu_1}$",
    "location_mean_slope_x": r"Mean Slope (X-Coord), $\beta_{\mu_1}$",
    "location_mean_slope_y": r"Mean Slope (Y-Coord), $\beta_{\mu_1}$",
    "location_mean_slope_z": r"Mean Slope (Z-Coord), $\beta_{\mu_1}$",
    "location_kernel_variance": r"Kernel Variance, $\nu_{\mu_1}$",
    "location_kernel_scale_x": r"Kernel Scale (X-Coord), $\ell_{\mu_1}$",
    "location_kernel_scale_y": r"Kernel Scale (Y-Coord), $\ell_{\mu_1}$",
    "location_kernel_scale_z": r"Kernel Scale (Z-Coord), $\ell_{\mu_1}$",
    "location_difference": r"Location Difference, $\Delta\boldsymbol{\mu}(\mathbf{s},t)$ [°C]",
    "location_intercept": r"Location Intercept, $\boldsymbol{\mu}_1(\mathbf{s})$ [°C]",
    "location_intercept_mean_intercept": r"Mean Intercept, $\alpha_{\mu_1}$",
    "location_intercept_mean_slope_x": r"Mean Slope (X-Coord), $\beta_{\mu_1}$",
    "location_intercept_mean_slope_y": r"Mean Slope (Y-Coord), $\beta_{\mu_1}$",
    "location_intercept_mean_slope_z": r"Mean Slope (Z-Coord), $\beta_{\mu_1}$",
    "location_intercept_kernel_variance": r"Kernel Variance, $\nu_{\mu_1}$",
    "location_intercept_kernel_scale_x": r"Kernel Scale (X-Coord), $\ell_{\mu_1}$",
    "location_intercept_kernel_scale_y": r"Kernel Scale (Y-Coord), $\ell_{\mu_1}$",
    "location_intercept_kernel_scale_z": r"Kernel Scale (Z-Coord), $\ell_{\mu_1}$",
    "location_slope": r"Location Slope, $\boldsymbol{\mu}_2(\mathbf{s})$ [°C]",
    "location_slope_mean_intercept": r"Mean Intercept, $\alpha_{\mu_2}$",
    "location_slope_mean_slope_x": r"Mean Slope (X-Coord), $\beta_{\mu_2}$",
    "location_slope_mean_slope_y": r"Mean Slope (Y-Coord), $\beta_{\mu_2}$",
    "location_slope_mean_slope_z": r"Mean Slope (Z-Coord), $\beta_{\mu_2}$",
    "location_slope_kernel_variance": r"Kernel Variance, $\nu_{\mu_2}$",
    "location_slope_kernel_scale_x": r"Kernel Scale (X-Coord), $\ell_{\mu_2}$",
    "location_slope_kernel_scale_y": r"Kernel Scale (Y-Coord), $\ell_{\mu_2}$",
    "location_slope_kernel_scale_z": r"Kernel Scale (Z-Coord), $\ell_{\mu_2}$",
    "scale": r"Scale, $\boldsymbol{\sigma}_1(\mathbf{s})$ [°C]",
    "scale_mean_intercept": r"Mean Intercept, $\alpha_{\sigma_1}$",
    "scale_mean_slope_x": r"Mean Slope (X-Coord), $\beta_{\sigma_1}$",
    "scale_mean_slope_y": r"Mean Slope (Y-Coord), $\beta_{\sigma_1}$",
    "scale_mean_slope_z": r"Mean Slope (Z-Coord), $\beta_{\sigma_1}$",
    "scale_kernel_variance": r"Kernel Variance, $\nu_{\sigma_1}$",
    "scale_kernel_scale_x": r"Kernel Scale (X-Coord), $\ell_{\sigma_2}$",
    "scale_kernel_scale_y": r"Kernel Scale (Y-Coord), $\ell_{\sigma_2}$",
    "scale_kernel_scale_z": r"Kernel Scale (Z-Coord), $\ell_{\sigma_2}$",
}


def plot_covariates(ds, cbar_label="", figure_dpi: int = 300, save_path: str=""):
    isub_mainland = ds.where(ds.lon > -10, drop=True )

    parameter = ds.name
    
    # LONGITUDE
    fig, ax = plt.subplots(figsize=(5,4.5))
    isub_mainland.to_dataset().plot.scatter(ax=ax,  x=parameter, y="lon", s=30.0, color="black")
    ax.set(
        xlabel=cbar_label,
        title=""
    )
    ax.grid(True, which="major", linestyle='--', linewidth='0.5', color='gray')
    ax.minorticks_on()
    fig.set(dpi=figure_dpi)
    plt.tight_layout()
    save_name = save_path.joinpath(f"{parameter}_covariates_lon.png")
    fig.savefig(save_name)
    logger.debug(f"Saved Figure: \n{save_name}")
    plt.close()
    
    # LONGITUDE
    fig, ax = plt.subplots(figsize=(5,4.5))
    isub_mainland.to_dataset().plot.scatter(ax=ax,  x=parameter, y="lat", s=30.0, color="black")
    ax.set(
        xlabel=cbar_label,
        title=""
    )
    ax.grid(True, which="major", linestyle='--', linewidth='0.5', color='gray')
    ax.minorticks_on()
    fig.set(dpi=figure_dpi)
    plt.tight_layout()
    save_name = save_path.joinpath(f"{parameter}_covariates_lat.png")
    fig.savefig(save_name)
    logger.debug(f"Saved Figure: \n{save_name}")
    plt.close()
    
    # LONGITUDE
    fig, ax = plt.subplots(figsize=(5,4.5))
    isub_mainland.to_dataset().plot.scatter(ax=ax,  x=parameter, y="alt", s=30.0, color="black")
    ax.set(
        xlabel=cbar_label,
        title=""
    )
    ax.grid(True, which="major", linestyle='--', linewidth='0.5', color='gray')
    ax.minorticks_on()
    fig.set(dpi=figure_dpi)
    plt.tight_layout()
    save_name = save_path.joinpath(f"{parameter}_covariates_alt.png")
    fig.savefig(save_name)
    logger.debug(f"Saved Figure: \n{save_name}")
    plt.close()
    return None


def plot_residuals_posterior_predictive(
    ds,
    figures_path: str = "",
    cbar_label: str = "",
    figure_dpi: int = 300,
    absolute: bool = False
):

    figures_path = Path(figures_path)
    variable = ds.name


    logger.info("Plotting NLL Density...")
    returns = []

    returns.append({
        "period": "Red Feten",
        "color": "tab:blue",
        "values":  ds.where(ds.red_feten_mask == 1, drop=True).values.ravel(),
        "linestyle": "-",
        "values_units": "", # "[mm/day]",
    })
    returns.append({
        "period": "Not Red Feten",
        "color": "tab:red",
        "linestyle": "--",
        "values":  ds.where(ds.red_feten_mask == 0, drop=True).values.ravel(),
        "values_units": "", # "[mm/day]",
    })
    fig, ax = plot_density_multiple(returns, log_bins=False if absolute else True)
    fig.set_size_inches(6, 5)
    ax.set(
        xlabel=cbar_label,
    )
    plt.legend()
    plt.tight_layout()
    save_name = figures_path.joinpath(f"density_{variable}_group.png")
    fig.savefig(save_name)
    logger.debug(f"Saved Figure: \n{save_name}")
    plt.close()

    logger.info("Plotting Map...")
    
    
    # colormap
    cmap = plt.get_cmap('coolwarm' if not absolute else "Reds", 10)
    vmin = -7.5 if not absolute else 0.0
    vmax = 7.5
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    logger.info("Plotting Map of Spain...")
    fig, ax, cbar = plot_spain(ds, s=30.0, norm=norm, vmin=vmin, vmax=vmax, cmap=cmap, region="mainland")
    cbar.set_label(cbar_label)
    fig.set(
        # dpi=figure_dpi,
        # size_inches=(10,4)
    )
    plt.tight_layout()
    save_name = figures_path.joinpath(f"map_{variable}.png")
    fig.savefig(save_name)
    logger.debug(f"Saved Figure: \n{save_name}")
    plt.close()

    logger.info("Plotting Covariates...")
    plot_covariates(ds,  cbar_label=cbar_label, figure_dpi=figure_dpi, save_path=figures_path)
    
    return None


def plot_static_spatial_variable_redfeten(
    ds,
    figures_path: str = "",
    cbar_label: str = "",
    figure_dpi: int = 300,
    cmap: str = "Reds",
    bounds: tuple = (0.5, 10, 3.0),
):
    logger.info("Starting Static Figures...")

    figures_path = Path(figures_path)
    
    variable = ds.name


    logger.info(f"Plotting {variable.upper()} Density...")
    returns = []
    returns.append({
        "period": "Red Feten",
        "color": "tab:blue",
        "values":  ds.where(ds.red_feten_mask == 1, drop=True).values.ravel(),
        "linestyle": "-",
        "values_units": "", # "[mm/day]",
    })
    returns.append({
        "period": "Not Red Feten",
        "color": "tab:red",
        "linestyle": "--",
        "values":  ds.where(ds.red_feten_mask == 0, drop=True).values.ravel(),
        "values_units": "", # "[mm/day]",
    })
    fig, ax = plot_density_multiple(returns, log_bins=False)
    fig.set_size_inches(6, 5)
    ax.set(
        xlabel=cbar_label,
    )
    save_name = figures_path.joinpath(f"{variable}_density_groups.png")
    fig.savefig(save_name)
    logger.debug(f"Saved Figure: \n{save_name}")
    plt.close()

    logger.info("Plotting Map...")
    
    
    # colormap
    cmap = plt.get_cmap(cmap, bounds[1])
    vmin = bounds[0]
    vmax = bounds[2]
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    logger.info("Plotting Map of Spain...")
    fig, ax, cbar = plot_spain(ds, s=50.0, norm=norm, vmin=vmin, vmax=vmax, cmap=cmap, region="mainland")
    cbar.set_label(cbar_label)
    fig.set(
        # dpi=figure_dpi,
        # size_inches=(10,4)
    )
    plt.tight_layout()
    save_name = figures_path.joinpath(f"{variable}_map.png")
    fig.savefig(save_name)
    logger.debug(f"Saved Figure: \n{save_name}")
    plt.close()

    logger.info("Plotting Covariates...")
    plot_covariates(ds,  cbar_label=cbar_label, figure_dpi=figure_dpi, save_path=figures_path)
    
    return None


def plot_static_spatial_variable(
    ds,
    figures_path: str = "",
    cbar_label: str = "",
    figure_dpi: int = 300,
    cmap: str = "Reds",
    bounds: tuple = (0.5, 10, 3.0),
):
    logger.info("Starting Static Figures...")

    figures_path = Path(figures_path)
    
    variable = ds.name


    logger.info(f"Plotting {variable.upper()} Density...")
    fig, ax = plot_density(ds)
    ax.set(
        title="",
        xlabel=cbar_label
    )
    fig.set(
        dpi=figure_dpi,
        size_inches=(5,4)
    )
    plt.tight_layout()
    save_name = figures_path.joinpath(f"{variable}_density.png")
    fig.savefig(save_name)
    logger.debug(f"Saved Figure: \n{save_name}")
    plt.close()

    logger.info("Plotting Map...")
    
    
    # colormap
    cmap = plt.get_cmap(cmap, bounds[1])
    vmin = bounds[0]
    vmax = bounds[2]
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    logger.info("Plotting Map of Spain...")
    fig, ax, cbar = plot_spain(ds, s=75.0, norm=norm, vmin=vmin, vmax=vmax, cmap=cmap, region="mainland")
    cbar.set_label(cbar_label)
    fig.set(
        # dpi=figure_dpi,
        # size_inches=(10,4)
    )
    plt.tight_layout()
    save_name = figures_path.joinpath(f"{variable}_map.png")
    fig.savefig(save_name)
    logger.debug(f"Saved Figure: \n{save_name}")
    plt.close()

    logger.info("Plotting Covariates...")
    plot_covariates(ds,  cbar_label=cbar_label, figure_dpi=figure_dpi, save_path=figures_path)
    
    return None


def plot_dynamic_spatial_variable_postpred(
    ds,
    covariate: str = "gmst",
    figures_path: str = "",
    cbar_label: str = "",
    figure_dpi: int = 300,
    cmap: str = "Reds",
    bounds: tuple = (0.5, 10, 3.0),
    units: str = "[°C]"
):
    
    ds = ds.drop_duplicates(covariate)
    ds = ds.sortby(covariate)
    min_period = ds[covariate].min().values
    max_period = ds[covariate].max().values
    figures_path = Path(figures_path)
    
    variable = ds.name


    logger.info(f"Plotting {variable.upper()} Density...")
    returns = []
    returns.append({
        "period": f"Period Start, {min_period:.1f} {units}",
        "color": "tab:green",
        "values":  ds.sel({covariate: min_period}, method="nearest").values.ravel(),
        "linestyle": "-",
        "values_units": "",
    })
    returns.append({
        "period":f"Period Start, {max_period:.1f} {units}",
        "color": "tab:red",
        "values":  ds.sel({covariate: max_period}, method="nearest").values.ravel(),
        "linestyle": "--",
        "values_units": "",
    })
    fig, ax = plot_density_multiple(returns, log_bins=False)
    fig.set_size_inches(6, 5)
    ax.set(
        xlabel=cbar_label,
    )
    plt.legend()
    save_name = figures_path.joinpath(f"{variable}_density_groups.png")
    fig.savefig(save_name)
    logger.debug(f"Saved Figure: \n{save_name}")
    plt.close()

    logger.info("Plotting Map...")
    
    for ilabel, icovariate in [("start", min_period), ("end", max_period)]:
        # colormap
        cmap = plt.get_cmap(cmap, bounds[1])
        vmin = bounds[0]
        vmax = bounds[2]
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        
        idata = ds.sel({covariate: icovariate}, method="nearest")
        logger.info("Plotting Map of Spain...")
        fig, ax, cbar = plot_spain(idata, s=75.0, norm=norm, vmin=vmin, vmax=vmax, cmap=cmap, region="mainland")
        cbar.set_label(cbar_label)
        plt.tight_layout()
        save_name = figures_path.joinpath(f"{variable}_map_{ilabel}.png")
        fig.savefig(save_name)
        logger.debug(f"Saved Figure: \n{save_name}")
        plt.close()

        logger.info("Plotting Covariates...")
        
        idata = idata.rename(f"{idata.name}_{ilabel}")
        plot_covariates(idata,  cbar_label=cbar_label, figure_dpi=figure_dpi, save_path=figures_path)
    
    return None


def plot_dynamic_spatial_variable_diff_postpred(
    ds,
    covariate: str = "gmst",
    figures_path: str = "",
    cbar_label: str = "",
    figure_dpi: int = 300,
    cmap: str = "Reds",
    bounds: tuple = (0.5, 10, 3.0),
    units: str = "[°C]"
):
    
    ds = ds.drop_duplicates(covariate)
    ds = ds.sortby(covariate)
    min_period = ds[covariate].min().values
    max_period = ds[covariate].max().values
    
    ds = (
        ds.sel({covariate: max_period}, method="nearest")
        - 
        ds.sel({covariate: min_period}, method="nearest")
    ).rename(f"{ds.name}_difference")
    figures_path = Path(figures_path)
    
    variable = ds.name


    logger.info(f"Plotting {variable.upper()} Density...")
    fig, ax = plot_density(ds)
    ax.set(
        title="",
        xlabel=cbar_label
    )
    fig.set(
        dpi=figure_dpi,
        size_inches=(5,4)
    )
    plt.tight_layout()
    save_name = figures_path.joinpath(f"{variable}_density.png")
    fig.savefig(save_name)
    logger.debug(f"Saved Figure: \n{save_name}")
    plt.close()

    logger.info("Plotting Map...")
    
    
    # colormap
    cmap = plt.get_cmap(cmap, bounds[1])
    vmin = bounds[0]
    vmax = bounds[2]
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    logger.info("Plotting Map of Spain...")
    fig, ax, cbar = plot_spain(ds, s=75.0, norm=norm, vmin=vmin, vmax=vmax, cmap=cmap, region="mainland")
    cbar.set_label(cbar_label)
    fig.set(
        # dpi=figure_dpi,
        # size_inches=(10,4)
    )
    plt.tight_layout()
    save_name = figures_path.joinpath(f"{variable}_map.png")
    fig.savefig(save_name)
    logger.debug(f"Saved Figure: \n{save_name}")
    plt.close()

    logger.info("Plotting Covariates...")
    plot_covariates(ds,  cbar_label=cbar_label, figure_dpi=figure_dpi, save_path=figures_path)
    
    return None


def plot_dynamic_spatial_variable_gmst_pred(
    ds,
    covariate: str = "gmst",
    figures_path: str = "",
    cbar_label: str = "",
    figure_dpi: int = 300,
    cmap: str = "Reds",
    bounds: tuple = (0.5, 10, 3.0),
    periods: list = [0.0, 1.3, 2.5],
):
    
    ds = ds.drop_duplicates(covariate)
    ds = ds.sortby(covariate)
    figures_path = Path(figures_path)
    
    variable = ds.name


    logger.info(f"Plotting {variable.upper()} Density...")
    returns = []
    returns.append({
        "period": f"Pre-Industrial, {periods[0]:.1f} [°C]",
        "color": "tab:green",
        "values":  ds.sel({covariate: periods[0]}, method="nearest").values.ravel(),
        "linestyle": "-",
        "values_units": "",
    })
    returns.append({
        "period": f"Current, {periods[1]:.1f} [°C]",
        "color": "tab:blue",
        "values":  ds.sel({covariate: periods[1]}, method="nearest").values.ravel(),
        "linestyle": "-",
        "values_units": "",
    })
    returns.append({
        "period": f"Current, {periods[2]:.1f} [°C]",
        "color": "tab:red",
        "values":  ds.sel({covariate: periods[2]}, method="nearest").values.ravel(),
        "linestyle": "-",
        "values_units": "",
    })
    fig, ax = plot_density_multiple(returns, log_bins=False)
    fig.set_size_inches(6, 5)
    ax.set(
        xlabel=cbar_label,
    )
    plt.legend()
    save_name = figures_path.joinpath(f"{variable}_density_groups.png")
    fig.savefig(save_name)
    logger.debug(f"Saved Figure: \n{save_name}")
    plt.close()

    logger.info("Plotting Map...")
    
    for icovariate in periods:
        # colormap
        cmap = plt.get_cmap(cmap, bounds[1])
        vmin = bounds[0]
        vmax = bounds[2]
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        
        idata = ds.sel({covariate: icovariate}, method="nearest")
        logger.info("Plotting Map of Spain...")
        fig, ax, cbar = plot_spain(idata, s=75.0, norm=norm, vmin=vmin, vmax=vmax, cmap=cmap, region="mainland")
        cbar.set_label(cbar_label)
        plt.tight_layout()
        save_name = figures_path.joinpath(f"{variable}_map_{icovariate}.png")
        fig.savefig(save_name)
        logger.debug(f"Saved Figure: \n{save_name}")
        plt.close()

    logger.info("Plotting Covariates...")
    
    plot_covariates(idata,  cbar_label=cbar_label, figure_dpi=figure_dpi, save_path=figures_path)
    
    return None


def plot_dynamic_spatial_variable_diff_gmst_pred(
    ds,
    covariate: str = "gmst",
    figures_path: str = "",
    cbar_label: str = "",
    figure_dpi: int = 300,
    cmap: str = "Reds",
    bounds: tuple = (0.5, 10, 3.0),
    periods: list = [0.0, 1.3, 2.5],
):
    
    assert len(periods) == 3
    ds = ds.drop_duplicates(covariate)
    ds = ds.sortby(covariate)
    ds_10_00 = (
        ds.sel({covariate: periods[1]}, method="nearest")
        - 
        ds.sel({covariate: periods[0]}, method="nearest")
    ).rename(f"{ds.name}_diff-10-00")
    
    ds_20_10 = (
        ds.sel({covariate: periods[2]}, method="nearest")
        - 
        ds.sel({covariate: periods[1]}, method="nearest")
    ).rename(f"{ds.name}_diff-20-10")
    
    figures_path = Path(figures_path)
    
    variable = ds.name


    logger.info(f"Plotting {variable.upper()} Density...")
    returns = []
    returns.append({
        "period": r"$\Delta$ GMST: Pre-Industrial $\rightarrow$ Current",
        "color": "tab:orange",
        "values": ds_10_00.values.ravel(),
        "linestyle": "-",
        "values_units": "",
    })
    returns.append({
        "period": r"$\Delta$ GMST: Current $\rightarrow$ Future",
        "color": "tab:red",
        "values":  ds_20_10.values.ravel(),
        "linestyle": "--",
        "values_units": "",
    })
    fig, ax = plot_density_multiple(returns, log_bins=False)
    fig.set_size_inches(6, 5)
    ax.set(
        xlabel=cbar_label,
    )
    plt.legend()
    save_name = figures_path.joinpath(f"{variable}_density_groups.png")
    fig.savefig(save_name)
    logger.debug(f"Saved Figure: \n{save_name}")
    plt.close()

    logger.info("Plotting Map...")
    
    for ilabel, ids in [("10-00", ds_10_00), ("20-10", ds_20_10)]:
        # colormap
        cmap = plt.get_cmap(cmap, bounds[1])
        vmin = bounds[0]
        vmax = bounds[2]
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        
        logger.info("Plotting Map of Spain...")
        fig, ax, cbar = plot_spain(ids, s=75.0, norm=norm, vmin=vmin, vmax=vmax, cmap=cmap, region="mainland")
        cbar.set_label(cbar_label)
        plt.tight_layout()
        save_name = figures_path.joinpath(f"{variable}_map_{ilabel}.png")
        fig.savefig(save_name)
        logger.debug(f"Saved Figure: \n{save_name}")
        plt.close()

        logger.info("Plotting Covariates...")
        
        ids = ids.rename(f"{ids.name}_{ilabel}")
        plot_covariates(ids,  cbar_label=cbar_label, figure_dpi=figure_dpi, save_path=figures_path)
    
    return None


def plot_static_global_variable(
    ds,
    figures_path: str = "",
    cbar_label: str = "",
    figure_dpi: int = 300,
):
    logger.info("Starting Static Figures...")

    figures_path = Path(figures_path)
    
    variable = ds.name


    logger.info(f"Plotting {variable.upper()} Density...")
    fig, ax = plot_density(ds)
    ax.set(
        title="",
        xlabel=cbar_label
    )
    fig.set(
        dpi=figure_dpi,
        size_inches=(5,4)
    )
    plt.tight_layout()
    save_name = figures_path.joinpath(f"{variable}_density.png")
    fig.savefig(save_name)
    logger.debug(f"Saved Figure: \n{save_name}")
    plt.close()
    
    return None


def calculate_ds_return_periods(az_ds, batch_variables: str = ["draw"]):

    from st_evt.extremes import estimate_return_level_gevd, calculate_exceedence_probs

    RETURN_PERIODS_GEVD = np.logspace(0.001, 4, 100)
    
    fn_gevd = jax.jit(estimate_return_level_gevd)
    
    def calculate_return_period(return_periods, location, scale, shape):
        rl = jax.vmap(fn_gevd, in_axes=(0,None,None,None))(return_periods, location, scale, shape)
        return rl

    az_ds["return_level_100"] = xr.apply_ufunc(
        calculate_return_period,
        [100],
        az_ds.location,
        az_ds.scale,
        az_ds.concentration,
        input_core_dims=[[""], batch_variables, batch_variables, batch_variables],
        output_core_dims=[batch_variables],
        vectorize=True
    )
    
    az_ds["return_level"] = xr.apply_ufunc(
        calculate_return_period,
        RETURN_PERIODS_GEVD,
        az_ds.location,
        az_ds.scale,
        az_ds.concentration,
        input_core_dims=[["return_period"], batch_variables, batch_variables, batch_variables],
        output_core_dims=[["return_period"] + batch_variables],
        vectorize=True
    )
    
    az_ds = az_ds.assign_coords({"return_period": RETURN_PERIODS_GEVD})

    return az_ds


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

    


if __name__ == '__main__':
    app()   

