import autoroot
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" # first gpu
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'FALSE'

import jax
jax.config.update('jax_platform_name', 'cpu')

from pathlib import Path
from loguru import logger
import numpy as np
import xarray as xr
import arviz as az
from st_evt.viz import (
    plot_density,
    plot_density_multiple,
    plot_spain
)
import matplotlib.pyplot as plt
import matplotlib.colors as colors
plt.style.use(
    "https://raw.githubusercontent.com/ClimateMatchAcademy/course-content/main/cma.mplstyle"
)


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
    "concentration_iid": r"Shape, $\kappa_0(\mathbf{s})$",
    "location": r"Location, $\boldsymbol{\mu}(\mathbf{s},t)$ [°C]",
    "location_iid": r"Location, $\boldsymbol{\mu}_0(\mathbf{s})$ [°C]",
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
    "scale_iid": r"Scale, $\boldsymbol{\sigma}_0(\mathbf{s})$ [°C]",
    "scale_mean_intercept": r"Mean Intercept, $\alpha_{\sigma_1}$",
    "scale_mean_slope_x": r"Mean Slope (X-Coord), $\beta_{\sigma_1}$",
    "scale_mean_slope_y": r"Mean Slope (Y-Coord), $\beta_{\sigma_1}$",
    "scale_mean_slope_z": r"Mean Slope (Z-Coord), $\beta_{\sigma_1}$",
    "scale_kernel_variance": r"Kernel Variance, $\nu_{\sigma_1}$",
    "scale_kernel_scale_x": r"Kernel Scale (X-Coord), $\ell_{\sigma_2}$",
    "scale_kernel_scale_y": r"Kernel Scale (Y-Coord), $\ell_{\sigma_2}$",
    "scale_kernel_scale_z": r"Kernel Scale (Z-Coord), $\ell_{\sigma_2}$",
}


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


def plot_static_global_variable(
    ds,
    figures_path: str = "",
    cbar_label: str = "",
    figure_dpi: int = 300,
    mcmc_stats: bool = False,
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
    save_name = figures_path.joinpath(f"{variable}_density.pdf")
    fig.savefig(save_name)
    logger.debug(f"Saved Figure: \n{save_name}")
    plt.close()
    
    from st_evt._src.modules.models.aemet.utils_station import plot_model_params_critique
    
    logger.info(f"Plotting Parameter Traces...")
    fig = az.plot_trace(
        ds, 
        var_names=[variable],
        figsize=(6,2.5)
    );
    plt.gcf().set_dpi(figure_dpi)
    plt.tight_layout()
    save_name = figures_path.joinpath(f"{variable}_trace.pdf")
    plt.savefig(save_name)
    logger.debug(f"Saved Figure: \n{save_name}")
    plt.close()
    
    

    logger.info(f"Plotting AutoCorrelation...")
    fig = az.plot_autocorr(
        ds, 
        var_names=[variable],
        # figsize=(6,2.5),
        max_lag=100,
        combined=True,
    );
    # plt.gcf().set_dpi(figure_dpi)
    plt.tight_layout()
    save_name = figures_path.joinpath(f"{variable}_autocorr.pdf")
    plt.savefig(save_name)
    logger.debug(f"Saved Figure: \n{save_name}")
    plt.close()
    
    logger.info(f"Plotting ESS...")
    fig = az.plot_ess(
        ds, 
        var_names=[variable],
        # figsize=(6,2.5),
        kind="evolution",
        relative=True,
    );
    # plt.gcf().set_dpi(figure_dpi)
    plt.tight_layout()
    save_name = figures_path.joinpath(f"{variable}_ess.pdf")
    plt.savefig(save_name)
    logger.debug(f"Saved Figure: \n{save_name}")
    plt.close()
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
    save_name = figures_path.joinpath(f"{variable}_density.pdf")
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
        xlim=[bounds[0],bounds[2]]
        
    )
    plt.legend()
    save_name = figures_path.joinpath(f"{variable}_density_groups.pdf")
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
    save_name = figures_path.joinpath(f"{variable}_map.pdf")
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
        xlim=[bounds[0],bounds[2]]
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
        xlim=[bounds[0],bounds[2]]
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

