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


def plot_eda(da: xr.DataArray, variable_label: str, figures_path: str, figure_dpi: int=300):
    
    figures_path = Path(figures_path)
    

    logger.info(f"Plotting BM Data Time Series...")
    fig, ax, pts = plot_scatter_ts(da.squeeze(), markersize=5.0)
    ax.set(
        title="",
        xlabel=variable_label
    )
    fig.set(
        dpi=figure_dpi,
        size_inches=(7,4)
    )
    plt.tight_layout()
    subfigures_path = figures_path.joinpath("eda")
    subfigures_path.mkdir(parents=True, exist_ok=True)
    save_file_path = subfigures_path.joinpath("ts_bm_data.png")
    plt.savefig(save_file_path)
    plt.close()
    logger.debug(f"Saved Figure:\n{save_file_path}")
    
    logger.info(f"Plotting BM Data Histogram...")
    fig, ax = plot_histogram(da.squeeze())
    ax.set(
        title="",
        xlabel=variable_label
    )
    fig.set(
        dpi=figure_dpi,
        size_inches=(5,4)
    )
    plt.tight_layout()
    save_file_path = subfigures_path.joinpath("hist_bm_data.png")
    fig.savefig(save_file_path)
    plt.close()
    logger.debug(f"Saved Figure:\n{save_file_path}")
    
    logger.info(f"Plotting BM Data Density...")
    fig, ax = plot_density(da.squeeze())
    ax.set(
        title="",
        xlabel=variable_label
    )
    fig.set(
        dpi=figure_dpi,
        size_inches=(5,4)
    )
    plt.tight_layout()
    save_file_path = subfigures_path.joinpath("density_bm_data.png")
    fig.savefig(save_file_path)
    plt.close()
    logger.debug(f"Saved Figure:\n{save_file_path}")


def calculate_ds_return_periods(az_ds):

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
        input_core_dims=[[""], ["draw"], ["draw"], ["draw"]],
        output_core_dims=[["draw"]],
        vectorize=True
    )
    
    az_ds["return_level"] = xr.apply_ufunc(
        calculate_return_period,
        RETURN_PERIODS_GEVD,
        az_ds.location,
        az_ds.scale,
        az_ds.concentration,
        input_core_dims=[["return_period"], ["draw"], ["draw"], ["draw"]],
        output_core_dims=[["return_period", "draw"]],
        vectorize=True
    )
    
    az_ds = az_ds.assign_coords({"return_period": RETURN_PERIODS_GEVD})

    return az_ds


def plot_model_params_critique(
    ds,
    variables: list = [
        "concentration",
        "scale",
        "location_slope",
        "location_intercept",
        ],
    save_path: str = "./",
    figure_dpi: int = 300,
):
    save_path = Path(save_path)
    logger.info(f"Plotting Parameter Traces...")
    fig = az.plot_trace(
        ds, 
        var_names=variables,
        figsize=(10, 7)
    );
    plt.gcf().set_dpi(figure_dpi)
    plt.tight_layout()
    sub_figures_path = save_path.joinpath("params")
    sub_figures_path.mkdir(parents=True, exist_ok=True)
    save_file_name = sub_figures_path.joinpath("trace.png")
    plt.savefig(save_file_name)
    plt.close()
    logger.debug(f"Saved Figure:\n{save_file_name}")
    
    logger.info(f"Plotting Parameter Jonts...")
    fig = az.plot_pair(
        ds,
        # group="posterior",
        var_names=variables,
        kind=["scatter", "kde"],
        kde_kwargs={"fill_last": False},
        marginals=True,
        # coords=coords,
        point_estimate="median",
        figsize=(10, 8),
    )
    plt.gcf().set_dpi(figure_dpi)
    plt.tight_layout()
    sub_figures_path = save_path.joinpath("params")
    sub_figures_path.mkdir(parents=True, exist_ok=True)
    save_file_name = sub_figures_path.joinpath("joint.png")
    plt.savefig(save_file_name)
    plt.close()
    logger.debug(f"Saved Figure:\n{save_file_name}")

    
    return None


def plot_regression_posterior(
    x: xr.DataArray,
    y: xr.DataArray,
    y_model: xr.DataArray,
    y_hat: xr.DataArray,
    figures_path: str | None = None,
    figure_dpi: int = 300,
    covariate_label: str = "",
    y_label: str = "",
):  
    
    # INITIALIZE PLOT
    fig, ax = plt.subplots()
    
    # PLOT KWARGS
    kind_pp = "hdi"
    kind_model = "hdi"
    y_hat_plot_kwargs = dict(
        color="tab:red",
        markersize=10.0,
        # label=variable
    )
    y_model_mean_kwargs = dict(
        color="black",
        linewidth=5.0,
        # label=variable
    )
    
    y_kwargs = dict(
        color="tab:red",
        label="Measurements"
    )
    
    # PLOT
    az.plot_lm(
        axes=ax,
        y=y, y_hat=y_hat, x=x, y_model=y_model,
        kind_pp=kind_pp, kind_model=kind_model,
        y_kwargs=y_kwargs,
        y_hat_plot_kwargs=y_hat_plot_kwargs,
        y_model_mean_kwargs=y_model_mean_kwargs,
    )
    
    ax.set(
        xlabel=covariate_label,
        ylabel=y_label,
        xlim=[0.0, 1.3]
    )
    fig.set(dpi=figure_dpi)
    plt.grid(True, linestyle='--', linewidth='0.5', color='gray')
    plt.minorticks_on()
    plt.legend()
    plt.tight_layout()
    if figures_path is not None:
        figures_path = Path(figures_path)
        subfigures_path = figures_path.joinpath("regression")
        subfigures_path.mkdir(parents=True, exist_ok=True)
        save_path = subfigures_path.joinpath("regression.png")
        plt.savefig(save_path)
        plt.close()
        logger.debug(f"Saved Figure:\n{save_path}")
    else:
        return fig, ax
    

def plot_regression_prediction(
    ds_quantiles: xr.Dataset,
    observations: xr.DataArray,
    figures_path: str | None = None,
    figure_dpi: int = 300,
    covariate: str = "time",
    covariate_label: str = "",
    y_label: str = "",
    scale: bool = False,
    location_only: bool = False
):  
    
    # Grab Variables
    locations = ds_quantiles["location"]
    scales = ds_quantiles["scale"]
    return_level_100 = ds_quantiles["return_level_100"]
    
    from st_evt.viz import plot_locationscale_return_regression

    fig, ax = plot_locationscale_return_regression(
        locations,
        x_axis=covariate,
        scales=scales if not location_only else None,
        returns=return_level_100 if not location_only else None,
        observations=observations,
        observations_window=False
    )


    ax.set(
        ylabel=y_label,
        xlabel=covariate_label,
        xlim=[0.0, 2.5],
        # ylim=[-0.1, 1_000]
    )
    
    
    ax.set(
        xlabel=covariate_label,
        ylabel=y_label,
    )
    fig.set_size_inches(8, 4.5)
    fig.set(dpi=figure_dpi)
    plt.grid(True, linestyle='--', linewidth='0.5', color='gray')
    plt.minorticks_on()
    plt.legend()
    plt.tight_layout()
    if figures_path is not None:
        figures_path = Path(figures_path)
        subfigures_path = figures_path.joinpath("regression")
        subfigures_path.mkdir(parents=True, exist_ok=True)
        if location_only:
            save_path = subfigures_path.joinpath("regression_pred_location.png")
        else:
            save_path = subfigures_path.joinpath("regression_pred.png")
        plt.savefig(save_path)
        plt.close()
        logger.debug(f"Saved Figure:\n{save_path}")
    else:
        return fig, ax
    

def plot_residual_error_metric(
    y_pred: xr.DataArray,
    y_true: xr.DataArray,
    figures_path: str | None = None,
    figures_dpi: int = 300,
    units: str = ""
):
    # calculate residual error
    logger.info("Calculating residual error...")
    y_error = y_pred - y_true

    fig, ax = plot_density(y_error.median(dim=["draw", "chain"]))
    ax.set(
        title="",
        xlabel="Residuals" + f" {units}"
    )
    fig.set_size_inches(6, 4)
    fig.set_dpi(figures_dpi)
    plt.tight_layout()
    if figures_path is not None:
        figures_path = Path(figures_path)
        subfigures_path = figures_path.joinpath("metrics")
        subfigures_path.mkdir(parents=True, exist_ok=True)
        save_path = subfigures_path.joinpath("density_residuals.png")
        plt.savefig(save_path)
        logger.debug(f"Saved Figure:\n{save_path}")
        plt.close()
    else:
        return fig, ax


def plot_residual_abs_error_metric(
    y_pred: xr.DataArray,
    y_true: xr.DataArray,
    figures_path: str | None = None,
    figures_dpi: int = 300,
    units: str = ""
):
    # calculate residual error
    logger.info("Calculating residual error...")
    y_error = np.abs(y_pred - y_true)

    fig, ax = plot_density(y_error.median(dim=["draw", "chain"]))
    ax.set(
        title="",
        xlabel="Absolute Residuals" + f" {units}"
    )
    fig.set_size_inches(6, 4)
    fig.set_dpi(figures_dpi)
    plt.tight_layout()
    if figures_path is not None:
        figures_path = Path(figures_path)
        subfigures_path = figures_path.joinpath("metrics")
        subfigures_path.mkdir(parents=True, exist_ok=True)
        save_path = subfigures_path.joinpath("density_residuals_abs.png")
        plt.savefig(save_path)
        logger.debug(f"Saved Figure:\n{save_path}")
        plt.close()
    else:
        return fig, ax


def plot_qq(
    y_pred: xr.DataArray,
    y_true: xr.DataArray,
    figures_path: str | None = None,
    figures_dpi: int = 300,
):

    # ====================
    # results text
    # ====================
    from sklearn.metrics import root_mean_squared_error, mean_absolute_error

    logger.info("Calculating Metrics (RMSE, MAE, MAPD)...")
    metrics = {}
    metrics["rmse"] = root_mean_squared_error(y_true, y_pred)
    metrics["mae"] = mean_absolute_error(y_true, y_pred)
    # MAPD% of original data
    metrics['mapd'] = np.median(np.abs((y_pred) - (y_true)) / (y_true))


    import statsmodels.api as sm
    from scipy.stats import genextreme
    from matplotlib.offsetbox import AnchoredText
    
    # create Q-Q plot with 45-degree line added to plot
    logger.info("Plotting QQ-Plot...")
    fig, ax = plt.subplots()
    
    fig = sm.qqplot(y_pred, ax=ax, fit=True, line="45",  alpha=0.5, zorder=3)
    
    text = f"RMSE: {metrics['rmse']:.3f}"
    text += f"\nMAE: {metrics['mae']:.3f}"
    text += f"\nMAPD: {metrics['mapd']:.2%}"
    at = AnchoredText(
        text,
        prop=dict(fontsize=16), frameon=True,
        loc='upper left'
    )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    ax.autoscale(enable=True, axis='both', tight=True)
    plt.grid(True, linestyle='--', linewidth='0.5', color='gray')
    plt.gca().set_aspect('equal')
    plt.minorticks_on()
    plt.tight_layout()
    if figures_path is not None:
        figures_path = Path(figures_path)
        subfigures_path = figures_path.joinpath("metrics")
        subfigures_path.mkdir(parents=True, exist_ok=True)
        save_path = subfigures_path.joinpath("qq_plot.png")
        plt.savefig(save_path)
        logger.debug(f"Saved Figure:\n{save_path}")
        plt.close()
    else:
        return fig, ax


def calculate_empirical_return_level_gevd_ds(
    da: xr.DataArray,
    covariate: str,
    num_samples: int = 1_000,
    seed: int = 123,
):
    variable = da.name
    from st_evt.extremes import calculate_exceedence_probs
    logger.info("Calculating Return Level...")
    da["return_level"] = 1/xr.apply_ufunc(
        calculate_exceedence_probs,
        da,
        input_core_dims=[[covariate]],
        output_core_dims=[[covariate]],
        vectorize=True
    )
    logger.info("Swapping Dims...")
    da = da.swap_dims({covariate: variable})
    return da


def plot_return_periods_ds(
    rl_model_quantiles,
    y,
    covariate: str,
    y_label: str="",
    figures_path: str | None = None,
    figures_dpi: int = 300,
):
    rl_model_quantiles = rl_model_quantiles.drop_duplicates(covariate)
    rl_model_quantiles = rl_model_quantiles.sortby(covariate)
    logger.info("Getting Appropriate Periods...")
    min_period = rl_model_quantiles[covariate].min().values
    max_period = rl_model_quantiles[covariate].max().values

    logger.info("Intialize Returns...")
    returns = []
    
    
    logger.info("Creating Data structures...")
    returns.append({
        "color": "tab:green",
        "values":  rl_model_quantiles.sel({covariate: min_period}, method="nearest"),
        "label": r"Period Start"
    
    })
    
    returns.append({
        "color": "tab:red",
        "values":  rl_model_quantiles.sel({covariate: max_period}, method="nearest"),
        "label": r"Period End"
    })

    from st_evt.viz import plot_return_level_gevd_manual_unc_multiple
    
    logger.info("Plotting...")
    fig, ax = plot_return_level_gevd_manual_unc_multiple(
        returns,
        observations=y,
    )
    ax.set(
        ylabel=y_label,
    )
    
    plt.tight_layout()

    if figures_path is not None:
        figures_path = Path(figures_path)
        subfigures_path = figures_path.joinpath("returns")
        subfigures_path.mkdir(parents=True, exist_ok=True)
        save_path = subfigures_path.joinpath("returns_prob_posterior_vs_empirical.png")
        plt.savefig(save_path)
        logger.debug(f"Saved Figure:\n{save_path}")
        plt.close()
    else:
        return fig, ax
    
    
def plot_return_periods_gmst_ds(
    rl_model_quantiles,
    y,
    covariate: str,
    y_label: str="",
    figures_path: str | None = None,
    figures_dpi: int = 300,
):
    rl_model_quantiles = rl_model_quantiles.drop_duplicates(covariate)
    rl_model_quantiles = rl_model_quantiles.sortby(covariate)
    logger.info("Getting Appropriate Periods...")

    logger.info("Intialize Returns...")
    returns = []
    
    
    logger.info("Creating Data structures...")
    returns.append({
        "color": "tab:green",
        "values":  rl_model_quantiles.sel({covariate: 0.0}, method="nearest"),
        "label": "Pre-Industrial, 0.0 [°C]",
    
    })
    
    returns.append({
        "color": "tab:blue",
        "values":  rl_model_quantiles.sel({covariate: 1.3}, method="nearest"),
        "label": "Actual, 1.3 [°C]",
    
    })
    
    returns.append({
        "color": "tab:red",
        "values":  rl_model_quantiles.sel({covariate: 2.5}, method="nearest"),
        "label": "Future, 0.0 [°C]",
    })

    from st_evt.viz import plot_return_level_gevd_manual_unc_multiple
    
    logger.info("Plotting...")
    fig, ax = plot_return_level_gevd_manual_unc_multiple(
        returns,
        observations=y,
    )
    ax.set(
        ylabel=y_label,
    )
    
    plt.tight_layout()

    if figures_path is not None:
        figures_path = Path(figures_path)
        subfigures_path = figures_path.joinpath("returns")
        subfigures_path.mkdir(parents=True, exist_ok=True)
        save_path = subfigures_path.joinpath("returns_gmst_prob_posterior_vs_empirical.png")
        plt.savefig(save_path)
        logger.debug(f"Saved Figure:\n{save_path}")
        plt.close()
    else:
        return fig, ax


def plot_return_periods_100_ds(
    rl_model,
    covariate: str,
    x_label: str="",
    figures_path: str | None = None,
    figures_dpi: int = 300,
):
    
    rl_model = rl_model.drop_duplicates(covariate)
    rl_model = rl_model.sortby(covariate)
    logger.info("Getting Appropriate Periods...")
    min_period = rl_model[covariate].min().values
    max_period = rl_model[covariate].max().values

    logger.info("Intialize Returns...")
    returns = []
    
    
    logger.info("Creating Data structures...")
    returns.append({
        "color": "tab:green",
        "values":  rl_model.sel({covariate: min_period}, method="nearest").values,
        "period": "Period Start",
    
    })
    
    returns.append({
        "color": "tab:red",
        "values":  rl_model.sel({covariate: max_period}, method="nearest").values,
        "period": r"Period End"
    })

    from st_evt.viz import plot_density_multiple
    
    logger.info("Plotting...")
    fig, ax = plot_density_multiple(
        returns,
    )
    ax.set(
        xlabel=x_label,
    )
    fig.set_size_inches(6.5, 4)
    plt.legend(fontsize=10)
    plt.tight_layout()

    if figures_path is not None:
        figures_path = Path(figures_path)
        subfigures_path = figures_path.joinpath("returns")
        subfigures_path.mkdir(parents=True, exist_ok=True)
        save_path = subfigures_path.joinpath("returns_100years_density.png")
        plt.savefig(save_path)
        logger.debug(f"Saved Figure:\n{save_path}")
        plt.close()
    else:
        return fig, ax
    
    
def plot_return_periods_100_gmst_ds(
    rl_model,
    covariate: str,
    x_label: str="",
    figures_path: str | None = None,
    figures_dpi: int = 300,
):
    
    rl_model = rl_model.drop_duplicates(covariate)
    rl_model = rl_model.sortby(covariate)
    logger.info("Getting Appropriate Periods...")

    logger.info("Intialize Returns...")
    returns = []
    
    
    logger.info("Creating Data structures...")
    returns.append({
        "color": "tab:green",
        "values":  rl_model.sel({covariate: 0.0}, method="nearest").values.ravel(),
        "period": "Pre-Industrial, 0.0 [°C]",
    
    })
    
    returns.append({
        "color": "tab:blue",
        "values":  rl_model.sel({covariate: 1.3}, method="nearest").values.ravel(),
        "period": "Actual, 1.3 [°C]",
    
    })
    
    returns.append({
        "color": "tab:red",
        "values":  rl_model.sel({covariate: 2.5}, method="nearest").values.ravel(),
        "period": "Future, 0.0 [°C]",
    })

    from st_evt.viz import plot_density_multiple
    
    logger.info("Plotting...")
    fig, ax = plot_density_multiple(
        returns,
    )
    ax.set(
        xlabel=x_label,
    )
    fig.set_size_inches(6.5, 4)
    plt.legend(fontsize=10)
    plt.tight_layout()

    if figures_path is not None:
        figures_path = Path(figures_path)
        subfigures_path = figures_path.joinpath("returns")
        subfigures_path.mkdir(parents=True, exist_ok=True)
        save_path = subfigures_path.joinpath("returns_100years_gmst_density.png")
        plt.savefig(save_path)
        logger.debug(f"Saved Figure:\n{save_path}")
        plt.close()
    else:
        return fig, ax


def plot_return_periods_100_difference_ds(
    rl_model,
    covariate: str,
    x_label: str="",
    units: str="",
    color: str="gray",
    figures_path: str | None = None,
    figures_dpi: int = 300,
):
    rl_model = rl_model.drop_duplicates(covariate)
    rl_model = rl_model.sortby(covariate)
    logger.info("Getting Appropriate Periods...")
    min_period = rl_model[covariate].min().values
    max_period = rl_model[covariate].max().values

    logger.info("Calculating Difference...")

    period_0 = rl_model.sel({f"{covariate}": min_period}, method="nearest")
    period_1 = rl_model.sel({f"{covariate}": max_period}, method="nearest")
    diff_00_10 = period_1 - period_0

    logger.info("Intialize Returns...")
    returns = []
    
    
    logger.info("Creating Data structures...")
    returns_diffs = []
    
    returns_diffs.append({
        "period": r"",
        "color": color,
        "values":  diff_00_10.values.ravel(),
        "values_units": "",
        "label": "",
    })

    from st_evt.viz import plot_density_multiple
    
    logger.info("Plotting...")
    fig, ax = plot_density_multiple(
        returns_diffs,
    )
    ax.set(
        xlabel=x_label,
    )
    fig.set_size_inches(6.5, 4)
    plt.legend(fontsize=10)
    plt.tight_layout()

    if figures_path is not None:
        figures_path = Path(figures_path)
        subfigures_path = figures_path.joinpath("returns")
        subfigures_path.mkdir(parents=True, exist_ok=True)
        save_path = subfigures_path.joinpath("returns_100years_difference_density.png")
        plt.savefig(save_path)
        logger.debug(f"Saved Figure:\n{save_path}")
        plt.close()
    else:
        return fig, ax


def plot_return_periods_100_difference_gmst_ds(
    rl_model,
    covariate: str,
    x_label: str="",
    units: str="",
    figures_path: str | None = None,
    figures_dpi: int = 300,
):
    rl_model = rl_model.drop_duplicates(covariate)
    rl_model = rl_model.sortby(covariate)
    logger.info("Getting Appropriate Periods...")

    logger.info("Calculating Difference...")

    period_0 = rl_model.sel({f"{covariate}": 0.0}, method="nearest")
    period_1 = rl_model.sel({f"{covariate}": 1.3}, method="nearest")
    period_2 = rl_model.sel({f"{covariate}": 2.5}, method="nearest")
    diff_00_10 = period_1 - period_0
    diff_10_20 = period_2 - period_1

    logger.info("Intialize Returns...")
    
    
    logger.info("Creating Data structures...")
    returns_diffs = []
    
    returns_diffs.append({
        "period": r"$\Delta$ GMST: Pre-Industrial $\rightarrow$ Current",
        "color": "tab:orange",
        "values":  diff_00_10.values.ravel(),
        "values_units": "",
        "label": "",
    })
    returns_diffs.append({
        "period": r"$\Delta$ GMST: Current $\rightarrow$ Future",
        "color": "tab:red",
        "values":  diff_10_20.values.ravel(),
        "values_units": "",
        "label": "",
    })

    from st_evt.viz import plot_density_multiple
    
    logger.info("Plotting...")
    fig, ax = plot_density_multiple(
        returns_diffs,
    )
    ax.set(
        xlabel=x_label,
    )
    fig.set_size_inches(6.5, 4)
    plt.legend(fontsize=10)
    plt.tight_layout()

    if figures_path is not None:
        figures_path = Path(figures_path)
        subfigures_path = figures_path.joinpath("returns")
        subfigures_path.mkdir(parents=True, exist_ok=True)
        save_path = subfigures_path.joinpath("returns_100years_difference_gmst_density.png")
        plt.savefig(save_path)
        logger.debug(f"Saved Figure:\n{save_path}")
        plt.close()
    else:
        return fig, ax

  
def plot_return_periods_100_difference_prcnt_gmst_ds(
    rl_model,
    covariate: str,
    x_label: str="",
    figures_path: str | None = None,
    figures_dpi: int = 300,
):
    rl_model = rl_model.drop_duplicates(covariate)
    rl_model = rl_model.sortby(covariate)
    logger.info("Getting Appropriate Periods...")

    logger.info("Calculating Difference...")

    period_0 = rl_model.sel({f"{covariate}": 0.0}, method="nearest")
    period_1 = rl_model.sel({f"{covariate}": 1.3}, method="nearest")
    period_2 = rl_model.sel({f"{covariate}": 2.5}, method="nearest")
    diff_00_10 = ((period_1 - period_0) / np.abs(period_0)) * 100
    diff_10_20 = ((period_2 - period_1) / np.abs(period_1)) * 100

    logger.info("Intialize Returns...")
    
    
    logger.info("Creating Data structures...")
    returns_diffs = []
    
    returns_diffs.append({
        "period": r"$\Delta$ GMST: Pre-Industrial $\rightarrow$ Current",
        "color": "tab:orange",
        "values":  diff_00_10.values.ravel(),
        "values_units": "",
        "label": "",
    })
    returns_diffs.append({
        "period": r"$\Delta$ GMST: Current $\rightarrow$ Future",
        "color": "tab:red",
        "values":  diff_10_20.values.ravel(),
        "values_units": "",
        "label": "",
    })

    from st_evt.viz import plot_density_multiple
    
    logger.info("Plotting...")
    fig, ax = plot_density_multiple(
        returns_diffs,
    )
    ax.set(
        xlabel=x_label,
    )
    fig.set_size_inches(6.5, 4)
    plt.legend(fontsize=10)
    plt.tight_layout()

    if figures_path is not None:
        figures_path = Path(figures_path)
        subfigures_path = figures_path.joinpath("returns")
        subfigures_path.mkdir(parents=True, exist_ok=True)
        save_path = subfigures_path.joinpath("returns_100years_difference_prcnt_gmst_density.png")
        plt.savefig(save_path)
        logger.debug(f"Saved Figure:\n{save_path}")
        plt.close()
    else:
        return fig, ax


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
def evaluate_model_station_posterior(
    dataset_path: str = "",
    results_path: str = "",
    variable: str = "t2max",
    covariate: str = "gmst",
    station_id: str = "8414A",
    figures_path: str = "",
    figure_dpi: int=300,
):
    logger.info(f"Starting script...")
    # NONE
    
    dataset_path = Path(dataset_path)
    results_path = Path(results_path)
    figures_path = Path(figures_path)
    logger.debug(f"Dataset path {dataset_path}")
    logger.debug(f"Results path {results_path}")
    logger.debug(f"Figures path {figures_path}")
    logger.debug(f"Variable {variable}")
    logger.debug(f"Covariate {covariate}")
    logger.debug(f"Station ID {station_id}")
    
    logger.info(f"Load data...")
    logger.info(f"Creating figures directory...")
    
    
    # LOAD DATA
    with xr.open_dataset(dataset_path, engine="zarr") as f:
        ds_bm = f.load()
    az_ds = az.from_netcdf(results_path)
    
    logger.info("Selecting")
    az_ds_station_postpred = az_ds.posterior_predictive.sel(station_id = station_id)
    ds_station = ds_bm.sel(station_id = station_id)
    
    logger.info("Creating Station Directory...")
    figures_path = figures_path.joinpath(f"{station_id}")
    figures_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting EDA...")
    
    plot_eda(
        da=ds_station[variable],
        variable_label=VARIABLE_LABEL[variable],
        figures_path=figures_path,
        figure_dpi=figure_dpi
    )
    
    logger.info("Posterior Calculations...")
    az_ds_station_postpred = calculate_ds_return_periods(az_ds_station_postpred)
    
    logger.info("Starting EDA...")
    variables = [
        "concentration",
        "scale",
        "location_slope",
        "location_intercept",
        ]
    plot_model_params_critique(
        ds=az_ds_station_postpred,
        variables=variables,
        save_path=figures_path,
        figure_dpi=figure_dpi
        
    )
    
    logger.info("Starting Posterior Regression Plot...")
    x = az_ds_station_postpred[covariate]
    y = az_ds_station_postpred[f"{variable}_true"]
    y_hat = az_ds_station_postpred[variable]
    y_model = az_ds_station_postpred["location_slope"] * x + az_ds_station_postpred["location_intercept"]
    
    plot_regression_posterior(
        x=x,
        y=y,
        y_hat=y_hat,
        y_model=y_model,
        figures_path=figures_path,
        figure_dpi=figure_dpi,
        covariate_label=VARIABLE_LABEL[covariate],
        y_label=VARIABLE_LABEL[variable],
    )
    
    logger.info("Starting Model Metrics...")

    y_pred = az_ds_station_postpred[variable].rename("y_pred")
    y_true = az_ds_station_postpred[f"{variable}_true"]
        
    plot_residual_error_metric(
        y_pred=y_pred,
        y_true=y_true,
        figures_dpi=figure_dpi,
        figures_path=figures_path,
        units=UNITS[variable]
    )
    plot_residual_abs_error_metric(
        y_pred=y_pred,
        y_true=y_true,
        figures_dpi=figure_dpi,
        figures_path=figures_path,
        units=UNITS[variable]
    )

    y_pred_median = y_pred.mean(dim=["draw", "chain"])
    
    plot_qq(
        y_true=y_true,
        y_pred=y_pred_median,
        figures_path=figures_path,
        figures_dpi=figure_dpi,
    )
    
    # select clean data
    logger.info("Starting Model Returns...")
    
    logger.info("Calculating Empirical Return Levels...")
    y_clean = az_ds_station_postpred.dropna(dim=covariate)[f"{variable}_true"]

    # calculate return period
    y_clean = calculate_empirical_return_level_gevd_ds(y_clean, covariate=covariate)

    # calculate model return periods
    logger.info("Calculating Model Return Levels...")
    az_ds_station_postpred = calculate_ds_return_periods(az_ds_station_postpred)

    # Calculate Quantiles
    logger.info("Grabbing Quantiles...")
    rl_model_quantiles = az_ds_station_postpred["return_level"].quantile(q=[0.025, 0.5, 0.975], dim=["chain", "draw"])
    
    logger.info("Plotting Return Levels...")
    plot_return_periods_ds(
        rl_model_quantiles=rl_model_quantiles,
        y=y_clean,
        covariate=covariate,
        figures_path=figures_path,
        y_label=VARIABLE_LABEL_RETURNS[variable],
    )
    
    # calculate model return periods
    logger.info("Plotting 100-Year Return Periods...")
    az_ds_station_postpred = calculate_ds_return_periods(az_ds_station_postpred)

    # Calculate Quantiles
    rl_model = az_ds_station_postpred["return_level_100"]
    
    plot_return_periods_100_ds(
        rl_model=rl_model,
        covariate=covariate,
        figures_path=figures_path,
        x_label=VARIABLE_LABEL_RETURNS[variable]
    )
    
    logger.info("Plotting 100-Year Return Periods Difference...")
    plot_return_periods_100_difference_ds(
        rl_model=rl_model,
        covariate=covariate,
        figures_path=figures_path,
        x_label=VARIABLE_LABEL_RETURNS_DIFFERENCE_100[variable],
        units=UNITS[variable],
        color="black"
    )
                

@app.command()
def evaluate_model_station_predictions(
    dataset_path: str = "",
    results_path: str = "",
    variable: str = "t2max",
    covariate: str = "gmst",
    station_id: str = "8414A",
    figures_path: str = "",
    figure_dpi: int=300,
):
    logger.info(f"Starting script...")
    # NONE
    
    dataset_path = Path(dataset_path)
    results_path = Path(results_path)
    figures_path = Path(figures_path)
    logger.debug(f"Dataset path {dataset_path}")
    logger.debug(f"Results path {results_path}")
    logger.debug(f"Figures path {figures_path}")
    logger.debug(f"Variable {variable}")
    logger.debug(f"Covariate {covariate}")
    logger.debug(f"Station ID {station_id}")
    
    logger.info(f"Load data...")
    logger.info(f"Creating figures directory...")
    
    
    # LOAD DATA
    with xr.open_dataset(dataset_path, engine="zarr") as f:
        ds_bm = f.load()
    az_ds = az.from_netcdf(results_path)
    
    logger.info("Selecting")
    az_ds_station_pred = az_ds.predictions.sel(station_id = station_id)
    ds_station = ds_bm.sel(station_id = station_id)
    
    logger.info("Creating Station Directory...")
    figures_path = figures_path.joinpath(f"{station_id}")
    figures_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting EDA...")
    
    plot_eda(
        da=ds_station[variable],
        variable_label=VARIABLE_LABEL[variable],
        figures_path=figures_path,
        figure_dpi=figure_dpi
    )
    
    logger.info("Posterior Calculations...")
    az_ds_station_pred = calculate_ds_return_periods(az_ds_station_pred)
    
    logger.info("Starting EDA...")
    variables = [
        "concentration",
        "scale",
        "location_slope",
        "location_intercept",
        ]
    plot_model_params_critique(
        ds=az_ds_station_pred,
        variables=variables,
        save_path=figures_path,
        figure_dpi=figure_dpi
        
    )
    
    logger.info("Starting Predictions Regression Plot...")
    # calculate model return periods
    az_ds_station_pred = calculate_ds_return_periods(az_ds_station_pred)

    az_ds_quantiles = az_ds_station_pred.quantile(q=[0.025, 0.5, 0.975], dim=["chain", "draw"]).squeeze()
    locations = az_ds_station_pred["location"].quantile(q=[0.025, 0.5, 0.975], dim=["chain", "draw"]).squeeze()
    scales = az_ds_station_pred["scale"].quantile(q=[0.025, 0.5, 0.975], dim=["chain", "draw"]).squeeze()
    return_level_100 = az_ds_station_pred["return_level_100"].quantile(q=[0.025, 0.5, 0.975], dim=["chain", "draw"]).squeeze()
    observations = ds_station[variable].squeeze()
    
    plot_regression_prediction(
        ds_quantiles=az_ds_quantiles,
        observations=observations,
        figures_path=figures_path,
        figure_dpi=figure_dpi,
        covariate=covariate,
        covariate_label=VARIABLE_LABEL[covariate],
        y_label=VARIABLE_LABEL[variable],
        location_only=True
    )
    
    plot_regression_prediction(
        ds_quantiles=az_ds_quantiles,
        observations=observations,
        figures_path=figures_path,
        figure_dpi=figure_dpi,
        covariate=covariate,
        covariate_label=VARIABLE_LABEL[covariate],
        y_label=VARIABLE_LABEL[variable],
        location_only=False
    )
    
    # select clean data
    logger.info("Starting Model Returns...")
    
    logger.info("Calculating Empirical Return Levels...")
    y_clean = ds_station[variable].squeeze()

    # calculate return period
    y_clean = calculate_empirical_return_level_gevd_ds(y_clean, covariate=covariate)

    # Calculate Quantiles
    logger.info("Grabbing Quantiles...")
    rl_model_quantiles = az_ds_station_pred["return_level"].quantile(q=[0.025, 0.5, 0.975], dim=["chain", "draw"])
    
    logger.info("Plotting Return Levels...")
    plot_return_periods_gmst_ds(
        rl_model_quantiles=rl_model_quantiles,
        y=y_clean,
        covariate=covariate,
        figures_path=figures_path,
        y_label=VARIABLE_LABEL_RETURNS[variable],
    )
    
    # calculate model return periods
    logger.info("Plotting 100-Year Return Periods...")

    # Calculate Quantiles
    rl_model = az_ds_station_pred["return_level_100"]
    
    plot_return_periods_100_gmst_ds(
        rl_model=rl_model,
        covariate=covariate,
        figures_path=figures_path,
        x_label=VARIABLE_LABEL_RETURNS[variable]
    )
    
    logger.info("Plotting 100-Year Return Periods Difference...")
    plot_return_periods_100_difference_gmst_ds(
        rl_model=rl_model,
        covariate=covariate,
        figures_path=figures_path,
        x_label=VARIABLE_LABEL_RETURNS_DIFFERENCE_100[variable],
        units=UNITS[variable],
    )
    
    plot_return_periods_100_difference_prcnt_gmst_ds(
        rl_model=rl_model,
        covariate=covariate,
        figures_path=figures_path,
        x_label=VARIABLE_LABEL_RETURNS_DIFFERENCE_100[variable],
    )
                


if __name__ == '__main__':
    app()   


