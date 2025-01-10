import autoroot
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" # first gpu
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'FALSE'

import jax
jax.config.update('jax_platform_name', 'cpu')
import arviz as az
import numpy as np
import xarray as xr
from pathlib import Path
from loguru import logger
from st_evt.viz import (
    plot_scatter_ts,
    plot_histogram,
    plot_density,
)
import matplotlib.pyplot as plt
import matplotlib

plt.style.use(
    "https://raw.githubusercontent.com/ClimateMatchAcademy/course-content/main/cma.mplstyle"
)


def plot_model_params_critique(
    ds,
    variables: list = [
        "concentration",
        "scale",
        "location_slope",
        "location_intercept",
        ],
    figures_path: str = "./",
    figure_dpi: int = 300,
    figure_name: str | None = None
):
    figures_path = Path(figures_path)
    logger.info(f"Plotting Parameter Traces...")
    fig = az.plot_trace(
        ds, 
        var_names=variables,
        figsize=(10, 7)
    );
    plt.gcf().set_dpi(figure_dpi)
    plt.tight_layout()
    sub_figures_path = figures_path.joinpath("params")
    sub_figures_path.mkdir(parents=True, exist_ok=True)
    save_file_name = "trace.pdf" if figure_name is None else f"trace_{figure_name}.pdf"
    save_file_name = sub_figures_path.joinpath(save_file_name)
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
    sub_figures_path = figures_path.joinpath("params")
    sub_figures_path.mkdir(parents=True, exist_ok=True)
    save_file_name = "joint.pdf" if figure_name is None else f"joint_{figure_name}.pdf"
    save_file_name = sub_figures_path.joinpath(save_file_name)
    plt.savefig(save_file_name)
    plt.close()
    
    logger.debug(f"Saved Figure:\n{save_file_name}")
    logger.info(f"Plotting AutoCorrelation...")
    fig = az.plot_autocorr(
        ds, 
        var_names=variables,
        # figsize=(6,2.5),
        max_lag=50,
        combined=True,
    );
    # plt.gcf().set_dpi(figure_dpi)
    plt.tight_layout()
    save_name = figures_path.joinpath(f"autocorr.pdf")
    plt.savefig(save_name)
    logger.debug(f"Saved Figure: \n{save_name}")
    plt.close()
    
    logger.info(f"Plotting ESS...")
    fig = az.plot_ess(
        ds, 
        var_names=variables,
        # figsize=(6,2.5),
        kind="evolution",
        relative=True,
    );
    # plt.gcf().set_dpi(figure_dpi)
    plt.tight_layout()
    save_name = figures_path.joinpath(f"_ess.pdf")
    plt.savefig(save_name)
    logger.debug(f"Saved Figure: \n{save_name}")
    plt.close()
    return None


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
    save_file_path = subfigures_path.joinpath("ts_bm_data.pdf")
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
    save_file_path = subfigures_path.joinpath("hist_bm_data.pdf")
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
    save_file_path = subfigures_path.joinpath("density_bm_data.pdf")
    fig.savefig(save_file_path)
    plt.close()
    logger.debug(f"Saved Figure:\n{save_file_path}")


def calculate_ds_return_periods(az_ds, batch_variables: str = ["draw"]):

    from st_evt.extremes import estimate_return_level_gevd, calculate_exceedence_probs

    RETURN_PERIODS_GEVD = np.logspace(0.001, 4, 100)
    
    fn_gevd = jax.jit(estimate_return_level_gevd)
    
    def calculate_return_period(return_periods, location, scale, shape):
        rl = jax.vmap(fn_gevd, in_axes=(0,None,None,None))(return_periods, location, scale, shape)
        return rl
    
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
    
    az_ds = az_ds.assign_coords({"return_period": RETURN_PERIODS_GEVD})

    return az_ds

def calculate_ds_return_periods_100(az_ds, batch_variables: str = ["draw"]):

    from st_evt.extremes import estimate_return_level_gevd, calculate_exceedence_probs
    
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
    

    return az_ds


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
        save_path = subfigures_path.joinpath("density_residuals.pdf")
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
        save_path = subfigures_path.joinpath("density_residuals_abs.pdf")
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
        save_path = subfigures_path.joinpath("qq_plot.pdf")
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
    y_label: str="",
    figures_path: str | None = None,
    figures_dpi: int = 300,
):

    logger.info("Intialize Returns...")
    returns = []
    
    
    logger.info("Creating Data structures...")
    returns.append({
        "color": "tab:red",
        "values":  rl_model_quantiles,
        "label": r"Mean Return Level"
    
    })

    from st_evt.viz import plot_return_level_gevd_manual_unc
    
    logger.info("Plotting...")
    fig, ax = plot_return_level_gevd_manual_unc(
        return_level=rl_model_quantiles,
        return_periods=rl_model_quantiles.return_period,
        observations=y
    )
    ax.set(
        ylabel=y_label,
    )
    
    plt.tight_layout()

    figures_path = Path(figures_path)
    subfigures_path = figures_path.joinpath("returns")
    subfigures_path.mkdir(parents=True, exist_ok=True)
    save_path = subfigures_path.joinpath("returns_prob_posterior_vs_empirical.pdf")
    plt.savefig(save_path)
    logger.debug(f"Saved Figure:\n{save_path}")
    plt.close()
    return None


def plot_return_periods_dyn_ds(
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

    figures_path = Path(figures_path)
    subfigures_path = figures_path.joinpath("returns")
    subfigures_path.mkdir(parents=True, exist_ok=True)
    save_path = subfigures_path.joinpath("returns_prob_posterior_vs_empirical.pdf")
    plt.savefig(save_path)
    logger.debug(f"Saved Figure:\n{save_path}")
    plt.close()
    return None
    

def plot_return_periods_gmst_ds(
    rl_model_quantiles,
    y,
    covariate: str,
    y_label: str="",
    figures_path: str = "./",
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
        "label": "Future, 2.5 [°C]",
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

    figures_path = Path(figures_path)
    subfigures_path = figures_path.joinpath("returns")
    subfigures_path.mkdir(parents=True, exist_ok=True)
    save_path = subfigures_path.joinpath("returns_gmst_prob_posterior_vs_empirical.png")
    plt.savefig(save_path)
    logger.debug(f"Saved Figure:\n{save_path}")
    plt.close()


def plot_return_periods_100_ds(
    rl_model,
    x_label: str="",
    figures_path: str | None = None,
    figures_dpi: int = 300,
):
    
    logger.info("Plotting...")
    returns = []
    returns.append({
        "color": "black",
        "values":  rl_model.values,
        "period": r"KDE Fit"
    })

    from st_evt.viz import plot_return_level_hist_manual_unc
    
    logger.info("Plotting...")
    fig, ax = plot_return_level_hist_manual_unc(
        rl_model,
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
        save_path = subfigures_path.joinpath("returns_100years_density.pdf")
        plt.savefig(save_path)
        logger.debug(f"Saved Figure:\n{save_path}")
        plt.close()
    else:
        return fig, ax
    
    
def plot_return_periods_100_dyn_ds(
    rl_model,
    x_label: str="",
    covariate: str = "gmst",
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
        "linestyle": "-"
    
    })
    
    returns.append({
        "color": "tab:red",
        "values":  rl_model.sel({covariate: max_period}, method="nearest").values,
        "period": r"Period End",
        "linestyle": "--",
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

    figures_path = Path(figures_path)
    subfigures_path = figures_path.joinpath("returns")
    subfigures_path.mkdir(parents=True, exist_ok=True)
    save_path = subfigures_path.joinpath("returns_100years_density.pdf")
    plt.savefig(save_path)
    logger.debug(f"Saved Figure:\n{save_path}")
    plt.close()
    return None
 
 
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
        "period": "Future, 2.5 [°C]",
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

    figures_path = Path(figures_path)
    subfigures_path = figures_path.joinpath("returns")
    subfigures_path.mkdir(parents=True, exist_ok=True)
    save_path = subfigures_path.joinpath("returns_100years_gmst_density.pdf")
    plt.savefig(save_path)
    logger.debug(f"Saved Figure:\n{save_path}")
    plt.close()
    return None


def plot_return_periods_100_difference_dyn_ds(
    rl_model,
    covariate: str,
    x_label: str="",
    units: str="",
    color: str="gray",
    figures_path: str = "./",
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

    figures_path = Path(figures_path)
    subfigures_path = figures_path.joinpath("returns")
    subfigures_path.mkdir(parents=True, exist_ok=True)
    save_path = subfigures_path.joinpath("returns_100years_difference_density.pdf")
    plt.savefig(save_path)
    logger.debug(f"Saved Figure:\n{save_path}")
    plt.close()


def plot_return_periods_100_difference_prct_dyn_ds(
    rl_model,
    covariate: str,
    x_label: str="",
    color: str="gray",
    figures_path: str = "./",
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
    diff_00_10 = ((period_1 - period_0) / np.abs(period_0)) * 100

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

    figures_path = Path(figures_path)
    subfigures_path = figures_path.joinpath("returns")
    subfigures_path.mkdir(parents=True, exist_ok=True)
    save_path = subfigures_path.joinpath("returns_100years_difference_prct_density.pdf")
    plt.savefig(save_path)
    logger.debug(f"Saved Figure:\n{save_path}")
    plt.close()


def plot_return_periods_100_difference_gmst_ds(
    rl_model,
    covariate: str,
    x_label: str="",
    units: str="",
    figures_path: str = "./",
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

    figures_path = Path(figures_path)
    subfigures_path = figures_path.joinpath("returns")
    subfigures_path.mkdir(parents=True, exist_ok=True)
    save_path = subfigures_path.joinpath("returns_100years_difference_gmst_density.pdf")
    plt.savefig(save_path)
    logger.debug(f"Saved Figure:\n{save_path}")
    plt.close()


def plot_return_periods_100_difference_prct_gmst_ds(
    rl_model,
    covariate: str,
    x_label: str="",
    figures_path: str = "./",
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

    figures_path = Path(figures_path)
    subfigures_path = figures_path.joinpath("returns")
    subfigures_path.mkdir(parents=True, exist_ok=True)
    save_path = subfigures_path.joinpath("returns_100years_difference_prcnt_gmst_density.pdf")
    plt.savefig(save_path)
    logger.debug(f"Saved Figure:\n{save_path}")
    plt.close()



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
        save_path = subfigures_path.joinpath("regression.pdf")
        plt.savefig(save_path)
        plt.close()
        logger.debug(f"Saved Figure:\n{save_path}")
    else:
        return fig, ax
    
    
def plot_regression_prediction(
    ds_quantiles: xr.Dataset,
    observations: xr.DataArray,
    figures_path: str = "./",
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

