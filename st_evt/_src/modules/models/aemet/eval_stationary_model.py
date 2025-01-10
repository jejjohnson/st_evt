import autoroot
from pathlib import Path
import arviz as az
import xarray as xr
from st_evt._src.modules.models.aemet import utils_station, utils_region
import typer
import numpy as np
from loguru import logger

app = typer.Typer()


@app.command()
def evaluate_model_posterior_station(
    results_dataset: str = "",
    dataset_url: str = "",
    figures_path: str = "",
    station_id: str = "3129A",
    variable: str = "t2max",
    spatial_dim_name: str = "station_id",
    covariate: str = "gmst"
):
    
    figures_path = Path(figures_path).joinpath(f"{station_id}/posterior")
    figures_path.mkdir(parents=True, exist_ok=True)
    
    # LOAD DATA
    logger.info("Loading Original Dataset")
    with xr.open_dataset(dataset_url, engine="zarr") as f:
        ds_bm = f.load()
        
    logger.info("Loading Datasets...")
    az_ds = az.from_zarr(str(results_dataset))
    az_ds_station = az_ds.sel(station_id = station_id)
    ds_station = ds_bm.sel(station_id = station_id)
    
    logger.info("Plotting EDA ...")
    utils_station.plot_eda(
        da=ds_station[variable].squeeze(),
        variable_label="2m Max Temperature [°C]",
        # figures_path="./", 
        figures_path=figures_path, 
        figure_dpi=300,
    )
    
    logger.info("Plotting Trace Plots ...")
    az_ds_station.posterior = utils_station.calculate_ds_return_periods_100(az_ds_station.posterior)
    variables = [
        "concentration",
        "scale",
        "location",
        "return_level_100"
        ]

    utils_station.plot_model_params_critique(
        ds=az_ds_station.posterior,
        variables=variables,
        # figures_path="./", 
        figures_path=figures_path, 
        
    )
    
    logger.info("Plotting Model Critique...")
    idata = az.extract(az_ds_station, group="posterior_predictive", num_samples=10_000)


    y_pred = az_ds_station.posterior_predictive[variable].rename("y_pred")
    y_true = az_ds_station.observed_data[variable]
    
    utils_station.plot_residual_error_metric(
        y_pred=y_pred,
        y_true=y_true,
        figures_dpi=300,
        # figures_path="./", 
        figures_path=figures_path, 
        units="[°C]"
    )
    utils_station.plot_residual_abs_error_metric(
        y_pred=y_pred,
        y_true=y_true,
        figures_dpi=300,
        # figures_path="./", 
        figures_path=figures_path, 
        units="[°C]"
    )
    
    logger.info("Plotting QQ-Plot...")
    y_pred_median = y_pred.mean(dim=["draw", "chain"])

    utils_station.plot_qq(
        y_true=y_true,
        y_pred=y_pred_median,
        # figures_path="./", 
        figures_path=figures_path, 
        figures_dpi=300,
    )
    
    logger.info("Plotting Return Levels...")
    
    # select clean data
    y_clean = az_ds_station.observed_data.dropna(dim=covariate)[variable]

    # calculate return period
    y_clean = utils_station.calculate_empirical_return_level_gevd_ds(y_clean, covariate=covariate)

    # calculate model return periods
    az_ds_station.posterior_predictive = utils_station.calculate_ds_return_periods(az_ds_station.posterior_predictive)

    # Calculate Quantiles
    rl_model_quantiles = az_ds_station.posterior_predictive["return_level"].quantile(q=[0.025, 0.5, 0.975], dim=["chain", "draw"])
    
    utils_station.plot_return_periods_ds(
        rl_model_quantiles=rl_model_quantiles,
        y=y_clean,
        # figures_path="./", 
        figures_path=figures_path, 
        y_label="2m Max Temperature, $R_a$ [°C]"
    )
    
    logger.info("Plotting 100-Year Return Level")
    # calculate model return periods
    az_ds_station.posterior_predictive = utils_station.calculate_ds_return_periods(az_ds_station.posterior_predictive)

    # Calculate Quantiles
    rl_model_quantiles = az_ds_station.posterior_predictive["return_level_100"]
            
    utils_station.plot_return_periods_100_ds(
        rl_model=rl_model_quantiles,
        # figures_path="./", 
        figures_path=figures_path, 
        x_label="2m Max Temperature, $R_{100}$ [°C]"
    )
    return None
    
@app.command()
def evaluate_model_posterior_predictive_station(
    results_dataset: str = "",
    dataset_url: str = "",
    figures_path: str = "",
    station_id: str = "3129A",
    variable: str = "t2max",
    spatial_dim_name: str = "station_id",
    covariate: str = "gmst"
):
    
    figures_path = Path(figures_path).joinpath(f"{station_id}/posterior_predictive")
    figures_path.mkdir(parents=True, exist_ok=True)
    
    # LOAD DATA
    logger.info("Loading Original Dataset")
    with xr.open_dataset(dataset_url, engine="zarr") as f:
        ds_bm = f.load()
        
    logger.info("Loading Datasets...")
    az_ds = az.from_zarr(store=str(results_dataset))
    az_ds_station_pred = az_ds.posterior_predictive.sel(station_id = station_id)
    y_data = az_ds.posterior_predictive.sel(station_id = station_id)[f"{variable}_true"]
    ds_station = ds_bm.sel(station_id = station_id)
    
    logger.info("Plotting EDA ...")
    utils_station.plot_eda(
        da=ds_station[variable].squeeze(),
        variable_label="2m Max Temperature [°C]",
        # figures_path="./", 
        figures_path=figures_path, 
        figure_dpi=300,
    )
    
    logger.info("Plotting Trace Plots ...")
    az_ds_station_pred = utils_station.calculate_ds_return_periods_100(az_ds_station_pred)
    variables = [
        "concentration",
        "scale",
        "location",
        "return_level_100"
        ]
    utils_station.plot_model_params_critique(
        ds=az_ds_station_pred,
        variables=variables,
        # figures_path="./", 
        figures_path=figures_path, 
        
    )

    logger.info("Plotting Return Levels...")
    
    # select clean data
    y_clean = y_data.dropna(dim=covariate)

    # calculate return period
    y_clean = utils_station.calculate_empirical_return_level_gevd_ds(y_clean, covariate=covariate)

    # calculate model return periods
    az_ds_station_pred = utils_station.calculate_ds_return_periods(az_ds_station_pred)

    # Calculate Quantiles
    rl_model_quantiles = az_ds_station_pred["return_level"].quantile(q=[0.025, 0.5, 0.975], dim=["chain", "draw"])
    
    utils_station.plot_return_periods_ds(
        rl_model_quantiles=rl_model_quantiles,
        y=y_clean,
        # figures_path="./", 
        figures_path=figures_path, 
        y_label="2m Max Temperature, $R_a$ [°C]"
    )
    
    logger.info("Plotting 100-Year Return Level")

    # Calculate Quantiles
    rl_model_quantiles = az_ds_station_pred["return_level_100"]
    
    utils_station.plot_return_periods_100_ds(
        rl_model=rl_model_quantiles,
        # figures_path="./", 
        figures_path=figures_path, 
        x_label="2m Max Temperature, $R_{100}$ [°C]"
    )
            
            
    return None
   

@app.command()
def evaluate_model_posterior_region(
    results_dataset: str = "",
    figures_path: str = "",
    variable: str = "t2max",
    spatial_dim_name: str = "station_id",
    covariate: str = "gmst",
    num_samples: int = 1_000,
):
    VARIABLE = variable
    COVARIATE = covariate
    
    logger.info("Loading Dataset...")
    az_ds = az.from_zarr(store=str(results_dataset))
    
    # logger.info("Plotting Log-Likelihood")
    # PLOT_VARIABLE = "nll"

    
    # data_results = az.extract(az_ds, group="log_likelihood", num_samples=num_samples).median(dim=["sample"]).load()
    # data_results = data_results.sortby(covariate)
    # idata = data_results[VARIABLE].rename(PLOT_VARIABLE).sum(dim=covariate)
    
    # subfigures_path = Path(figures_path).joinpath(f"{PLOT_VARIABLE}/posterior")
    # subfigures_path.mkdir(parents=True, exist_ok=True)

    # utils_region.plot_static_spatial_variable(
    #     idata,
    #     figures_path=subfigures_path,
    #     cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    #     cmap="Reds_r",
    #     bounds = (-160, 10, -50)
    # )
    
    # logger.info("Plotting Posterior-Predictive Log-Likelihood")
    # data_results = az.extract(az_ds, group="posterior_predictive", num_samples=num_samples).median(dim=["sample"]).load()
    # data_results = data_results.sortby(covariate)
    # idata = data_results["nll"].rename(PLOT_VARIABLE).sum(dim=covariate)
    
    # subfigures_path = Path(figures_path).joinpath(f"{PLOT_VARIABLE}/posterior_predictive")
    # subfigures_path.mkdir(parents=True, exist_ok=True)

    # utils_region.plot_static_spatial_variable_redfeten(
    #     idata,
    #     figures_path=subfigures_path,
    #     cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    #     cmap="Reds_r",
    #     bounds = (-160, 10, -50)
    # )
    
    # logger.info("Plotting Posterior-Predictive Residuals...")
    # PLOT_VARIABLE = "residuals"
    
    # data_results = az.extract(az_ds, group="posterior_predictive", num_samples=num_samples).median(dim=["sample"]).load()
    # data_results = data_results.sortby("gmst")
    # y_pred = data_results[variable].rename("y_pred")
    # y_true = data_results[f"{variable}_true"].rename("y_true")
    # idata = (y_true - y_pred).mean(dim=[covariate]).load().rename(PLOT_VARIABLE)
    
    # subfigures_path = Path(figures_path).joinpath(f"{PLOT_VARIABLE}/posterior_predictive")
    # subfigures_path.mkdir(parents=True, exist_ok=True)

    # utils_region.plot_static_spatial_variable_redfeten(
    #     idata,
    #     figures_path=subfigures_path,
    #     cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    #     cmap="Reds_r",
    #     bounds = (-7.5, 10, 7.5)
    # )
    
    # logger.info("Plotting Posterior-Predictive Absolute Residuals...")
    # PLOT_VARIABLE = "residuals_abs"
    
    # data_results = az.extract(az_ds, group="posterior_predictive", num_samples=num_samples).median(dim=["sample"]).load()
    # data_results = data_results.sortby("gmst")
    # y_pred = data_results[variable].rename("y_pred")
    # y_true = data_results[f"{variable}_true"].rename("y_true")
    # idata = np.abs(y_true - y_pred).mean(dim=[covariate]).load().rename(PLOT_VARIABLE)
    
    # subfigures_path = Path(figures_path).joinpath(f"{PLOT_VARIABLE}/posterior_predictive")
    # subfigures_path.mkdir(parents=True, exist_ok=True)

    # utils_region.plot_static_spatial_variable_redfeten(
    #     idata,
    #     figures_path=subfigures_path,
    #     cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    #     cmap="Reds",
    #     bounds = (0.0, 10, 7.5)
    # )
    
    # logger.info("Plotting SHAPE Parameter (POSTERIOR)...")
    # PLOT_VARIABLE = "concentration"
    
    
    # subfigures_path = Path(figures_path).joinpath(f"{PLOT_VARIABLE}/posterior")
    # subfigures_path.mkdir(parents=True, exist_ok=True)
    
    # idata = az_ds.posterior[PLOT_VARIABLE]
    
    # if len(idata.shape) == 2:

    #     utils_region.plot_static_global_variable(
    #         idata,
    #         figures_path=subfigures_path,
    #         cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    #     )
    # elif len(idata.shape) == 3:
    #     idata = az.extract(az_ds, group="posterior", num_samples=num_samples).median(dim=["sample"]).load()
    #     idata = idata[PLOT_VARIABLE]
    #     utils_region.plot_static_spatial_variable(
    #         idata,
    #         figures_path=subfigures_path,
    #         cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    #         cmap="Reds",
    #         bounds = (-1.0, 10, 0.0)
    #     )

        
    # logger.info("Plotting SHAPE Parameter (Posterior Predictive)...")
    
    
    # subfigures_path = Path(figures_path).joinpath(f"{PLOT_VARIABLE}/posterior_predictive")
    # subfigures_path.mkdir(parents=True, exist_ok=True)
    
    # idata = az_ds.posterior_predictive[PLOT_VARIABLE]
    
    # if len(idata.shape) == 2:

    #     utils_region.plot_static_global_variable(
    #         idata,
    #         figures_path=subfigures_path,
    #         cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    #     )
    # elif len(idata.shape) == 3:
    #     idata = az.extract(az_ds, group="posterior_predictive", num_samples=num_samples).median(dim=["sample"]).load()
    #     idata = idata[PLOT_VARIABLE]
    #     utils_region.plot_static_spatial_variable(
    #         idata,
    #         figures_path=subfigures_path,
    #         cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    #         cmap="Reds",
    #         bounds = (-1.0, 10, 0.0)
    #     )
        
    # logger.info("Plotting SCALE Parameter (POSTERIOR)...")
    # PLOT_VARIABLE = "scale"
    
    
    # subfigures_path = Path(figures_path).joinpath(f"{PLOT_VARIABLE}/posterior")
    # subfigures_path.mkdir(parents=True, exist_ok=True)
    
    # idata = az_ds.posterior[PLOT_VARIABLE]
    
    # if len(idata.shape) == 2:

    #     utils_region.plot_static_global_variable(
    #         idata,
    #         figures_path=subfigures_path,
    #         cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    #     )
    # elif len(idata.shape) == 3:
    #     idata = az.extract(az_ds, group="posterior", num_samples=num_samples).median(dim=["sample"]).load()
    #     idata = idata[PLOT_VARIABLE]
    #     utils_region.plot_static_spatial_variable(
    #         idata,
    #         figures_path=subfigures_path,
    #         cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    #         cmap="Reds",
    #         bounds = (0.5, 10, 3.0)
    #     )

        
    # logger.info("Plotting SCALE Parameter (Posterior Predictive)...")
    
    
    # subfigures_path = Path(figures_path).joinpath(f"{PLOT_VARIABLE}/posterior_predictive")
    # subfigures_path.mkdir(parents=True, exist_ok=True)
    
    # idata = az_ds.posterior_predictive[PLOT_VARIABLE]
    
    # if len(idata.shape) == 2:

    #     utils_region.plot_static_global_variable(
    #         idata,
    #         figures_path=subfigures_path,
    #         cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    #     )
    # elif len(idata.shape) == 3:
    #     idata = az.extract(az_ds, group="posterior_predictive", num_samples=num_samples).median(dim=["sample"]).load()
    #     idata = idata[PLOT_VARIABLE]
    #     utils_region.plot_static_spatial_variable(
    #         idata,
    #         figures_path=subfigures_path,
    #         cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    #         cmap="Reds",
    #         bounds = (0.5, 10, 3.0)
    #     )
        
    # logger.info("Plotting LOCATION Parameter (POSTERIOR)...")
    # PLOT_VARIABLE = "location"
    
    
    # subfigures_path = Path(figures_path).joinpath(f"{PLOT_VARIABLE}/posterior")
    # subfigures_path.mkdir(parents=True, exist_ok=True)
    
    # idata = az_ds.posterior[PLOT_VARIABLE]
    
    # if len(idata.shape) == 2:

    #     utils_region.plot_static_global_variable(
    #         idata,
    #         figures_path=subfigures_path,
    #         cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    #     )
    # elif len(idata.shape) == 3:
    #     idata = az.extract(az_ds, group="posterior", num_samples=num_samples).median(dim=["sample"]).load()
    #     idata = idata[PLOT_VARIABLE]
    #     utils_region.plot_static_spatial_variable(
    #         idata,
    #         figures_path=subfigures_path,
    #         cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    #         cmap="Reds",
    #         bounds = (20, 10, 45)
    #     )

        
    # logger.info("Plotting LOCATION Parameter (Posterior Predictive)...")
    
    
    # subfigures_path = Path(figures_path).joinpath(f"{PLOT_VARIABLE}/posterior_predictive")
    # subfigures_path.mkdir(parents=True, exist_ok=True)
    
    # idata = az_ds.posterior_predictive[PLOT_VARIABLE]
    
    # if len(idata.shape) == 2:

    #     utils_region.plot_static_global_variable(
    #         idata,
    #         figures_path=subfigures_path,
    #         cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    #     )
    # elif len(idata.shape) == 3:
    #     idata = az.extract(az_ds, group="posterior_predictive", num_samples=num_samples).median(dim=["sample"]).load()
    #     idata = idata[PLOT_VARIABLE]
    #     utils_region.plot_static_spatial_variable(
    #         idata,
    #         figures_path=subfigures_path,
    #         cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    #         cmap="Reds",
    #         bounds = (20, 10, 45)
    #     )


    logger.info("Plotting 100-YEAR RETURN PERIOD (POSTERIOR)...")
    PLOT_VARIABLE = "return_level_100"
    
    idata = az.extract(az_ds, group="posterior", num_samples=num_samples)
    idata = idata.sortby("gmst")
    idata = utils_station.calculate_ds_return_periods_100(idata, ["sample"])
    idata = idata[PLOT_VARIABLE]
    print(idata)
    
    subfigures_path = Path(figures_path).joinpath(f"{PLOT_VARIABLE}/posterior")
    subfigures_path.mkdir(parents=True, exist_ok=True)
    
    if len(idata.shape) == 1:

        utils_region.plot_static_global_variable(
            idata,
            figures_path=subfigures_path,
            cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
        )
    elif len(idata.shape) == 2:
        utils_region.plot_static_spatial_variable(
            idata.median(dim="sample"),
            figures_path=subfigures_path,
            cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
            cmap="Reds",
            bounds = (25, 10, 50)
        )

        
    logger.info("Plotting 100-YEAR RETURN PERIOD (Posterior Predictive)...")
    
    
    idata = az.extract(az_ds, group="posterior_predictive", num_samples=num_samples)
    idata = idata.sortby("gmst")
    idata = utils_station.calculate_ds_return_periods_100(idata, ["sample"])
    idata = idata[PLOT_VARIABLE]
    
    subfigures_path = Path(figures_path).joinpath(f"{PLOT_VARIABLE}/posterior_predictive")
    subfigures_path.mkdir(parents=True, exist_ok=True)
    
    if len(idata.shape) == 1:

        utils_region.plot_static_global_variable(
            idata,
            figures_path=subfigures_path,
            cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
        )
    if len(idata.shape) == 2:
        utils_region.plot_static_spatial_variable(
            idata.median(dim="sample"),
            figures_path=subfigures_path,
            cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
            cmap="Reds",
            bounds = (25, 10, 50)
        )
    
    return None


@app.command()
def evaluate_model_posterior_gp_params(
    results_dataset: str = "",
    figures_path: str = "",
    variable: str = "t2max",
    spatial_dim_name: str = "station_id",
    covariate: str = "gmst",
    num_samples: int = 1_000,
):
    VARIABLE = variable
    COVARIATE = covariate
    
    logger.info("Loading Dataset...")
    az_ds = az.from_zarr(store=str(results_dataset))
    
    logger.info("Plotting Location Kernel VARIANCE...")
    PLOT_VARIABLE = "location_kernel_variance"
    
    idata = az_ds.posterior[PLOT_VARIABLE]
    subfigures_path = Path(figures_path).joinpath(f"location/posterior")
    subfigures_path.mkdir(parents=True, exist_ok=True)

    utils_region.plot_static_global_variable(
        idata,
        figures_path=subfigures_path,
        cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    )
    
    logger.info("Plotting Location Kernel SCALES...")
    PLOT_VARIABLE = "location_kernel_scale"
    
    idata = az_ds.posterior[PLOT_VARIABLE]
    subfigures_path = Path(figures_path).joinpath(f"location/posterior")
    subfigures_path.mkdir(parents=True, exist_ok=True)

    utils_region.plot_static_global_variable(
        idata.sel(spherical="lon").rename(f"{PLOT_VARIABLE}_x"),
        figures_path=subfigures_path,
        cbar_label=utils_region.VARIABLE_LABELS[f"{PLOT_VARIABLE}_x"],
    )
    utils_region.plot_static_global_variable(
        idata.sel(spherical="lat").rename(f"{PLOT_VARIABLE}_y"),
        figures_path=subfigures_path,
        cbar_label=utils_region.VARIABLE_LABELS[f"{PLOT_VARIABLE}_y"],
    )
    utils_region.plot_static_global_variable(
        idata.sel(spherical="alt").rename(f"{PLOT_VARIABLE}_z"),
        figures_path=subfigures_path,
        cbar_label=utils_region.VARIABLE_LABELS[f"{PLOT_VARIABLE}_z"],
    )
    

    
    logger.info("Plotting SCALE Kernel VARIANCE...")
    PLOT_VARIABLE = "scale_kernel_variance"
    
    idata = az_ds.posterior[PLOT_VARIABLE]
    subfigures_path = Path(figures_path).joinpath(f"scale/posterior")
    subfigures_path.mkdir(parents=True, exist_ok=True)

    utils_region.plot_static_global_variable(
        idata,
        figures_path=subfigures_path,
        cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    )
    
    logger.info("Plotting SCALE Kernel SCALES...")
    PLOT_VARIABLE = "scale_kernel_scale"
    
    idata = az_ds.posterior[PLOT_VARIABLE]
    subfigures_path = Path(figures_path).joinpath(f"scale/posterior")
    subfigures_path.mkdir(parents=True, exist_ok=True)

    utils_region.plot_static_global_variable(
        idata.sel(spherical="lon").rename(f"{PLOT_VARIABLE}_x"),
        figures_path=subfigures_path,
        cbar_label=utils_region.VARIABLE_LABELS[f"{PLOT_VARIABLE}_x"],
    )
    utils_region.plot_static_global_variable(
        idata.sel(spherical="lat").rename(f"{PLOT_VARIABLE}_y"),
        figures_path=subfigures_path,
        cbar_label=utils_region.VARIABLE_LABELS[f"{PLOT_VARIABLE}_y"],
    )
    utils_region.plot_static_global_variable(
        idata.sel(spherical="alt").rename(f"{PLOT_VARIABLE}_z"),
        figures_path=subfigures_path,
        cbar_label=utils_region.VARIABLE_LABELS[f"{PLOT_VARIABLE}_z"],
    )
        
        
    logger.info("Plotting Location INTERCEPT Kernel VARIANCE...")
    PLOT_VARIABLE = "location_mean_intercept"
    
    idata = az_ds.posterior[PLOT_VARIABLE]
    subfigures_path = Path(figures_path).joinpath(f"location/posterior")
    subfigures_path.mkdir(parents=True, exist_ok=True)

    utils_region.plot_static_global_variable(
        idata,
        figures_path=subfigures_path,
        cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    )
    
    logger.info("Plotting Location INTERCEPT Kernel SCALES...")
    PLOT_VARIABLE = "location_mean_slope"
    
    idata = az_ds.posterior[PLOT_VARIABLE]
    subfigures_path = Path(figures_path).joinpath(f"location_/posterior")
    subfigures_path.mkdir(parents=True, exist_ok=True)

    utils_region.plot_static_global_variable(
        idata.sel(spherical="lon").rename(f"{PLOT_VARIABLE}_x"),
        figures_path=subfigures_path,
        cbar_label=utils_region.VARIABLE_LABELS[f"{PLOT_VARIABLE}_x"],
    )
    utils_region.plot_static_global_variable(
        idata.sel(spherical="lat").rename(f"{PLOT_VARIABLE}_y"),
        figures_path=subfigures_path,
        cbar_label=utils_region.VARIABLE_LABELS[f"{PLOT_VARIABLE}_y"],
    )
    utils_region.plot_static_global_variable(
        idata.sel(spherical="alt").rename(f"{PLOT_VARIABLE}_z"),
        figures_path=subfigures_path,
        cbar_label=utils_region.VARIABLE_LABELS[f"{PLOT_VARIABLE}_z"],
    )
    
    logger.info("Plotting SCALE Kernel VARIANCE...")
    PLOT_VARIABLE = "scale_mean_intercept"
    
    idata = az_ds.posterior[PLOT_VARIABLE]
    subfigures_path = Path(figures_path).joinpath(f"scale/posterior")
    subfigures_path.mkdir(parents=True, exist_ok=True)

    utils_region.plot_static_global_variable(
        idata,
        figures_path=subfigures_path,
        cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    )
    
    logger.info("Plotting SCALE Kernel SCALES...")
    PLOT_VARIABLE = "scale_mean_slope"
    
    idata = az_ds.posterior[PLOT_VARIABLE]
    subfigures_path = Path(figures_path).joinpath(f"scale/posterior")
    subfigures_path.mkdir(parents=True, exist_ok=True)

    utils_region.plot_static_global_variable(
        idata.sel(spherical="lon").rename(f"{PLOT_VARIABLE}_x"),
        figures_path=subfigures_path,
        cbar_label=utils_region.VARIABLE_LABELS[f"{PLOT_VARIABLE}_x"],
    )
    utils_region.plot_static_global_variable(
        idata.sel(spherical="lat").rename(f"{PLOT_VARIABLE}_y"),
        figures_path=subfigures_path,
        cbar_label=utils_region.VARIABLE_LABELS[f"{PLOT_VARIABLE}_y"],
    )
    utils_region.plot_static_global_variable(
        idata.sel(spherical="alt").rename(f"{PLOT_VARIABLE}_z"),
        figures_path=subfigures_path,
        cbar_label=utils_region.VARIABLE_LABELS[f"{PLOT_VARIABLE}_z"],
    )
        
    
    return None



if __name__ == "__main__":
    app()