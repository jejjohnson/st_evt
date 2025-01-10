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
    variables = [
        "concentration",
        "scale",
        "location_slope",
        "location_intercept",
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
    
    logger.info("Plotting Regression Plot...")
    x = az_ds_station.posterior_predictive[covariate]
    y = az_ds_station.observed_data[variable]
    y_hat = az_ds_station.posterior_predictive[variable]
    y_model = az_ds_station.posterior_predictive["location_slope"] * x + az_ds_station.posterior_predictive["location_intercept"]

    utils_station.plot_regression_posterior(
        x=x,
        y=y,
        y_hat=y_hat,
        y_model=y_model,
        # figures_path="./", 
        figures_path=figures_path, 
        figure_dpi=300,
        covariate_label="Global Mean Surface Temperature Anomaly [°C]",
        y_label="2m Max Temperature [°C]"
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
    
    utils_station.plot_return_periods_dyn_ds(
        rl_model_quantiles=rl_model_quantiles,
        y=y_clean,
        covariate=covariate,
        # figures_path="./", 
        figures_path=figures_path, 
        y_label="2m Max Temperature, $R_a$ [°C]"
    )
    
    logger.info("Plotting 100-Year Return Level")
    # calculate model return periods
    az_ds_station.posterior_predictive = utils_station.calculate_ds_return_periods(az_ds_station.posterior_predictive)

    # Calculate Quantiles
    rl_model_quantiles = az_ds_station.posterior_predictive["return_level_100"]
    
    utils_station.plot_return_periods_100_dyn_ds(
        rl_model=rl_model_quantiles,
        covariate=covariate,
        # figures_path="./", 
        figures_path=figures_path, 
        x_label="2m Max Temperature, $R_{100}$ [°C]"
    )
    
    logger.info("Plotting 100-Year Return Level Difference")
    # calculate model return periods
    az_ds_station.posterior_predictive = utils_station.calculate_ds_return_periods(az_ds_station.posterior_predictive)

    # Calculate Quantiles
    rl_model = az_ds_station.posterior_predictive["return_level_100"]
    
    utils_station.plot_return_periods_100_difference_dyn_ds(
        rl_model=rl_model,
        covariate=covariate,
        # figures_path="./", 
        figures_path=figures_path, 
        x_label="2m Max Temperature, $R_{100}$ [°C]",
        units="[°C]",
        color="black"
    )

    utils_station.plot_return_periods_100_difference_prct_dyn_ds(
        rl_model=rl_model,
        covariate=covariate,
        # figures_path="./", 
        figures_path=figures_path, 
        x_label="2m Max Temperature, $R_{100}$ [%]",
        color="black"
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
    az_ds = az.from_zarr(str(results_dataset))
    posterior_predictive = az_ds.posterior_predictive.sel(station_id = station_id)
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
    variables = [
        "concentration",
        "scale",
        "location_slope",
        "location_intercept",
        ]

    utils_station.plot_model_params_critique(
        ds=posterior_predictive,
        variables=variables,
        # figures_path="./", 
        figures_path=figures_path, 
        
    )
    
    logger.info("Plotting Model Critique...")
    idata = az.extract(posterior_predictive, group="posterior_predictive", num_samples=10_000)


    y_pred = posterior_predictive[variable].rename("y_pred")
    y_true = y_data.rename("y_true")
    
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
    
    logger.info("Plotting Regression Plot...")
    x = posterior_predictive[covariate]
    y = y_data
    y_hat = posterior_predictive[variable]
    y_model = posterior_predictive["location_slope"] * x + posterior_predictive["location_intercept"]

    utils_station.plot_regression_posterior(
        x=x,
        y=y,
        y_hat=y_hat,
        y_model=y_model,
        # figures_path="./", 
        figures_path=figures_path, 
        figure_dpi=300,
        covariate_label="Global Mean Surface Temperature Anomaly [°C]",
        y_label="2m Max Temperature [°C]"
    )
    
    logger.info("Plotting Return Levels...")
    # select clean data
    y_clean = y_data.dropna(dim=covariate)

    # calculate return period
    y_clean = utils_station.calculate_empirical_return_level_gevd_ds(y_clean, covariate=covariate)

    # calculate model return periods
    posterior_predictive = utils_station.calculate_ds_return_periods(posterior_predictive)

    # Calculate Quantiles
    rl_model_quantiles = posterior_predictive["return_level"].quantile(q=[0.025, 0.5, 0.975], dim=["chain", "draw"])
    
    utils_station.plot_return_periods_dyn_ds(
        rl_model_quantiles=rl_model_quantiles,
        y=y_clean,
        covariate=covariate,
        # figures_path="./", 
        figures_path=figures_path, 
        y_label="2m Max Temperature, $R_a$ [°C]"
    )
    
    logger.info("Plotting 100-Year Return Level")
    # calculate model return periods
    posterior_predictive = utils_station.calculate_ds_return_periods(posterior_predictive)

    # Calculate Quantiles
    rl_model_quantiles = posterior_predictive["return_level_100"]
    
    utils_station.plot_return_periods_100_dyn_ds(
        rl_model=rl_model_quantiles,
        covariate=covariate,
        # figures_path="./", 
        figures_path=figures_path, 
        x_label="2m Max Temperature, $R_{100}$ [°C]"
    )
    
    logger.info("Plotting 100-Year Return Level Difference")
    # calculate model return periods
    posterior_predictive = utils_station.calculate_ds_return_periods(posterior_predictive)

    # Calculate Quantiles
    rl_model = posterior_predictive["return_level_100"]
    
    utils_station.plot_return_periods_100_difference_dyn_ds(
        rl_model=rl_model,
        covariate=covariate,
        # figures_path="./", 
        figures_path=figures_path, 
        x_label="2m Max Temperature, $R_{100}$ [°C]",
        units="[°C]",
        color="black"
    )

    utils_station.plot_return_periods_100_difference_prct_dyn_ds(
        rl_model=rl_model,
        covariate=covariate,
        # figures_path="./", 
        figures_path=figures_path, 
        x_label="2m Max Temperature, $R_{100}$ [%]",
        color="black"
    )
            
    return None
   
@app.command()
def evaluate_model_predictions_station(
    results_dataset: str = "",
    dataset_url: str = "",
    figures_path: str = "",
    station_id: str = "3129A",
    variable: str = "t2max",
    spatial_dim_name: str = "station_id",
    covariate: str = "gmst"
):
    
    figures_path = Path(figures_path).joinpath(f"{station_id}/predictions")
    figures_path.mkdir(parents=True, exist_ok=True)
    
    # LOAD DATA
    logger.info("Loading Original Dataset")
    with xr.open_dataset(dataset_url, engine="zarr") as f:
        ds_bm = f.load()
        
    logger.info("Loading Datasets...")
    az_ds = az.from_zarr(store=str(results_dataset))
    az_ds_station_pred = az_ds.predictions.sel(station_id = station_id)
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
    variables = [
        "concentration",
        "scale",
        "location_slope",
        "location_intercept",
        ]

    utils_station.plot_model_params_critique(
        ds=az_ds_station_pred,
        variables=variables,
        # figures_path="./", 
        figures_path=figures_path, 
        
    )
    
    logger.info("Plotting Regression Plot...")
    # calculate model return periods
    az_ds_station_pred = utils_station.calculate_ds_return_periods(az_ds_station_pred)

    az_ds_quantiles = az_ds_station_pred.quantile(q=[0.025, 0.5, 0.975], dim=["chain", "draw"]).squeeze()
    locations = az_ds_station_pred["location"].quantile(q=[0.025, 0.5, 0.975], dim=["chain", "draw"]).squeeze()
    scales = az_ds_station_pred["scale"].quantile(q=[0.025, 0.5, 0.975], dim=["chain", "draw"]).squeeze()
    return_level_100 = az_ds_station_pred["return_level_100"].quantile(q=[0.025, 0.5, 0.975], dim=["chain", "draw"]).squeeze()
    observations = ds_station[variable].squeeze()
    
    utils_station.plot_regression_prediction(
        ds_quantiles=az_ds_quantiles,
        observations=observations,
        # figures_path="./", 
        figures_path=figures_path, 
        covariate=covariate,
        figure_dpi=300,
        y_label="2m Max Temperature, $R_a$ [°C]",
        covariate_label="Global Mean Surface Temperature Anomaly [°C]",
        location_only=True
    )

    utils_station.plot_regression_prediction(
        ds_quantiles=az_ds_quantiles,
        observations=observations,
        # figures_path="./", 
        figures_path=figures_path, 
        covariate=covariate,
        figure_dpi=300,
        y_label=r"2m Max Temperature, $R_a$ [°C]",
        covariate_label="Global Mean Surface Temperature Anomaly [°C]",
        location_only=False
    )
    
    logger.info("Plotting Return Levels...")
    # select clean data
    y_clean = ds_station[variable].squeeze()

    # calculate return period
    y_clean = utils_station.calculate_empirical_return_level_gevd_ds(y_clean, covariate=covariate)

    # calculate model return periods
    az_ds_station_pred = utils_station.calculate_ds_return_periods(az_ds_station_pred)

    # Calculate Quantiles
    rl_model_quantiles = az_ds_station_pred["return_level"].quantile(q=[0.025, 0.5, 0.975], dim=["chain", "draw"])
    
    utils_station.plot_return_periods_gmst_ds(
        rl_model_quantiles=rl_model_quantiles,
        y=y_clean,
        covariate=covariate,
        # figures_path="./", 
        figures_path=figures_path, 
        y_label=r"2m Max Temperature, $R_a$ [°C]"
    )
    
    logger.info("Plotting 100-Year Return Period...")
    # calculate model return periods
    az_ds_station_pred = utils_station.calculate_ds_return_periods(az_ds_station_pred)

    # Calculate Quantiles
    rl_model_quantiles = az_ds_station_pred["return_level_100"]
    

    utils_station.plot_return_periods_100_gmst_ds(
        rl_model=rl_model_quantiles,
        covariate=covariate,
        # figures_path="./", 
        figures_path=figures_path, 
        x_label=r"2m Max Temperature, $R_{100}$ [°C]"
    )
    
    logger.info("Plotting 100-Year Return Period Difference...")
    # calculate model return periods
    az_ds_station_pred = utils_station.calculate_ds_return_periods(az_ds_station_pred)

    # Calculate Quantiles
    rl_model = az_ds_station_pred["return_level_100"]
    rl_model
        

    utils_station.plot_return_periods_100_difference_gmst_ds(
        rl_model=rl_model,
        covariate=covariate,
        # figures_path="./", 
        figures_path=figures_path, 
        x_label=r"2m Max Temperature, $\Delta R_{100}$ [°C]",
        units="[°C]",
    )
    utils_station.plot_return_periods_100_difference_prct_gmst_ds(
        rl_model=rl_model,
        covariate=covariate,
        # figures_path="./", 
        figures_path=figures_path, 
        x_label=r"2m Max Temperature, $\Delta R_{100}$ [%]",
    )

            
    return None
   

@app.command()
def evaluate_model_posterior_region(
    results_dataset: str = "",
    figures_path: str = "",
    variable: str = "t2max",
    spatial_dim_name: str = "station_id",
    covariate: str = "gmst",
    num_samples: int = 10_000,
):
    VARIABLE = variable
    COVARIATE = covariate
    
    logger.info("Loading Dataset...")
    az_ds = az.from_zarr(store=str(results_dataset))
    
    logger.info("Plotting Log-Likelihood")
    PLOT_VARIABLE = "nll"

    
    data_results = az.extract(az_ds, group="log_likelihood", num_samples=num_samples).median(dim=["sample"]).load()
    data_results = data_results.sortby(covariate)
    idata = data_results[VARIABLE].rename(PLOT_VARIABLE).sum(dim=covariate)
    
    subfigures_path = Path(figures_path).joinpath(f"{PLOT_VARIABLE}/posterior")
    subfigures_path.mkdir(parents=True, exist_ok=True)

    utils_region.plot_static_spatial_variable(
        idata,
        figures_path=subfigures_path,
        cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
        cmap="Reds_r",
        bounds = (-160, 10, -50)
    )
    
    logger.info("Plotting Posterior-Predictive Log-Likelihood")
    data_results = az.extract(az_ds, group="posterior_predictive", num_samples=num_samples).median(dim=["sample"]).load()
    data_results = data_results.sortby(covariate)
    idata = data_results["nll"].rename(PLOT_VARIABLE).sum(dim=covariate)
    
    subfigures_path = Path(figures_path).joinpath(f"{PLOT_VARIABLE}/posterior_predictive")
    subfigures_path.mkdir(parents=True, exist_ok=True)

    utils_region.plot_static_spatial_variable_redfeten(
        idata,
        figures_path=subfigures_path,
        cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
        cmap="Reds_r",
        bounds = (-160, 10, -50)
    )
    
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
    #     idata = idata.sortby("gmst")
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
    #     idata = idata.sortby("gmst")
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
    #     idata = idata.sortby("gmst")
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
    #     idata = idata.sortby("gmst")
    #     idata = idata[PLOT_VARIABLE]
    #     utils_region.plot_static_spatial_variable(
    #         idata,
    #         figures_path=subfigures_path,
    #         cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    #         cmap="Reds",
    #         bounds = (0.5, 10, 3.0)
    #     )
    
    # logger.info("Plotting Location Intercept Parameter (POSTERIOR)...")
    # PLOT_VARIABLE = "location_intercept"
    # idata = az.extract(az_ds, group="posterior", num_samples=num_samples).median(dim=["sample"]).load()
    # idata = idata.sortby("gmst")
    # idata = idata[PLOT_VARIABLE]
    # subfigures_path = Path(figures_path).joinpath(f"{PLOT_VARIABLE}/posterior")
    # subfigures_path.mkdir(parents=True, exist_ok=True)

    # utils_region.plot_static_spatial_variable(
    #     idata,
    #     figures_path=subfigures_path,
    #     cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    #     cmap="Reds",
    #     bounds = (20.0, 10, 45.0)
    # )
    # logger.info("Plotting Location Intercept Parameter (POSTERIOR Predictive)...")
    # idata = az.extract(az_ds, group="posterior_predictive", num_samples=num_samples).median(dim=["sample"]).load()
    # idata = idata.sortby("gmst")
    # idata = idata[PLOT_VARIABLE]
    # subfigures_path = Path(figures_path).joinpath(f"{PLOT_VARIABLE}/posterior_predictive")
    # subfigures_path.mkdir(parents=True, exist_ok=True)

    # utils_region.plot_static_spatial_variable(
    #     idata,
    #     figures_path=subfigures_path,
    #     cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    #     cmap="Reds",
    #     bounds = (20.0, 10, 45.0)
    # )
    
    # logger.info("Plotting Location SLOPE Parameter (POSTERIOR)...")
    # PLOT_VARIABLE = "location_slope"
    # idata = az.extract(az_ds, group="posterior", num_samples=num_samples).median(dim=["sample"]).load()
    # idata = idata.sortby("gmst")
    # idata = idata[PLOT_VARIABLE]
    # subfigures_path = Path(figures_path).joinpath(f"{PLOT_VARIABLE}/posterior")
    # subfigures_path.mkdir(parents=True, exist_ok=True)

    # utils_region.plot_static_spatial_variable(
    #     idata,
    #     figures_path=subfigures_path,
    #     cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    #     cmap="Reds",
    #     bounds = (0.0, 8, 2.5)
    # )
    # logger.info("Plotting Location SLOPE Parameter (POSTERIOR Predictive)...")
    # idata = az.extract(az_ds, group="posterior_predictive", num_samples=num_samples).median(dim=["sample"]).load()
    # idata = idata.sortby("gmst")
    # idata = idata[PLOT_VARIABLE]
    # subfigures_path = Path(figures_path).joinpath(f"{PLOT_VARIABLE}/posterior_predictive")
    # subfigures_path.mkdir(parents=True, exist_ok=True)

    # utils_region.plot_static_spatial_variable(
    #     idata,
    #     figures_path=subfigures_path,
    #     cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    #     cmap="Reds",
    #     bounds = (0.0, 8, 2.5)
    # )
        
    # logger.info("Plotting LOCATION (POSTERIOR)...")
    # PLOT_VARIABLE = "location"
    # idata = az.extract(az_ds, group="posterior", num_samples=num_samples).median(dim=["sample"]).load()
    # idata = idata.sortby("gmst")
    # idata = idata[PLOT_VARIABLE]
    # subfigures_path = Path(figures_path).joinpath(f"{PLOT_VARIABLE}/posterior")
    # subfigures_path.mkdir(parents=True, exist_ok=True)

    # utils_region.plot_dynamic_spatial_variable_postpred(
    #     idata,
    #     figures_path=subfigures_path,
    #     cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    #     cmap="Reds",
    #     bounds = (20, 10, 45),
    #     covariate = covariate,
    #     units = "[°C]"
    # )
    # subfigures_path = Path(figures_path).joinpath(f"{PLOT_VARIABLE}_difference/posterior")
    # subfigures_path.mkdir(parents=True, exist_ok=True)

    # utils_region.plot_dynamic_spatial_variable_diff_postpred(
    #     idata,
    #     figures_path=subfigures_path,
    #     cbar_label=utils_region.VARIABLE_LABELS[f"{PLOT_VARIABLE}_difference"],
    #     cmap="Reds",
    #     bounds = (1.0, 10, 2.5),
    #     covariate = covariate,
    #     units = "[°C]"
    # )
    
    # logger.info("Plotting Location (POSTERIOR Predictive)...")
    # idata = az.extract(az_ds, group="posterior_predictive", num_samples=num_samples).median(dim=["sample"]).load()
    # idata = idata.sortby("gmst")
    # idata = idata[PLOT_VARIABLE]
    # subfigures_path = Path(figures_path).joinpath(f"{PLOT_VARIABLE}/posterior_predictive")
    # subfigures_path.mkdir(parents=True, exist_ok=True)

    # utils_region.plot_dynamic_spatial_variable_postpred(
    #     idata,
    #     figures_path=subfigures_path,
    #     cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    #     cmap="Reds",
    #     bounds = (20, 10, 45),
    #     covariate = covariate,
    #     units = "[°C]"
    # )
    # subfigures_path = Path(figures_path).joinpath(f"{PLOT_VARIABLE}_difference/posterior_predictive")
    # subfigures_path.mkdir(parents=True, exist_ok=True)

    # utils_region.plot_dynamic_spatial_variable_diff_postpred(
    #     idata,
    #     figures_path=subfigures_path,
    #     cbar_label=utils_region.VARIABLE_LABELS[f"{PLOT_VARIABLE}_difference"],
    #     cmap="Reds",
    #     bounds = (1.0, 10, 2.5),
    #     covariate = covariate,
    #     units = "[°C]"
    # )
    
    # logger.info("Plotting Location (PREDICTIONS)...")
    # idata = az.extract(az_ds, group="predictions", num_samples=num_samples).median(dim=["sample"]).load()
    # idata = idata.sortby("gmst")
    # idata = idata[PLOT_VARIABLE]
    # subfigures_path = Path(figures_path).joinpath(f"{PLOT_VARIABLE}/predictions")
    # subfigures_path.mkdir(parents=True, exist_ok=True)

    # utils_region.plot_dynamic_spatial_variable_gmst_pred(
    #     idata,
    #     figures_path=subfigures_path,
    #     cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    #     cmap="Reds",
    #     bounds = (20, 10, 45),
    #     covariate = covariate
    # )
    # subfigures_path = Path(figures_path).joinpath(f"{PLOT_VARIABLE}_difference/predictions")
    # subfigures_path.mkdir(parents=True, exist_ok=True)

    # utils_region.plot_dynamic_spatial_variable_diff_gmst_pred(
    #     idata,
    #     figures_path=subfigures_path,
    #     cbar_label=utils_region.VARIABLE_LABELS[f"{PLOT_VARIABLE}_difference"],
    #     cmap="Reds",
    #     bounds = (1.0, 10, 2.5),
    #     covariate = covariate,
    #     periods = [0.0, 1.3, 2.5]
    # )

    
    # logger.info("Plotting 100-Year RETURN PERIOD (POSTERIOR)...")
    # PLOT_VARIABLE = "return_level_100"
    # idata = az.extract(az_ds, group="posterior", num_samples=num_samples)
    # idata = idata.sortby("gmst")
    # idata = utils_station.calculate_ds_return_periods_100(idata, ["sample"])
    # idata = idata[PLOT_VARIABLE].median(dim=["sample"]).load()
    
    # subfigures_path = Path(figures_path).joinpath(f"{PLOT_VARIABLE}/posterior")
    # subfigures_path.mkdir(parents=True, exist_ok=True)

    # utils_region.plot_dynamic_spatial_variable_postpred(
    #     idata,
    #     figures_path=subfigures_path,
    #     cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    #     cmap="Reds",
    #     bounds = (25, 10, 50),
    #     covariate = covariate,
    #     units = "[°C]"
    # )
    # logger.info("Plotting 100-Year RETURN PERIOD DIFFERENCE (POSTERIOR)...")
    # subfigures_path = Path(figures_path).joinpath(f"{PLOT_VARIABLE}_difference/posterior")
    # subfigures_path.mkdir(parents=True, exist_ok=True)
    # utils_region.plot_dynamic_spatial_variable_diff_postpred(
    #     idata,
    #     figures_path=subfigures_path,
    #     cbar_label=utils_region.VARIABLE_LABELS[f"{PLOT_VARIABLE}_difference"],
    #     cmap="Reds",
    #     bounds = (0.0, 10, 3.5),
    #     covariate = covariate,
    #     units = "[°C]"
    # )
    
    # logger.info("Plotting 100-Year RETURN PERIOD (POSTERIOR Predictive)...")
    # idata = az.extract(az_ds, group="posterior_predictive", num_samples=num_samples)
    # idata = idata.sortby("gmst")
    # idata = utils_station.calculate_ds_return_periods_100(idata, ["sample"])
    # idata = idata[PLOT_VARIABLE].median(dim=["sample"]).load()
    
    # subfigures_path = Path(figures_path).joinpath(f"{PLOT_VARIABLE}/posterior_predictive")
    # subfigures_path.mkdir(parents=True, exist_ok=True)

    # utils_region.plot_dynamic_spatial_variable_postpred(
    #     idata,
    #     figures_path=subfigures_path,
    #     cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    #     cmap="Reds",
    #     bounds = (25, 10, 50),
    #     covariate = covariate,
    #     units = "[°C]"
    # )
    # subfigures_path = Path(figures_path).joinpath(f"{PLOT_VARIABLE}_difference/posterior_predictive")
    # subfigures_path.mkdir(parents=True, exist_ok=True)

    # utils_region.plot_dynamic_spatial_variable_diff_postpred(
    #     idata,
    #     figures_path=subfigures_path,
    #     cbar_label=utils_region.VARIABLE_LABELS[f"{PLOT_VARIABLE}_difference"],
    #     cmap="Reds",
    #     bounds = (0.0, 10, 3.5),
    #     covariate = covariate,
    #     units = "[°C]"
    # )
    
    # logger.info("Plotting 100-Year RETURN PERIOD (PREDICTIONS)...")
    # idata = az.extract(az_ds, group="predictions", num_samples=num_samples)
    # idata = idata.sortby("gmst")
    # idata = utils_station.calculate_ds_return_periods_100(idata, ["sample"])
    # idata = idata[PLOT_VARIABLE].median(dim=["sample"]).load()
    
    # subfigures_path = Path(figures_path).joinpath(f"{PLOT_VARIABLE}/predictions")
    # subfigures_path.mkdir(parents=True, exist_ok=True)
    
    # utils_region.plot_dynamic_spatial_variable_gmst_pred(
    #     idata,
    #     figures_path=subfigures_path,
    #     cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    #     cmap="Reds",
    #     bounds = (25, 10, 50),
    #     covariate = covariate
    # )
    # subfigures_path = Path(figures_path).joinpath(f"{PLOT_VARIABLE}_difference/predictions")
    # subfigures_path.mkdir(parents=True, exist_ok=True)

    # utils_region.plot_dynamic_spatial_variable_diff_gmst_pred(
    #     idata,
    #     figures_path=subfigures_path,
    #     cbar_label=utils_region.VARIABLE_LABELS[f"{PLOT_VARIABLE}_difference"],
    #     cmap="Reds",
    #     bounds = (0.0, 10, 3.5),
    #     covariate = covariate,
    #     periods = [0.0, 1.3, 2.5]
    # )
    
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
    
    logger.info("Plotting Location INTERCEPT Kernel VARIANCE...")
    PLOT_VARIABLE = "location_intercept_kernel_variance"
    
    idata = az_ds.posterior[PLOT_VARIABLE]
    subfigures_path = Path(figures_path).joinpath(f"location_intercept/posterior")
    subfigures_path.mkdir(parents=True, exist_ok=True)

    utils_region.plot_static_global_variable(
        idata,
        figures_path=subfigures_path,
        cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    )
    
    logger.info("Plotting Location INTERCEPT Kernel SCALES...")
    PLOT_VARIABLE = "location_intercept_kernel_scale"
    
    idata = az_ds.posterior[PLOT_VARIABLE]
    subfigures_path = Path(figures_path).joinpath(f"location_intercept/posterior")
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
    
    logger.info("Plotting Location SLOPE Kernel VARIANCE...")
    PLOT_VARIABLE = "location_slope_kernel_variance"
    
    idata = az_ds.posterior[PLOT_VARIABLE]
    subfigures_path = Path(figures_path).joinpath(f"location_slope/posterior")
    subfigures_path.mkdir(parents=True, exist_ok=True)

    utils_region.plot_static_global_variable(
        idata,
        figures_path=subfigures_path,
        cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    )
    
    logger.info("Plotting Location SLOPE Kernel SCALES...")
    PLOT_VARIABLE = "location_slope_kernel_scale"
    
    idata = az_ds.posterior[PLOT_VARIABLE]
    subfigures_path = Path(figures_path).joinpath(f"location_slope/posterior")
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
    PLOT_VARIABLE = "location_intercept_mean_intercept"
    
    idata = az_ds.posterior[PLOT_VARIABLE]
    subfigures_path = Path(figures_path).joinpath(f"location_intercept/posterior")
    subfigures_path.mkdir(parents=True, exist_ok=True)

    utils_region.plot_static_global_variable(
        idata,
        figures_path=subfigures_path,
        cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    )
    
    logger.info("Plotting Location INTERCEPT Kernel SCALES...")
    PLOT_VARIABLE = "location_intercept_mean_slope"
    
    idata = az_ds.posterior[PLOT_VARIABLE]
    subfigures_path = Path(figures_path).joinpath(f"location_intercept/posterior")
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
    
    logger.info("Plotting Location SLOPE Kernel VARIANCE...")
    PLOT_VARIABLE = "location_slope_mean_intercept"
    
    idata = az_ds.posterior[PLOT_VARIABLE]
    subfigures_path = Path(figures_path).joinpath(f"location_slope/posterior")
    subfigures_path.mkdir(parents=True, exist_ok=True)

    utils_region.plot_static_global_variable(
        idata,
        figures_path=subfigures_path,
        cbar_label=utils_region.VARIABLE_LABELS[PLOT_VARIABLE],
    )
    
    logger.info("Plotting Location SLOPE Kernel SCALES...")
    PLOT_VARIABLE = "location_slope_mean_slope"
    
    idata = az_ds.posterior[PLOT_VARIABLE]
    subfigures_path = Path(figures_path).joinpath(f"location_slope/posterior")
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