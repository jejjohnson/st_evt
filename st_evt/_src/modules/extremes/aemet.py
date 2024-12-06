import autoroot
from typing import List
import typer
from typing_extensions import Annotated
from loguru import logger
from pathlib import Path
import xarray as xr
from st_evt.extremes import block_maxima_year, block_maxima_yearly_group



app = typer.Typer()


@app.command()
def aemet_t2m_bm_year(
    data_path: str="",
    cov_path: str="",
    save_path: str="",
    months: str = '6, 7, 8',
    year_min: str="1960",
    year_max: str="2019",
    output_name: str | None = None
    ):
    logger.info("Starting Script...")
    logger.info("Loading Data")
    # LOAD DATA
    data_url = Path(data_path).joinpath("t2max_stations.zarr")
    ds = xr.open_dataset(data_url, engine="zarr")
    ds = ds.transpose("time", "station_id")
    logger.info("Selecting Station")
    logger.info("Adding GMST Covariate")
    
    VARIABLE = "t2m_max"
    VARIABLE_BM = "t2max_bm_year"
    
    # SELECT CANDIDATE STATION
    logger.info("Selecting Station...")

    
    # SELECT MONTHS
    logger.info("Selecting Months...")
    months = months.split(",")
    months = list(map(int, months))
    logger.debug(f"Months: {months}")
    ds = ds.sel(time=ds["time"].dt.month.isin(months))
    
    ds = ds.sel(time=slice(year_min, year_max))
    
    ds[VARIABLE_BM] = block_maxima_year(ds[VARIABLE])
    
    # calculate thresholds (for later)
    logger.info(f"Calculating Quantiles")
    quantiles = [0.90, 0.95, 0.98, 0.99, 0.995]
    ds["threshold"] = ds[VARIABLE].quantile(q=quantiles)
    
    # RESAMPLE (to remove NANS)
    logger.info(f"Removing NANs")
    ds_bm = ds[[VARIABLE_BM]].resample(time="1YE").max()
    
    # ADD COVARIATE
    logger.info(f"Adding Covariate")
    covariate_path = Path(cov_path).joinpath("gmst_david.zarr")
    ds_gmst = xr.open_dataset(covariate_path, engine="zarr").load()
    ds_gmst = ds_gmst.interp_like(ds_bm)
    ds_bm["gmst"] = ds_gmst.GISS_smooth
    
    # ADD THRESHDOLD
    ds_bm["threshold"] = ds["threshold"]

    # TICKERY
    logger.info("Cleaning File...")
    ds_bm = ds_bm.swap_dims({"time": "gmst"})
    ds_bm["gmst"].attrs["long_name"] = "Global Mean Surface Temperature Anomaly"
    ds_bm["gmst"].attrs["short_name"] = "Global Mean Surface Temperature"
    ds_bm["gmst"].attrs["units"] = "[°C]"
    
    # SAVE DATA
    logger.info("Saving data...")
    if output_name is None:
        output_name = "t2max_stations_bm_year"
    full_path = Path(save_path).joinpath(f"{output_name}.zarr")
    logger.debug(f"Save Path: {full_path}")
    
    ds_bm.to_zarr(full_path)


@app.command()
def aemet_t2m_bm_month(
    load_path: str="",
    save_path: str="",
    months: List[int] = [6, 7, 8]
    ):
    raise NotImplementedError("Not Implemented...")

if __name__ == '__main__':
    app()