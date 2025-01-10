import autoroot
import typer
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
from loguru import logger
from st_evt import validate_longitude, validate_latitude
from st_evt.utils import load_spain_provinces
from metpy.units import units
import pint_xarray
from pathlib import Path

app = typer.Typer()


@app.command()
def clean_elev_eobs(
    load_path: str = "data/raw/",
    save_path: str = "data/clean/",
):
    logger.info(f"Starting script...!")

    logger.info(f"Sorting out paths...")
    load_path = Path(load_path)
    elevation = load_path.joinpath("elev_eobs_0.1.nc")
    
    logger.debug(f"ELEVATION Path: {elevation.resolve()}")
    
    ds_alt = xr.open_dataset(elevation)
    
    spain_regionmask = load_spain_provinces(overlap=False)
    
    # valudate coordinates
    ds_alt_ = validate_longitude(ds_alt)
    ds_alt_ = validate_latitude(ds_alt_)

    mask = spain_regionmask.mask_3D(ds_alt_)
    mask = mask.max(dim="region")

    ds_alt_ = ds_alt_.where(mask, drop=True)
    ds_alt_ = ds_alt_.rename({"elevation": "alt"})
    ds_alt_ = ds_alt_.assign_coords({"alt": ds_alt_.alt})
    ds_alt_["alt"].attrs["standard_name"] = "altitude"
    ds_alt_["alt"].attrs["long_name"] = "Altitude"

    logger.info(f"Validating units...")
    ds_alt_ = ds_alt_.pint.quantify(
        {
        "lon": units.degrees_east,
        "lat": units.degrees_north,
        "alt": "meters"
        }
    )
    ds_alt_ = ds_alt_.pint.dequantify()
    # rename variable

    logger.info(f"Adding country mask...")
    mask = spain_regionmask.mask_3D(ds_alt_)
    mask = mask.max(dim="region")

    # drop extras
    logger.info(f"Dropping Non Spain...")
    ds_alt_ = ds_alt_.where(mask, drop=True)


    logger.info(f"Saving data to disk...")
    
    full_save_path = Path(save_path).joinpath("elev_eobs_spain.zarr")

    logger.debug(f"Saving to {full_save_path.resolve()}")
    # assert full_save_path.parent.is_dir()

    ds_alt_.to_zarr(full_save_path, mode="w")

    logger.info(f"Finished script...")


@app.command()
def clean_pr_eobs(
    load_path: str = "data/raw/",
    save_path: str = "data/clean/",
):
    logger.info(f"Starting script...!")

    logger.info(f"Sorting out paths...")
    load_path = Path(load_path)
    elevation = load_path.joinpath("elev_eobs_0.1.nc")
    variable_dataset = load_path.joinpath("pr_mean_eobs_0.1.nc")
    
    logger.debug(f"ELEVATION Path: {elevation.resolve()}")
    logger.debug(f"PRECIPITATION Path: {variable_dataset.resolve()}")
    
    logger.info("Loading Spain Region Mask...")
    spain_regionmask = load_spain_provinces(overlap=False)
    
    ds_alt = xr.open_dataset(elevation)
    
    logger.info("Loading Precipitation Dataset...")
    ds_variable = xr.open_dataset(variable_dataset)
    
    logger.info("Interpolating Elevation onto Precipitation Dataset...")
    ds_alt = ds_alt.interp_like(ds_variable)
    ds_variable = xr.merge([ds_variable, ds_alt])
    
    # valudate coordinates
    logger.info(f"Fixing Coordinates...")
    ds_variable = validate_longitude(ds_variable)
    ds_variable = validate_latitude(ds_variable)

    logger.info(f"Sorting by Time...")
    ds_variable = ds_variable.sortby("time")

    logger.info(f"Fixing Variable Labels...")
    ds_variable = ds_variable.rename({"rr": "pr"})
    ds_variable["pr"].attrs["standard_name"] = "thickness_of_rainfall_amount"
    ds_variable["pr"].attrs["long_name"] = "Thickness Amount of Rainfall"


    ds_variable = ds_variable.rename({"elevation": "alt"})
    ds_variable = ds_variable.assign_coords({"alt": ds_variable.alt})
    ds_variable["alt"].attrs["standard_name"] = "altitude"
    ds_variable["alt"].attrs["long_name"] = "Altitude"

    logger.info(f"Validating units...")
    ds_variable = ds_variable.pint.quantify(
        {"pr": "mm", 
        "lon": units.degrees_east,
        "lat": units.degrees_north,
        "alt": "meters"
        }
    )
    ds_variable = ds_variable.pint.dequantify()
    # rename variable

    logger.info(f"Adding country mask...")
    mask = spain_regionmask.mask_3D(ds_variable)
    mask = mask.max(dim="region")

    # drop extras
    logger.info(f"Dropping Non Spain...")
    ds_variable = ds_variable.where(mask, drop=True)


    logger.info(f"Saving data to disk...")
    
    full_save_path = Path(save_path).joinpath("pr_eobs_spain.zarr")

    logger.debug(f"Saving to {full_save_path.resolve()}")
    # assert full_save_path.parent.is_dir()

    ds_variable.to_zarr(full_save_path, mode="w")

    logger.info(f"Finished script...")


@app.command()
def clean_t2max_eobs(
    load_path: str="data/raw/",
    save_path: str="data/clean/",
    ):
    logger.info(f"Starting script...!")

    logger.info(f"Sorting out paths...")
    load_path = Path(load_path)
    elevation = load_path.joinpath("elev_eobs_0.1.nc")
    variable_dataset = load_path.joinpath("tmax_mean_eobs_0.1.nc")
    
    logger.info("Loading Spain Region Mask...")
    spain_regionmask = load_spain_provinces(overlap=False)
    
    logger.debug(f"ELEVATION Path: {elevation.resolve()}")
    logger.debug(f"2M Max Temperature Path: {variable_dataset.resolve()}")
    
    ds_alt = xr.open_dataset(elevation)
    
    logger.info("Loading Precipitation Dataset...")
    ds_variable = xr.open_dataset(variable_dataset)
    
    logger.info("Interpolating Elevation onto Precipitation Dataset...")
    ds_alt = ds_alt.interp_like(ds_variable)
    ds_variable = xr.merge([ds_variable, ds_alt])
    
    # valudate coordinates
    logger.info(f"Fixing Coordinates...")
    ds_variable = validate_longitude(ds_variable)
    ds_variable = validate_latitude(ds_variable)

    logger.info(f"Sorting by Time...")
    ds_variable = ds_variable.sortby("time")

    logger.info(f"Fixing Variable Labels...")
    ds_variable = ds_variable.rename({"tx": "t2max"})
    ds_variable["t2max"].attrs["standard_name"] = "2m_max_temperature"
    ds_variable["t2max"].attrs["long_name"] = "Thickness Amount of Rainfall"

    ds_variable = ds_variable.rename({"elevation": "alt"})
    ds_variable = ds_variable.assign_coords({"alt": ds_variable.alt})
    ds_variable["alt"].attrs["standard_name"] = "altitude"
    ds_variable["alt"].attrs["long_name"] = "Altitude"

    logger.info(f"Validating units...")
    ds_variable = ds_variable.pint.quantify(
        {"t2max": units.degree_Celsius,
         "lon": units.degrees_east,
         "lat": units.degrees_north,
         "alt": units.meter
        }
    )
    ds_variable = ds_variable.pint.dequantify()
    # rename variable

    logger.info(f"Adding country mask...")
    mask = spain_regionmask.mask_3D(ds_variable)
    mask = mask.max(dim="region")

    # drop extras
    logger.info(f"Dropping Non Spain...")
    ds_variable = ds_variable.where(mask, drop=True)


    logger.info(f"Saving data to disk...")
    
    full_save_path = Path(save_path).joinpath("t2max_eobs_spain.zarr")

    logger.debug(f"Saving to {full_save_path.resolve()}")
    # assert full_save_path.parent.is_dir()

    ds_variable.to_zarr(full_save_path, mode="w")

    logger.info(f"Finished script...")


@app.command()
def clean_windspeed_eobs(
    load_path: str="data/raw/",
    save_path: str="data/clean/",
    ):
    logger.info(f"Starting script...!")

    logger.info(f"Sorting out paths...")
    load_path = Path(load_path)
    elevation = load_path.joinpath("elev_eobs_0.1.nc")
    variable_dataset = load_path.joinpath("ws_mean_eobs_0.1.nc")
    
    logger.info("Loading Spain Region Mask...")
    spain_regionmask = load_spain_provinces(overlap=False)
    
    logger.debug(f"ELEVATION Path: {elevation.resolve()}")
    logger.debug(f"Wind Speed: {variable_dataset.resolve()}")
    
    ds_alt = xr.open_dataset(elevation)
    
    logger.info("Loading Precipitation Dataset...")
    ds_variable = xr.open_dataset(variable_dataset)
    
    logger.info("Interpolating Elevation onto Wind Speed Dataset...")
    ds_alt = ds_alt.interp_like(ds_variable)
    ds_variable = xr.merge([ds_variable, ds_alt])
    
    # valudate coordinates
    logger.info(f"Fixing Coordinates...")
    ds_variable = validate_longitude(ds_variable)
    ds_variable = validate_latitude(ds_variable)

    logger.info(f"Sorting by Time...")
    ds_variable = ds_variable.sortby("time")

    logger.info(f"Fixing Variable Labels...")
    ds_variable = ds_variable.rename({"fg": "ws"})
    ds_variable["ws"].attrs["standard_name"] = "wind_speed"
    ds_variable["ws"].attrs["long_name"] = "Wind Speed"

    ds_variable = ds_variable.rename({"elevation": "alt"})
    ds_variable = ds_variable.assign_coords({"alt": ds_variable.alt})
    ds_variable["alt"].attrs["standard_name"] = "altitude"
    ds_variable["alt"].attrs["long_name"] = "Altitude"

    logger.info(f"Validating units...")
    ds_variable = ds_variable.pint.quantify(
        {"ws": units.meter_per_second,
         "lon": units.degrees_east,
         "lat": units.degrees_north,
         "alt": units.meter
        }
    )
    ds_variable = ds_variable.pint.dequantify()
    # rename variable

    logger.info(f"Adding country mask...")
    mask = spain_regionmask.mask_3D(ds_variable)
    mask = mask.max(dim="region")

    # drop extras
    logger.info(f"Dropping Non Spain...")
    ds_variable = ds_variable.where(mask, drop=True)


    logger.info(f"Saving data to disk...")
    
    full_save_path = Path(save_path).joinpath("ws_eobs_spain.zarr")

    logger.debug(f"Saving to {full_save_path.resolve()}")
    # assert full_save_path.parent.is_dir()

    ds_variable.to_zarr(full_save_path, mode="w")

    logger.info(f"Finished script...")


if __name__ == '__main__':
    app()
