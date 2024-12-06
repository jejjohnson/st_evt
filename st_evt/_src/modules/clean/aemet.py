import autoroot
import typer
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
from loguru import logger
from st_evt import validate_longitude, validate_latitude
from metpy.units import units
import pint_xarray
from pathlib import Path

app = typer.Typer()


@app.command()
def clean_pr_stations(
    load_path: str = "data/raw/",
    save_path: str = "data/clean/",
):
    logger.info(f"Starting script...!")

    logger.info(f"Sorting out paths...")
    load_path = Path(load_path)
    stations = load_path.joinpath("spain_stations.csv")
    pr_dataset = load_path.joinpath("pr.csv")
    red_feten = load_path.joinpath("red_feten.csv")
    
    logger.debug(f"Stations Path: {stations.resolve()}")
    logger.debug(f"Data Path: {pr_dataset.resolve()}")
    logger.debug(f"Stations Path: {red_feten.resolve()}")

    logger.info(f"Loading station coordinates...")
    df_coords = pd.read_csv(stations, delimiter=";", index_col=0, decimal=",")

    logger.info(f"Loading precipitation values...")
    df_all = pd.read_csv(pr_dataset, index_col=0)

    coordinates = dict(
        station_id=list(),
        station_name=list(),
        lat=list(),
        lon=list(),
        alt=list(),
        values=list()
    )

    logger.info(f"Creating xarray datastructure...")
    pbar = tqdm(df_all.columns, leave=False)
    for iname in pbar:

        try:
            ids = df_all[str(iname)]
            icoords = df_coords.loc[str(iname)]
            # extract coordinates
            coordinates["station_id"].append(str(icoords.name))
            coordinates["station_name"].append(str(icoords["name"].lower()))
            coordinates["lat"].append(np.float32(icoords["lat"]))
            coordinates["lon"].append(np.float32(icoords["lon"]))
            coordinates["alt"].append(np.float32(icoords["alt"]))
            coordinates["values"].append(np.float32(ids.values))
        except KeyError:
            pass

    ds_pr = xr.Dataset(
        {
            "pr": (("station_id", "time"), coordinates['values']),
            "lon": (("station_id"), coordinates['lon']),
            "lat": (("station_id"), coordinates['lat']),
            "alt": (("station_id"), coordinates['alt']),
            "station_name": (("station_id"), coordinates['station_name']),
        },
        coords={
            "station_id": coordinates["station_id"],
            "time": pd.to_datetime(df_all.index.values)
        }
    )

    logger.info(f"Cleaning metadata and coordinates...")

    # assign coordinates
    ds_pr = ds_pr.set_coords(["lon", "lat", "alt", "station_name"])

    # valudate coordinates
    ds_pr = validate_longitude(ds_pr)
    ds_pr = validate_latitude(ds_pr)

    ds_pr = ds_pr.sortby("time")

    ds_pr["pr"].attrs["standard_name"] = "daily_cumulative_precipitation"
    ds_pr["pr"].attrs["long_name"] = "Daily Cumulative Precipitation"

    ds_pr["alt"].attrs["standard_name"] = "altitude"
    ds_pr["alt"].attrs["long_name"] = "Altitude"

    logger.info(f"Validating All Units...")
    ds_pr = ds_pr.pint.quantify(
        {
            "pr": "mm / day",
            "lon": units.degrees_east,
            "lat": units.degrees_north,
            "alt": "meters"
        }
    )
    ds_pr = ds_pr.pint.dequantify()

    # Load the GOOD Stations
    logger.info(f"Adding Red Feten Stations...")
    red_feten_stations = pd.read_csv(red_feten)
    
    tmax_red_feten_stations = np.intersect1d(red_feten_stations.id, ds_pr.station_id)

    logger.info(f"# Red Feten Stations: {len(tmax_red_feten_stations)}...")
    
    # create mask
    red_feten_mask = ds_pr.station_id.isin(tmax_red_feten_stations).rename("red_feten").astype(np.uint8)

    # assign as coordinates
    ds_pr = ds_pr.assign_coords({"red_feten_mask": red_feten_mask})

    logger.info(f"Saving data to disk...")
    
    full_save_path = Path(save_path).joinpath("pr_stations.zarr")

    logger.debug(f"Saving to {full_save_path.resolve()}")
    # assert full_save_path.parent.is_dir()

    ds_pr.to_zarr(full_save_path, mode="w")

    logger.info(f"Finished script...")


@app.command()
def clean_t2m_stations(
    load_path: str="data/raw/",
    save_path: str="data/clean/",
    ):
    logger.info(f"Starting script...")

    logger.info(f"Sorting out paths...")
    load_path = Path(load_path)
    stations = load_path.joinpath("spain_stations.csv")
    pr_dataset = load_path.joinpath("tmax_homo.csv")
    red_feten = load_path.joinpath("red_feten.csv")
    
    logger.debug(f"Stations Path: {stations.resolve()}")
    logger.debug(f"Data Path: {pr_dataset.resolve()}")
    logger.debug(f"Stations Path: {red_feten.resolve()}")

    logger.info(f"Loading station coordinates...")
    df_coords = pd.read_csv(stations, delimiter=";", index_col=0, decimal=",")

    logger.info(f"Loading max temperature values...")
    df_all = pd.read_csv(pr_dataset, index_col=0)

    coordinates = dict(
        station_id=list(),
        station_name=list(),
        lat=list(),
        lon=list(),
        alt=list(),
        values=list()
    )

    logger.info(f"Creating xarray datastructure...")
    pbar = tqdm(df_all.columns, leave=False)
    for iname in pbar:

        

        try:
            ids = df_all[str(iname)]
            icoords = df_coords.loc[str(iname)]
            # extract coordinates
            coordinates["station_id"].append(str(icoords.name))
            coordinates["station_name"].append(str(icoords["name"].lower()))
            coordinates["lat"].append(np.float32(icoords["lat"]))
            coordinates["lon"].append(np.float32(icoords["lon"]))
            coordinates["alt"].append(np.float32(icoords["alt"]))
            coordinates["values"].append(np.float32(ids.values))
        except KeyError:
            pass

    ds_pr = xr.Dataset(
        {
            "t2m_max": (("station_id", "time"), coordinates['values']),
            "lon": (("station_id"), coordinates['lon']),
            "lat": (("station_id"), coordinates['lat']),
            "alt": (("station_id"), coordinates['alt']),
            "station_name": (("station_id"), coordinates['station_name']),
        },
        coords={
            "station_id": coordinates["station_id"],
            "time": pd.to_datetime(df_all.index.values)
        }
    )

    logger.info(f"Cleaning metadata and coordinates...")

    # assign coordinates
    ds_pr = ds_pr.set_coords(["lon", "lat", "alt", "station_name"])

    # valudate coordinates
    ds_pr = validate_longitude(ds_pr)
    ds_pr = validate_latitude(ds_pr)

    ds_pr = ds_pr.sortby("time")

    ds_pr["t2m_max"].attrs["standard_name"] = "2m_temperature_max"
    ds_pr["t2m_max"].attrs["long_name"] = "2m Temperature Max"

    ds_pr["alt"].attrs["standard_name"] = "altitude"
    ds_pr["alt"].attrs["long_name"] = "Altitude"

    logger.info(f"Validating All Units...")
    ds_pr = ds_pr.pint.quantify(
        {"t2m_max": "degC", 
        "lon": units.degrees_east,
        "lat": units.degrees_north,
        "alt": "meters"
        }
    )
    ds_pr = ds_pr.pint.dequantify()

    # Load the GOOD Stations
    logger.info(f"Adding Red Feten Stations...")
    red_feten_stations = pd.read_csv(red_feten)
    
    tmax_red_feten_stations = np.intersect1d(red_feten_stations.id, ds_pr.station_id)

    logger.info(f"# Red Feten Stations: {len(tmax_red_feten_stations)}...")
    
    # create mask
    red_feten_mask = ds_pr.station_id.isin(tmax_red_feten_stations).rename("red_feten").astype(np.uint8)

    # assign as coordinates
    ds_pr = ds_pr.assign_coords({"red_feten_mask": red_feten_mask})

    #
    logger.info(f"Saving data to disk...")
    
    full_save_path = Path(save_path).joinpath("t2max_stations.zarr")

    logger.debug(f"Saving to {full_save_path.resolve()}")
    # assert full_save_path.parent.is_dir()

    ds_pr.to_zarr(full_save_path, mode="w")

    logger.info(f"Finished script...")


if __name__ == '__main__':
    app()
