import autoroot
import typer
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
from loguru import logger
from st_evt import validate_longitude, validate_latitude
from st_evt.viz import plot_scatter_ts, plot_histogram, plot_density
from metpy.units import units
import pint_xarray
from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 300  # Increase the DPI for higher quality
plt.style.use(
    "https://raw.githubusercontent.com/ClimateMatchAcademy/course-content/main/cma.mplstyle"
)

app = typer.Typer()


@app.command()
def viz_t2max_station(
    load_path: str="data/clean/",
    save_path: str="data/viz/",
    station_id: str='3129A',
    year_min: str="1960",
    year_max: str="2019"
    ):
    logger.info(f"Starting Visualization Script")
    
    DATA_URL = Path(load_path).joinpath("t2max_stations.zarr")
    
    # open all data
    with xr.open_dataset(DATA_URL, engine="zarr") as ds:
        
        ds_station = ds.sel(time=slice(year_min, year_max))
        
        # select candidate station
        ds_station = ds_station.where(ds_station.station_id == station_id, drop=True).squeeze()

        # load
        ds_station = ds_station.load()
    
    # SAVE SUBDIRECTORY
    logger.info(f"Creating Sub-Directory")
    sub_save_path = Path(save_path).joinpath(f"aemet/t2max/{station_id}/eda")
    sub_save_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Subdirectory: {sub_save_path}")
    
    # LABELS
    variable_label = "2m Max Temperature [°C]"
    
    # TIME SERIES PLOT
    logger.info("Plotting Time Series...")
    fig, ax, pts = plot_scatter_ts(ds_station["t2m_max"], x_variable="time", markersize=1.0)

    ax.set(
        xlabel="Time",
        ylabel=variable_label
    )
    fig.set(
        dpi=100,
        size_inches=(7,4)
    )
    plt.tight_layout()
    savename = sub_save_path.joinpath("ts_data.png")
    fig.savefig(savename)
    plt.close()
    
    
    # Plotting Histogram
    logger.info("Plotting Histogram...")
    fig, ax = plot_histogram(ds_station["t2m_max"])

    ax.set(
        xlabel=variable_label
    )
    fig.set(
        dpi=100,
        size_inches=(5,4)
    )
    plt.tight_layout()
    savename = sub_save_path.joinpath("hist_data.png")
    fig.savefig(savename)
    plt.close()
    
    # Plotting Histogram
    logger.info("Plotting Density...")
    fig, ax = plot_density(ds_station["t2m_max"])

    ax.set(
        xlabel=variable_label
    )
    fig.set(
        dpi=100,
        size_inches=(5,4)
    )
    plt.tight_layout()
    savename = sub_save_path.joinpath("density_data.png")
    fig.savefig(savename)
    plt.close()
    
    logger.info(f"Finished!")
    
    
@app.command()
def viz_pr_station(
    load_path: str="data/clean/",
    save_path: str="data/viz/",
    station_id: str='3129A',
    year_min: str="1960",
    year_max: str="2019"
    ):
    logger.info(f"Starting Visualization Script")
    
    DATA_URL = Path(load_path).joinpath("pr_stations.zarr")
    
    # open all data
    with xr.open_dataset(DATA_URL, engine="zarr") as ds:
        
        ds_station = ds.sel(time=slice(year_min, year_max))
        
        # select candidate station
        ds_station = ds_station.where(ds_station.station_id == station_id, drop=True).squeeze()

        # load
        ds_station = ds_station.load()
    
    # SAVE SUBDIRECTORY
    logger.info(f"Creating Sub-Directory")
    sub_save_path = Path(save_path).joinpath(f"aemet/pr/{station_id}/eda")
    sub_save_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Subdirectory: {sub_save_path}")
    
    # LABELS
    variable_label = "Cumulative Daily Precipitation [mm/day]"
    
    # TIME SERIES PLOT
    logger.info("Plotting Time Series...")
    fig, ax, pts = plot_scatter_ts(ds_station["pr"], x_variable="time", markersize=1.0)

    ax.set(
        xlabel="Time",
        ylabel=variable_label
    )
    fig.set(
        dpi=100,
        size_inches=(7,4)
    )
    plt.tight_layout()
    savename = sub_save_path.joinpath("ts_data.png")
    fig.savefig(savename)
    plt.close()
    
    
    # Plotting Histogram
    logger.info("Plotting Histogram...")
    fig, ax = plot_histogram(ds_station["pr"])

    ax.set(
        xlabel=variable_label
    )
    fig.set(
        dpi=100,
        size_inches=(5,4)
    )
    plt.tight_layout()
    savename = sub_save_path.joinpath("hist_data.png")
    fig.savefig(savename)
    plt.close()
    
    # Plotting Histogram
    logger.info("Plotting Density...")
    fig, ax = plot_density(ds_station["pr"])

    ax.set(
        xlabel=variable_label
    )
    fig.set(
        dpi=100,
        size_inches=(5,4)
    )
    plt.tight_layout()
    savename = sub_save_path.joinpath("density_data.png")
    fig.savefig(savename)
    plt.close()
    
    logger.info(f"Finished!")


if __name__ == '__main__':
    app()
