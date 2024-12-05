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
def viz_t2max_bm_year(
    load_path: str="data/ml_ready/",
    save_path: str="data/viz/",
    station_id: str='3129A',
    ):
    logger.info(f"Starting Visualization Script")
    
    DATA_URL = Path(load_path).joinpath("t2max_stations_bm_year.zarr")
    
    # open all data
    with xr.open_dataset(DATA_URL, engine="zarr") as ds:
        
        # select candidate station
        ds_station = ds.where(ds.station_id == station_id, drop=True).squeeze()

        # load
        ds_station = ds_station.load()
    
    # SAVE SUBDIRECTORY
    logger.info(f"Creating Sub-Directory")
    sub_save_path = Path(save_path).joinpath(f"aemet/t2max/{station_id}/bm_year/")
    sub_save_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Subdirectory: {sub_save_path}")
    
    # LABELS
    variable_label = "2m Max Temperature [°C]"
    
    # TIME SERIES PLOT
    logger.info("Plotting Time Series...")
    fig, ax, pts = plot_scatter_ts(ds_station["t2max_bm_year"], x_variable="gmst", markersize=5.0)

    ax.set(
        xlabel="Global Mean Surface Temperature [°C]",
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
    fig, ax = plot_histogram(ds_station["t2max_bm_year"])

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
    fig, ax = plot_density(ds_station["t2max_bm_year"])

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
def viz_t2max_bm_month(
    load_path: str="data/ml_ready/",
    save_path: str="data/viz/",
    station_id: str='3129A',
    year_min: str="1960",
    year_max: str="2019"
    ):
    raise NotImplementedError()
    
if __name__ == '__main__':
    app()