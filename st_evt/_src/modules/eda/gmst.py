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
def viz_gmst_david(
    load_path: str="data/clean/",
    save_path: str="data/viz/",
    ):
    logger.info(f"Starting Visualization Script")
    
    DATA_URL = Path(load_path).joinpath("gmst_david.zarr")
    
    # open all data
    with xr.open_dataset(DATA_URL, engine="zarr") as ds:
        
        ds_gmst = ds.load()
    
    # SAVE SUBDIRECTORY
    logger.info(f"Creating Sub-Directory")
    sub_save_path = Path(save_path).joinpath(f"gmst")
    sub_save_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Subdirectory: {sub_save_path}")
    
    # LABELS
    variable_label = "2m Max Temperature [°C]"
    
    # TIME SERIES PLOT
    logger.info("Plotting Original GMST")
    fig, ax = plt.subplots()

    ds_gmst.GWI.plot(ax=ax, linewidth=4, color="black", label="GWI")
    ds_gmst.GISS.plot(ax=ax, linewidth=4, color="tab:blue", label="GISS")
    ds_gmst.HadCRUT.plot(ax=ax, linewidth=4, color="tab:green", label="HadCRUT")
    ds_gmst.BEST.plot(ax=ax, linewidth=4, color="tab:red", label="BEST")

    ax.set(
        xlabel="Time [Years]",
        ylabel="Global Mean Surface Temperature\n[°C]",
        # ylim=[-0.5, 1.0],
        xlim=[pd.to_datetime("1750"), pd.to_datetime("2022")]
    )
    ax.grid(True, linestyle='--', linewidth='0.5', color='gray')
    ax.minorticks_on()
    plt.tight_layout()
    plt.legend()
    fig.savefig(sub_save_path.joinpath("ts_gmst.png"))
    plt.close()
    
    # TIME SERIES PLOT
    logger.info("Plotting Smoothed GMST")
    fig, ax = plt.subplots()

    ds_gmst.GWI.plot(ax=ax, linewidth=4, color="black", label="GWI")
    ds_gmst.GISS_smooth.plot(ax=ax, linewidth=4, color="tab:blue", label="GISS (5-Year Average)")
    ds_gmst.HadCRUT_smooth.plot(ax=ax, linewidth=4, color="tab:green", label="HadCRUT (5-Year Average)")
    ds_gmst.BEST_smooth.plot(ax=ax, linewidth=4, color="tab:red", label="BEST (5-Year Average)")

    ax.set(
        xlabel="Time [Years]",
        ylabel="Global Mean Surface Temperature\n[°C]",
        # ylim=[-0.5, 1.0],
        xlim=[pd.to_datetime("1750"), pd.to_datetime("2022")]
    )
    ax.grid(True, linestyle='--', linewidth='0.5', color='gray')
    ax.minorticks_on()
    plt.tight_layout()
    plt.legend()
    fig.savefig(sub_save_path.joinpath("ts_gmst_smooth.png"))
    plt.close()
    
    logger.info("Plotting Differences GMST")
    fig, ax = plt.subplots()

    # ds_gmst.GWI.sel(time=slice("1960", "2019")).plot(ax=ax, linewidth=4, zorder=4, color="black", label="GWI")


    ds_gmst.GISS_smooth.sel(time=slice("1960", "2019")).plot(ax=ax, linewidth=3, color="tab:blue", label="GISS (5-Year Average)")
    ds_gmst.HadCRUT_smooth.sel(time=slice("1960", "2019")).plot(ax=ax, linewidth=3, color="tab:green", label="HadCRUT (5-Year Average)")
    ds_gmst.BEST_smooth.sel(time=slice("1960", "2019")).plot(ax=ax, linewidth=3, color="tab:red", label="BEST (5-Year Average)")

    ds_gmst.GISS.sel(time=slice("1960", "2019")).plot(ax=ax, linewidth=2, linestyle="--", color="tab:blue", label="GISS ")
    ds_gmst.HadCRUT.sel(time=slice("1960", "2019")).plot(ax=ax, linewidth=2, linestyle="--", color="tab:green", label="HadCRUT ")
    ds_gmst.BEST.sel(time=slice("1960", "2019")).plot(ax=ax, linewidth=2, linestyle="--", color="tab:red", label="BEST ")

    ax.set(
        xlabel="Time [Years]",
        ylabel="Global Mean Surface Temperature\n[°C]",
        # ylim=[-0.5, 1.0],
        xlim=[pd.to_datetime("1960"), pd.to_datetime("2020")],
    )
    ax.grid(True, linestyle='--', linewidth='0.5', color='gray')
    ax.minorticks_on()
    plt.legend()
    plt.tight_layout()
    fig.savefig(sub_save_path.joinpath("ts_gmst_smooth_compare.png"))
    plt.show()
    
    logger.info(f"Finished!")
    
    
@app.command()
def viz_gmst(
    load_path: str="data/clean/",
    save_path: str="data/viz/",
    station_id: str='3129A',
    year_min: str="1960",
    year_max: str="2019"
    ):
    pass


if __name__ == '__main__':
    app()
