import autoroot
import typer
import pandas as pd
from loguru import logger
from pathlib import Path

app = typer.Typer()


@app.command()
def clean_gmst_raw(
    load_path: str="data/raw/",
    save_path: str="data/clean/",
    ):
    raise NotImplementedError()


@app.command()
def clean_gmst_david(
    load_path: str="data/raw/",
    save_path: str="data/clean/",
    ):
    logger.info(f"Starting script...")

    logger.info(f"Sorting out paths...")
    load_path = Path(load_path)
    file_name = load_path.joinpath("gmst_david.xlsx")
    logger.debug(f"File: {file_name.resolve()}")
    
    # read the file
    raw_df = pd.read_excel(file_name)

    logger.info("Cleaning DataFrame...")

    gmst_df = raw_df.copy()

    # transform date column
    gmst_df["Year"] = gmst_df["Year"].map(lambda x: pd.to_datetime(x, format="%Y"))
    gmst_df = gmst_df.set_index("Year", drop=True)
    gmst_df.index = gmst_df.index.rename("time")

    # rename columns to remove date
    gmst_df.columns = gmst_df.columns.map(lambda x: x.split(" ")[0])
    
    # conver to xarray
    logger.info(f"Converting to xarray")
    gmst_xrds = gmst_df.to_xarray()
    
    logger.info(f"Taking 5-Year Average")
    gmst_xrds["GISS_smooth"] = gmst_xrds.GISS.rolling(time=5, center=True).mean()
    gmst_xrds["HadCRUT_smooth"] = gmst_xrds.HadCRUT.rolling(time=5, center=True).mean()
    gmst_xrds["BEST_smooth"] = gmst_xrds.BEST.rolling(time=5, center=True).mean()
    
    logger.info(f"Saving to file")
    
    full_save_path = Path(save_path).joinpath("gmst_david.zarr")
    
    logger.debug(f"Saving to {full_save_path.resolve()}")
    
    gmst_xrds.to_zarr(full_save_path)
        

if __name__ == '__main__':
    app()
