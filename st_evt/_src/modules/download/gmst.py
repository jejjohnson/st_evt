import autoroot
import typer
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
from loguru import logger
import pint_xarray
import requests
from pathlib import Path
import gdown


app = typer.Typer()

GMST_DAVID = "1auvOTxbLHHbpHeioR8WiylDOBAYBahpm"
GMST_DAVID_NAME = "gmst_david.xlsx"


@app.command()
def download_gmst_data(filepath: str="data/raw/"):
    """
    Downloads various datasets from Google Drive and saves them to the specified filepath.
    Args:
        filepath (str): The directory where the downloaded files will be saved. Defaults to "data/raw/".
    Downloads:
        - T2MAX data
        - PR data
        - Stations data
        - RED FETEN data
    The function logs the progress of the downloads, including the start and end of the script, 
    and the file paths where the data is saved.
    """
    logger.info(f"Starting Script")
    
    logger.info(f"DOWNLOADING GMST DAVID...")
    
    full_path = Path(filepath).joinpath(GMST_DAVID_NAME)
    logger.debug(f"FilePath: {full_path}")
    
    url = f"https://drive.google.com/uc?id={GMST_DAVID}"
    gdown.download(url, str(full_path))
    logger.info(f"Downloaded {GMST_DAVID_NAME} from {url}")
    
    logger.info(f"Finished script...")



if __name__ == '__main__':
    app()
