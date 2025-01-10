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

T2MAX_MEAN_DATA_NC_LINK = "https://knmi-ecad-assets-prd.s3.amazonaws.com/ensembles/data/Grid_0.1deg_reg_ensemble/tx_ens_mean_0.1deg_reg_v30.0e.nc"
T2MAX_DATA_NAME = "tmax_mean_eobs_0.1.nc"

PR_MEAN_DATA_NC_LINK = "https://knmi-ecad-assets-prd.s3.amazonaws.com/ensembles/data/Grid_0.1deg_reg_ensemble/rr_ens_mean_0.1deg_reg_v30.0e.nc"
PR_DATA_NAME = "pr_mean_eobs_0.1.nc"

WS_MEAN_DATA_NC_LINK = "https://knmi-ecad-assets-prd.s3.amazonaws.com/ensembles/data/Grid_0.1deg_reg_ensemble/fg_ens_mean_0.1deg_reg_v30.0e.nc"
WS_DATA_NAME = "ws_mean_eobs_0.1.nc"

ELEV_DATA_NC_LINK = "https://knmi-ecad-assets-prd.s3.amazonaws.com/ensembles/data/Grid_0.1deg_reg_ensemble/elev_ens_0.1deg_reg_v30.0e.nc"
ELEV_DATA_NAME = "elev_eobs_0.1.nc"


def download_file(url: str, save_path: str):
    """
    Downloads a file from the given URL and saves it to the specified path.
    
    Args:
        url (str): The URL of the file to download.
        save_path (str): The path where the downloaded file will be saved.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check if the request was successful
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)  # Create directories if they don't exist
    
    total_size = int(response.headers.get('content-length', 0))
    chunk_size = 8192
    num_bars = total_size // chunk_size
    
    with open(save_path, 'wb') as file:
        with tqdm(enumerate(response.iter_content(chunk_size=chunk_size))) as pbar:
            for i, chunk in pbar:
                file.write(chunk)
                pbar.set_description(f"Downloaded {i+1}/{num_bars} chunks ({(i+1)*chunk_size}/{total_size} bytes)")
    
    print(f"\nDownloaded {url} to {save_path}")


@app.command()
def download_eobs_data(filepath: str="data/raw/"):
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
    logger.info(f"Starting Script...!")
    
    logger.info(f"DOWNLOADING E-OBS Data - 2m Max Temperature...")
    
    logger.debug(f"URL: {T2MAX_MEAN_DATA_NC_LINK}")
    full_path = Path(filepath).joinpath(T2MAX_DATA_NAME)
    logger.debug(f"FilePath: {full_path}")
    
    download_file(T2MAX_MEAN_DATA_NC_LINK, full_path)
    logger.info(f"Downloaded {T2MAX_DATA_NAME}")
    
    logger.info(f"DOWNLOADING E-OBS Data - Precipitation...")
    
    logger.debug(f"URL: {PR_MEAN_DATA_NC_LINK}")
    full_path = Path(filepath).joinpath(PR_DATA_NAME)
    logger.debug(f"FilePath: {full_path}")
    
    download_file(PR_MEAN_DATA_NC_LINK, full_path)
    logger.info(f"Downloaded {PR_DATA_NAME}")
    
    
    logger.info(f"DOWNLOADING E-OBS Data - Wind Speed...")
    
    logger.debug(f"URL: {WS_MEAN_DATA_NC_LINK}")
    full_path = Path(filepath).joinpath(WS_DATA_NAME)
    logger.debug(f"FilePath: {full_path}")
    
    download_file(WS_MEAN_DATA_NC_LINK, full_path)
    logger.info(f"Downloaded {WS_DATA_NAME}")
    
    logger.info(f"DOWNLOADING E-OBS Data - Elevation...")
    
    logger.debug(f"URL: {ELEV_DATA_NC_LINK}")
    full_path = Path(filepath).joinpath(ELEV_DATA_NAME)
    logger.debug(f"FilePath: {full_path}")
    
    download_file(ELEV_DATA_NC_LINK, full_path)
    logger.info(f"Downloaded {ELEV_DATA_NAME}")
    
    logger.info(f"Finished script...!")


if __name__ == '__main__':
    app()
