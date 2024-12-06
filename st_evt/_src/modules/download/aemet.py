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

TMAX_DATA_CSV_LINK = "1S1SF4eUbKrFuhLiOq7j2KPhWldgmaOdJ"
T2MAX_DATA_NAME = "tmax_homo.csv"

PR_DATA_CSV_LINK = "1_8ELgo3cWvkZaARMY1eAFffDHTAD1z02"
PR_DATA_NAME = "pr.csv"

STATIONS_CSV_ID = "11aZt-UuJ4lsh8ikyUxoUq6t6iJ0eyCk5"
STATIONS_CSV_NAME = "spain_stations.csv"

RED_FETEN_CSV_ID = "14O8XSkBRHCHBdxilTiTbeXIlqI85CIq4"
RED_FETEN_CSV_NAME = "red_feten.csv"


@app.command()
def download_aemet_data(filepath: str="data/raw/"):
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
    
    logger.info(f"DOWNLOADING AEMET Data - 2m Max Temperature...")
    
    full_path = Path(filepath).joinpath(T2MAX_DATA_NAME)
    logger.debug(f"FilePath: {full_path.resolve()}")
    
    url = f"https://drive.google.com/uc?id={TMAX_DATA_CSV_LINK}"
    gdown.download(url, str(full_path))
    logger.info(f"Downloaded {T2MAX_DATA_NAME} from {url}")
    
    logger.info(f"Downloading AEMET Data - Precipitation...")
    
    full_path = Path(filepath).joinpath(PR_DATA_NAME)
    logger.debug(f"FilePath: {full_path.resolve()}")
    
    url = f"https://drive.google.com/uc?id={PR_DATA_CSV_LINK}"
    gdown.download(url, str(full_path))
    logger.info(f"Downloaded {PR_DATA_NAME} from {url}")
    
    logger.info(f"DOWNLOADING AEMET Data - Station Metadata...")
    
    full_path = Path(filepath).joinpath(STATIONS_CSV_NAME)
    logger.debug(f"FilePath: {full_path.resolve()}")
    
    url = f"https://drive.google.com/uc?id={STATIONS_CSV_ID}"
    gdown.download(url, str(full_path))
    logger.info(f"Downloaded {STATIONS_CSV_NAME} from {url}")
    
    logger.info(f"DOWNLOADING AEMET Data - Red Feten Stations...")
    
    full_path = Path(filepath).joinpath(RED_FETEN_CSV_NAME)
    logger.debug(f"FilePath: {full_path.resolve()}")
    
    url = f"https://drive.google.com/uc?id={RED_FETEN_CSV_ID}"
    gdown.download(url, str(full_path))
    logger.info(f"Downloaded {RED_FETEN_CSV_NAME} from {url}")
    
    logger.info(f"Finished script...!")



if __name__ == '__main__':
    app()
