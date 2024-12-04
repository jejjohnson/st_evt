import numpy as np
import xarray as xr


def block_maxima_year(da: xr.DataArray):
    """
    Calculate the block maxima for each year in the given DataArray.
    This function groups the input DataArray by year and finds the maximum value
    for each year. The resulting DataArray will have the same attributes as the
    input DataArray.
    Parameters:
    -----------
    da : xr.DataArray
        Input DataArray with a time dimension.
    Returns:
    --------
    xr.DataArray
        DataArray containing the maximum values for each year, with the same
        attributes as the input DataArray.
    """
    # initialize extremes array
    da_extremes = np.nan * xr.ones_like(da)
    for i, igroup in da.groupby("time.year"):
        # select appropriate indice
        idx = igroup.argmax(dim="time")
        # set values
        da_extremes.loc[igroup.time[idx]] = igroup[idx]
    
    # save the attributes (e.g. units)
    da_extremes.attrs = da.attrs
    return da_extremes


def block_maxima_yearly_group(
    da: xr.DataArray,
    group: str = "time.season",
    ):
    """
    Calculate the block maxima for each year in the given DataArray.
    This function groups the input DataArray by year and finds the maximum value
    for each year. The resulting DataArray will have the same attributes as the
    input DataArray.
    Parameters:
    -----------
    da : xr.DataArray
        Input DataArray with a time dimension.
    Returns:
    --------
    xr.DataArray
        DataArray containing the maximum values for each year, with the same
        attributes as the input DataArray.
    """
    # initialize extremes array
    da_extremes = np.nan * xr.ones_like(da)
    for (iyear, igroup), group in da.groupby(['time.year', group]):
        # select appropriate indice
        idx = group.idxmax(dim="time")
        # set values
        da_extremes.loc[idx] = group.loc[idx]
    
    # save the attributes (e.g. units)
    da_extremes.attrs = da.attrs
    return da_extremes


def resample_seasonal_max(dataset: xr.Dataset) -> xr.Dataset:
    """
    Resample an xarray Dataset of daily data by the seasonal maximum.

    Parameters:
    dataset (xr.Dataset): The input xarray Dataset with daily data.

    Returns:
    xr.Dataset: The resampled xarray Dataset with seasonal maximum values.
    """
    # Define the seasons
    seasons = {
        'DJF': [12, 1, 2],  # December, January, February
        'MAM': [3, 4, 5],   # March, April, May
        'JJA': [6, 7, 8],   # June, July, August
        'SON': [9, 10, 11]  # September, October, November
    }

    # Create a season coordinate
    def assign_season(month):
        for season, months in seasons.items():
            if month in months:
                return season
        return None
    temp = [assign_season(month) for month in dataset['time.month'].values]
    dataset['season'] = ('time', temp)

    # Resample by season and take the maximum value for each season
    seasonal_max = dataset.groupby('season').max(dim='time')

    return seasonal_max