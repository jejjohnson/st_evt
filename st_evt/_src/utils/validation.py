import xarray as xr
import numpy as np
import pint_xarray
from metpy.units import units


def contains_nan(dictionary):
    """
    Check if any values in the dictionary are NaNs.

    Args:
        dictionary (dict): The dictionary to check.

    Returns:
        bool: True if any values are NaNs, False otherwise.
    """
    for value in dictionary.values():
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            return True
        elif isinstance(value, np.ndarray) and np.isnan(value).any():
            return True
    return False


def transform_360_to_180(coord: np.ndarray) -> np.ndarray:
    """
    This function converts the coordinates that are bounded from [-180, 180]
    to coordinates bounded by [0, 360].

    Args:
        coord (np.ndarray): The input array of coordinates.

    Returns:
        np.ndarray: The output array of coordinates.
    """
    return (coord + 180) % 360 - 180


def transform_180_to_360(coord: np.ndarray) -> np.ndarray:
    """
    This function converts the coordinates that are bounded from [0, 360] to coordinates bounded by [-180, 180].

    Args:
        coord (np.ndarray): The input array of coordinates.

    Returns:
        np.ndarray: The output array of coordinates.
    """
    return coord % 360


def transform_180_to_90(coord: np.ndarray) -> np.ndarray:
    """
    This function converts the coordinates that are bounded from [-180, 180]
    to coordinates bounded by [0, 360].

    Args:
        coord (np.ndarray): The input array of coordinates.

    Returns:
        np.ndarray: The output array of coordinates.
    """
    return (coord + 90) % 180 - 90


def transform_90_to_180(coord: np.ndarray) -> np.ndarray:
    """
    This function converts the coordinates that are bounded from [0, 360]
    to coordinates bounded by [-180, 180].

    Args:
        coord (np.ndarray): The input array of coordinates.

    Returns:
        np.ndarray: The output array of coordinates.
    """
    return coord % 180


def validate_altitude(ds: xr.Dataset) -> xr.Dataset:

    new_ds = ds.copy()

    new_ds = _rename_altitude(new_ds)

    ds_attrs = new_ds.alt.attrs

    new_ds["alt"] = new_ds.alt.assign_attrs(
        **{
            **ds_attrs,
            **dict(
                units="meters",
                standard_name="altitide",
                long_name="Altitude",
            ),
        }
    )

    return new_ds


def validate_longitude(ds: xr.Dataset) -> xr.Dataset:
    """Format lat and lon variables

    Set units, ranges and names

    Args:
        ds: input data

    Returns:
        formatted data
    """
    new_ds = ds.copy()

    new_ds = _rename_longitude(new_ds)

    ds_attrs = new_ds.lon.attrs

    new_ds["lon"] = transform_360_to_180(new_ds.lon)
    new_ds["lon"] = new_ds.lon.assign_attrs(
        **{
            **ds_attrs,
            **dict(
                standard_name="longitude",
                long_name="Longitude",
            ),
            
        }
    )
    new_ds["lon"] = new_ds["lon"].pint.quantify(units.degrees_east).pint.dequantify()

    return new_ds


def validate_latitude(ds: xr.Dataset) -> xr.Dataset:

    new_ds = ds.copy()

    new_ds = _rename_latitude(new_ds)

    ds_attrs = new_ds.lat.attrs

    new_ds["lat"] = transform_180_to_90(new_ds.lat)
    new_ds["lat"] = new_ds.lat.assign_attrs(
        **{
            **ds_attrs,
            **dict(
                standard_name="latitude",
                long_name="Latitude",
            ),
        }
    )
    new_ds["lat"] = new_ds["lat"].pint.quantify(units.degrees_north).pint.dequantify()
    return new_ds


def _rename_longitude(ds):
    try:
        ds = ds.rename({"longitude": "lon"})
    except:
        pass
    return ds

def _rename_latitude(ds):
    try:
        ds = ds.rename({"latitude": "lat"})
    except:
        pass
    return ds


def _rename_altitude(ds):
    variables = list(ds.variables)
    if "altitude" in variables:
        ds = ds.rename({"altitude": "alt"})
    elif "elevation" in variables:
        ds = ds.rename({"elevation": "alt"})
    return ds