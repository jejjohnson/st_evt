from typing import Tuple
from jaxtyping import ArrayLike
import pyproj
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# Function to convert Cartesian coordinates to Geodetic
def cartesian_to_geodetic_3d(x: ArrayLike, y: ArrayLike, z: ArrayLike) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Convert Cartesian coordinates (ECEF) to geodetic coordinates (latitude, longitude, altitude).

    Parameters:
    x (ArrayLike): X coordinate in meters.
    y (ArrayLike): Y coordinate in meters.
    z (ArrayLike): Z coordinate in meters.
    Returns:
    tuple: A tuple containing:
        - lon (ArrayLike): Longitude in degrees.
        - lat (ArrayLike): Latitude in degrees.
        - alt (ArrayLike): Altitude in meters.
    """

    # Define the coordinate systems
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    # Perform the coordinate transformation
    lon, lat, alt = pyproj.transform(ecef, lla, x, y, z, radians=False)
    return lon, lat, alt


# Function to convert Geodetic coordinates to Cartesian
def geodetic_to_cartesian_3d(lon: ArrayLike, lat: ArrayLike, alt: ArrayLike) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Converts geodetic coordinates (longitude, latitude, altitude) to Cartesian coordinates (x, y, z).

    Parameters:
    lon (ArrayLike): Array of longitudes in degrees.
    lat (ArrayLike): Array of latitudes in degrees.
    alt (ArrayLike): Array of altitudes in meters.

    Returns:
    Tuple[ArrayLike, ArrayLike, ArrayLike]: Arrays of Cartesian coordinates (x, y, z) in meters.
    """
    # Define the coordinate systems
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    # Perform the coordinate transformation
    x, y, z = pyproj.transform(lla, ecef, lon, lat, alt, radians=False)
    return x, y, z


class Geodetic2Cartesian(BaseEstimator, TransformerMixin):
    """
    Transformer that converts geodetic coordinates (longitude, latitude, altitude) to Cartesian coordinates (x, y, z) and vice versa.
    Parameters
    ----------
    radius : float, default=6371.010
        The radius of the Earth in kilometers. Default is the mean radius of the Earth.
    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer does not require fitting, so it returns itself.
    transform(X, y=None) -> np.ndarray
        Transform geodetic coordinates to Cartesian coordinates.
    inverse_transform(X, y=None) -> np.ndarray
        Transform Cartesian coordinates back to geodetic coordinates.
    """
    def __init__(
        self,
        radius: float = 6371.010
        ):
        self.radius = radius

    def fit(self, X: np.ndarray, y=None):
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transforms geodetic coordinates (longitude, latitude, altitude) to Cartesian coordinates (x, y, z).

        Parameters:
        X (np.ndarray): Input array of shape (n_samples, 3) containing geodetic coordinates.
        y (optional): Ignored, present for API consistency by convention.

        Returns:
        np.ndarray: Transformed array of shape (n_samples, 3) containing Cartesian coordinates.
        """
        lon, lat, alt = np.split(X, 3, axis=-1)

        x, y, z = geodetic_to_cartesian_3d(
            lon=lon,
            lat=lat,
            alt=alt,
        )
        X = np.concatenate([x, y, z], axis=-1)

        return X

    def inverse_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Inverse transforms the given Cartesian coordinates (X) into geodetic coordinates (longitude, latitude, altitude).
        Parameters
        ----------
        X : np.ndarray
            A numpy array of shape (n_samples, 3) containing Cartesian coordinates (x, y, z).
        y : None, optional
            Ignored. This parameter exists only for compatibility with scikit-learn's TransformerMixin.
        Returns
        -------
        np.ndarray
            A numpy array of shape (n_samples, 3) containing geodetic coordinates (longitude, latitude, altitude).
        """
        
        x, y, z = np.split(X, 3, axis=-1)

        lon, lat, alt = cartesian_to_geodetic_3d(
            x=x,
            y=y,
            z=z,
        )
        
        X = np.concatenate([lon, lat, alt], axis=-1)
        
        return X


class Cartesian2Geodetic(Geodetic2Cartesian):
    def __init__(self, radius: float = 6371.010):
        super().__init__(radius=radius)

    def transform(self, X: pd.DataFrame(), y=None) -> pd.DataFrame:

        X = super().inverse_transform(X=X, y=y)

        return X

    def inverse_transform(self, X: pd.DataFrame(), y=None) -> pd.DataFrame:

        X = super().transform(X=X, y=y)

        return X