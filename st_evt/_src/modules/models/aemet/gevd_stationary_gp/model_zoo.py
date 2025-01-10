from typing import Callable, Dict, List
from jaxtyping import Array, Float
import equinox as eqx
import jax
import jax.numpy as jnp
import einx
import numpyro
import numpyro.distributions as dist
from tensorflow_probability.substrates.jax import distributions as tfd
from st_evt._src.models.scalar import ScalarModel
from st_evt._src.models.gp import SpatialGP, SpatioTemporalModel, SpatioTemporalGEVD, get_kernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from st_evt._src.features.spatial import Geodetic2Cartesian
from loguru import logger


def init_feature_transformer(ds_bm):
    
    logger.info(f"Transforming from Geodecto to Cartesian Coordinates...")
    logger.info(f"Applying Standard Scaler to Coordinates...")
    
    spatial_coords_names = ["lon", "lat", "alt"]

    # Select Coordinates
    s_coords = ds_bm[spatial_coords_names].to_dataframe()[spatial_coords_names].values

    # initialize transformer
    spatial_transformer = Pipeline(
        steps=[
            ("cartesian", Geodetic2Cartesian()),
            ("standardize", StandardScaler())
        ]
    )

    # fit transformer
    spatial_transformer.fit(s_coords)

    # apply transformer
    s_coords_transformed = spatial_transformer.transform(s_coords)
    
    ds_bm["coords_norm"] = (("station_id", "spherical"), s_coords_transformed)
    ds_bm = ds_bm.assign_coords({"spherical": spatial_coords_names})
    
    return ds_bm, spatial_transformer


def init_t2m_stationary_gp_model(

    spatial_coords: Array,
    y_values: Array,
    spatial_dim_name: str = "space",
    time_dim_name: str = "time",
    variable_name: str = "obs",
    kernel: str = "rbf",
):
    """
    Initializes a non-stationary Gaussian Process (GP) model for temperature (t2m) data.
    Parameters:
    -----------
    spatial_coords : Array
        A 2D array of spatial coordinates.
    y_values : Array
        A 1D array of observed temperature values.
    t0 : float, optional
        Initial time value, by default 0.0.
    spatial_dim_name : str, optional
        Name of the spatial dimension, by default "space".
    time_dim_name : str, optional
        Name of the time dimension, by default "time".
    variable_name : str, optional
        Name of the variable, by default "obs".
    Returns:
    --------
    SpatioTemporalGEVD
        A spatio-temporal Generalized Extreme Value Distribution (GEVD) model.
    """
    assert len(spatial_coords.shape) == 2
    num_spatial_dims = spatial_coords.shape[1]
    
    
    NAME = "location"
    NUM_SPATIAL_DIMS = 3
    SPATIAL_COORDS = jnp.asarray(spatial_coords)
    
    kernel = get_kernel(kernel)

    # Slope - Priod Distribution
    slope_dist = dist.Normal(0.0, 1.0).expand_by(sample_shape=(num_spatial_dims,))

    # Intercept - Prior Distribution
    loc_init = jnp.mean(y_values)
    scale_init = jnp.std(y_values)
    intercept_dist = dist.Normal(float(loc_init), float(scale_init))
    # Noise
    noise_dist = dist.HalfNormal(0.5) # None # 
    jitter = 1e-5
    LINK_FUNCTION = lambda x: x

    location_model = SpatialGP(
        spatial_coords=SPATIAL_COORDS,
        name=NAME,
        slope_dist=slope_dist,
        intercept_dist=intercept_dist,
        noise_dist=noise_dist,
        link_function=LINK_FUNCTION,
        num_outputs=1,
        spatial_dim_name=spatial_dim_name,
        jitter=jitter,
        kernel=kernel
    )
    
    NAME = "scale"

    # Slope - Priod Distribution
    slope_dist = dist.Normal(0.0, 1.0).expand_by(sample_shape=(num_spatial_dims,))

    # Intercept - Prior Distribution
    intercept_dist = dist.HalfNormal(5.0)

    # Noise
    noise_dist = dist.HalfNormal(0.5) # None # 
    jitter = 1e-5

    # LINK_FUNCTION = lambda x: jnp.exp(x)
    LINK_FUNCTION = lambda x: jax.nn.softplus(x)
    num_outputs = 1

    scale_model = SpatialGP(
        spatial_coords=SPATIAL_COORDS,
        name=NAME,
        slope_dist=slope_dist,
        intercept_dist=intercept_dist,
        noise_dist=noise_dist,
        link_function=LINK_FUNCTION,
        num_outputs=num_outputs,
        spatial_dim_name=spatial_dim_name,
        jitter=jitter,
        kernel=kernel
    )
    
    prior_dist = dist.TruncatedNormal(-0.3, 0.1, low=-1.0, high=-1e-5)
    NAME = "concentration"

    concentration_model = ScalarModel(
        prior_dist=prior_dist,
        name=NAME
    )
    
    return SpatioTemporalGEVD(
        location_model=location_model,
        scale_model=scale_model,
        concentration_model=concentration_model,
        variable_name=variable_name,
        time_dim_name=time_dim_name,
        spatial_dim_name=spatial_dim_name
    )      

