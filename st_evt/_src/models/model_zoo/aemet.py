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
from st_evt._src.models.gevd import NonStationaryUnPooledGEVD, CoupledExponentialUnPooledGEVD
from st_evt._src.models.gp import SpatialGP, SpatioTemporalModel, SpatioTemporalGEVD
from loguru import logger


def init_t2m_nonstationary_iid_model(

    t_values: Array,
    y_values: Array,
    spatial_dim_name: str = "space",
    time_dim_name: str = "time",
    variable_name: str = "obs"
) -> NonStationaryUnPooledGEVD:
    """
    Initialize a non-stationary IID model for temperature (t2m) using a Generalized Extreme Value Distribution (GEVD).
    Parameters:
    -----------
    t_values : Array
        Array of time values.
    y_values : Array
        Array of observed values.
    spatial_dim_name : str, optional
        Name of the spatial dimension. Defaults to "space".
    time_dim_name : str, optional
        Name of the time dimension. Defaults to "time".
    variable_name: str, optional
        Name of the variable. Defaults to "obs".
    Returns:
    --------
    NonStationaryUnPooledGEVD
        An instance of the NonStationaryUnPooledGEVD model initialized with the specified priors.
    """
    t0 = float(t_values.min())
    
    # Intercept Parameter
    loc_init = jnp.mean(y_values)
    scale_init = jnp.std(y_values)
    logger.debug(f"Initial Location: Normal({loc_init:.2f}, {scale_init:.2f})")
    intercept_prior = dist.Normal(float(loc_init), float(scale_init))
    
    # Slope Prior
    slope_prior = dist.Normal(0.0, 1.0)
    
    # Scale Parameter is always positive
    loc_init = jnp.log(scale_init)
    logger.debug(f"Initial Scale: LogNormal({loc_init:.2f}, 0.5)")
    scale_prior = dist.LogNormal(loc_init, 0.5)
    
    # TEMPERATURE has a negative shape
    concentration_prior = dist.TruncatedNormal(-0.3, 0.1, low=-1.0, high=-1e-5)

    # initialize model
    return NonStationaryUnPooledGEVD(
        slope_prior=slope_prior,
        intercept_prior=intercept_prior,
        scale_prior=scale_prior,
        concentration_prior=concentration_prior,
        spatial_dim_name=spatial_dim_name,
        time_dim_name=time_dim_name,
        variable_name=variable_name,
        t0=t0,   
    )


def init_pr_nonstationary_iid_model(
    t_values: Array,
    y_values: Array,
    spatial_dim_name: str = "space",
    time_dim_name: str = "time",
    variable_name: str = "obs"
) -> CoupledExponentialUnPooledGEVD:
    """
    Initializes a non-stationary IID model for precipitation using a Coupled Exponential UnPooled Generalized Extreme Value Distribution (GEVD).
    
    Parameters:
    -----------
    t_values: Array
        Array of time values.
    y_values: Array
        Array of observed values.
    spatial_dim_name: str, optional
        Name of the spatial dimension. Defaults to "space".
    time_dim_name: str, optional
        Name of the time dimension. Defaults to "time".
    variable_name: str, optional
        Name of the variable. Defaults to "obs".
        
    Returns:
    --------
    CoupledExponentialUnPooledGEVD: An initialized GEVD model with specified priors.
    """
    t0 = float(t_values.min())
    
    # LOCATION PARAMETER
    loc_log_init = jnp.log(jnp.mean(y_values))
    scale_log_init = jnp.log(jnp.std(y_values))
    logger.debug(f"Initial Location: Normal({loc_log_init:.2f}, {scale_log_init:.2f})")
    loc_intercept_prior = dist.LogNormal(float(loc_log_init), float(scale_log_init))
    slope_prior = dist.Normal(0.0, 1.0)
    
    # Scale Parameter is always positive
    loc_init = scale_log_init
    logger.debug(f"Initial Scale: LogNormal({loc_init:.2f}, 0.5)")
    scale_intercept_prior = dist.LogNormal(loc_init, 0.5)
    
    # TEMPERATURE has a negative shape
    concentration_prior = dist.TruncatedNormal(0.3, 0.1, low=1e-5, high=1.0)

    # initialize model
    return CoupledExponentialUnPooledGEVD(
        loc_intercept_prior=loc_intercept_prior,
        loc_slope_prior=slope_prior,
        scale_intercept_prior=scale_intercept_prior,
        scale_slope_prior=slope_prior,
        concentration_prior=concentration_prior,
        spatial_dim_name=spatial_dim_name,
        time_dim_name=time_dim_name,
        variable_name=variable_name,
        t0=t0,
    )


def init_t2m_nonstationary_gp_model(

    spatial_coords: Array,
    y_values: Array,
    t0: float = 0.0,
    spatial_dim_name: str = "space",
    time_dim_name: str = "time",
    variable_name: str = "obs"
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
    
    
    NAME = "location_intercept"
    NUM_SPATIAL_DIMS = 3
    SPATIAL_COORDS = jnp.asarray(spatial_coords)

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
    num_outputs = 1

    spatial_intercept_model = SpatialGP(
        spatial_coords=SPATIAL_COORDS,
        name=NAME,
        slope_dist=slope_dist,
        intercept_dist=intercept_dist,
        noise_dist=noise_dist,
        link_function=LINK_FUNCTION,
        num_outputs=num_outputs,
        spatial_dim_name=spatial_dim_name,
        jitter=jitter
    )
    
    NAME = "location_slope"
    SPATIAL_COORDS = jnp.asarray(spatial_coords)

    # Slope - Priod Distribution
    slope_dist = dist.Normal(0.0, 1.0).expand_by(sample_shape=(num_spatial_dims,))

    # Intercept - Prior Distribution
    intercept_dist = dist.Normal(0.0, 1.0)

    # Noise
    noise_dist = dist.HalfNormal(0.5) # None # 
    jitter = 1e-5

    LINK_FUNCTION = lambda x: x
    num_outputs = 1


    spatial_slope_model = SpatialGP(
        spatial_coords=SPATIAL_COORDS,
        name=NAME,
        slope_dist=slope_dist,
        intercept_dist=intercept_dist,
        noise_dist=noise_dist,
        link_function=LINK_FUNCTION,
        num_outputs=num_outputs,
        spatial_dim_name=spatial_dim_name,
        jitter=jitter
    )
    
    NAME = "location"
    LINK_FUNCTION = lambda x: x

    location_model = SpatioTemporalModel(
        spatial_intercept_model=spatial_intercept_model,
        spatial_slope_model=spatial_slope_model,
        name=NAME,
        time_dim_name=time_dim_name,
        spatial_dim_name=spatial_dim_name,
        t0=t0,
        link_function=LINK_FUNCTION,
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
        jitter=jitter
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