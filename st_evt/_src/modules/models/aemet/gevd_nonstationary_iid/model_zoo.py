import numpy as np
from loguru import logger
from jaxtyping import Array
from st_evt._src.models.gevd import NonStationaryUnPooledGEVD, CoupledExponentialUnPooledGEVD
import numpyro.distributions as dist


def init_t2m_model(
    t_values: Array,
    y_values: Array,
    spatial_dim_name: str = "space",
    time_dim_name: str = "time",
    variable_name: str = "obs"
) -> NonStationaryUnPooledGEVD:
    
    t0 = float(t_values.min())
    
    # Intercept Parameter
    loc_init = np.mean(y_values)
    scale_init = np.std(y_values)
    logger.debug(f"Initial Location: Normal({loc_init:.2f}, {scale_init:.2f})")
    intercept_prior = dist.Normal(float(loc_init), float(scale_init))
    
    # Slope Prior
    slope_prior = dist.Normal(0.0, 1.0)
    
    # Scale Parameter is always positive
    loc_init = np.log(scale_init)
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


def init_pr_model(
    t_values: Array,
    y_values: Array,
    spatial_dim_name: str = "space",
    time_dim_name: str = "time",
    variable_name: str = "obs"
) -> CoupledExponentialUnPooledGEVD:
    
    t0 = float(t_values.min())
    
    # LOCATION PARAMETER
    loc_log_init = np.log(np.mean(y_values))
    scale_log_init = np.log(np.std(y_values))
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