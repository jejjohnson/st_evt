from jaxtyping import Float, Array
import equinox as eqx
import numpyro
import numpyro.distributions as dist
from tensorflow_probability.substrates.jax import distributions as tfd


class StationaryGEVD(eqx.Module):
    """
    StationaryGEVD is a probabilistic model for stationary Generalized Extreme Value Distribution (GEVD).
    Attributes:
    location_prior : dist.Distribution
        Prior distribution for the location parameter.
    scale_prior : dist.Distribution
        Prior distribution for the scale parameter.
    concentration_prior : dist.Distribution
        Prior distribution for the concentration parameter.
    time_dim_name : str
        Name of the time dimension.
    variable_name : str
        Name of the variable.
    Methods:
    __call__(y: Float[Array, "T"] | None = None, num_timesteps: int = 1, num_spatial: int = 1, **kwargs) -> Array
        Defines the probabilistic model and returns the sampled or observed data array.
    dimensions() -> dict
        Returns the dimensions of the model variables.
    """
    location_prior: dist.Distribution
    scale_prior: dist.Distribution
    concentration_prior: dist.Distribution
    time_dim_name: str
    variable_name: str

    def __call__(
        self,
        *,
        y: Float[Array, "T"] | None = None,
        num_timesteps: int = 1,
        **kwargs,
    ):
        """
        Call method to define the probabilistic model.
        Parameters:
        -----------
        y : Float[Array, "T S"] | None, optional
            Observed data array with shape (T, S), where T is the number of timesteps and S is the number of spatial points.
            Default is None.
        num_timesteps : int, optional
            Number of timesteps. Default is 1.
        num_spatial : int, optional
            Number of spatial points. Default is 1.
        **kwargs : dict
            Additional keyword arguments.
        Returns:
        --------
        y : Array
            The sampled or observed data array.
        """
        if y is not None:
            assert len(y.shape) == 1
            num_timesteps = y.shape[0]
    
        plate_time = numpyro.plate(f"{self.time_dim_name}", num_timesteps, dim=-1)

    
        # LOCATION PARAMETER
        location = numpyro.sample("location", fn=self.location_prior)
        
        # SCALE PARAMETER
        scale = numpyro.sample("scale", fn=self.scale_prior)
    
        # SHAPE Parameter
        concentration = numpyro.sample("concentration", fn=self.concentration_prior)

        # create likelihood distribution
        with plate_time:
            y_dist = tfd.GeneralizedExtremeValue(loc=location, scale=scale, concentration=concentration)
            y = numpyro.sample(f"{self.variable_name}", y_dist, obs=y)
    
        return y
    
    @property
    def dimensions(self):
        return {
            f"{self.variable_name}": [f"{self.time_dim_name}"],
            # REGRESSION PARAMETERS
            f"location": [],
            f"scale": [],
            f"concentration": [],
            
        }
    @property
    def variables(self):
        return list(self.dimensions.keys())


class StationaryUnPooledGEVD(eqx.Module):
    """
    A class representing a stationary unpooled Generalized Extreme Value Distribution (GEVD) model.
    Attributes:
        location_prior (dist.Distribution): Prior distribution for the location parameter.
        scale_prior (dist.Distribution): Prior distribution for the scale parameter.
        concentration_prior (dist.Distribution): Prior distribution for the concentration (shape) parameter.
        spatial_dim_name (str): Name of the spatial dimension.
        time_dim_name (str): Name of the time dimension.
        variable_name (str): Name of the variable being modeled.
    Methods:
        __call__(y: Float[Array, "T S"] | None = None, num_timesteps: int = 1, num_spatial: int = 1, **kwargs):
            Executes the model, sampling from the priors and creating the likelihood distribution.
        dimensions:
            Returns a dictionary mapping variable names to their respective dimensions.
        variables:
            Returns a list of variable names used in the model.
    """
    location_prior: dist.Distribution
    scale_prior: dist.Distribution
    concentration_prior: dist.Distribution
    spatial_dim_name: str
    time_dim_name: str
    variable_name: str

    def __call__(
        self,
        *,
        y: Float[Array, "T S"] | None = None,
        num_timesteps: int = 1,
        num_spatial: int = 1,
        **kwargs,
    ):
        """
        Call method to define the probabilistic model.
        Parameters:
        -----------
        y : Float[Array, "T S"] | None, optional
            Observed data array with shape (T, S), where T is the number of timesteps and S is the number of spatial points.
            Default is None.
        num_timesteps : int, optional
            Number of timesteps. Default is 1.
        num_spatial : int, optional
            Number of spatial points. Default is 1.
        **kwargs : dict
            Additional keyword arguments.
        Returns:
        --------
        y : Array
            The sampled or observed data array.
        """
        if y is not None:
            assert len(y.shape) == 2
            num_timesteps, num_spatial = y.shape
    
        plate_time = numpyro.plate(f"{self.time_dim_name}", num_timesteps, dim=-2)
        plate_space = numpyro.plate(f"{self.spatial_dim_name}", num_spatial, dim=-1)
    
        with plate_space:
            # LOCATION PARAMETER
            location = numpyro.sample("location", fn=self.location_prior)
            
            # SCALE PARAMETER
            scale = numpyro.sample("scale", fn=self.scale_prior)
        
            # SHAPE Parameter
            concentration = numpyro.sample("concentration", fn=self.concentration_prior)
    
        # create likelihood distribution
        with plate_time, plate_space:
            y_dist = tfd.GeneralizedExtremeValue(loc=location, scale=scale, concentration=concentration)
            y = numpyro.sample(f"{self.variable_name}", y_dist, obs=y)
    
        return y
    
    @property
    def dimensions(self):
        return {
            f"{self.variable_name}": [f"{self.time_dim_name}", f"{self.spatial_dim_name}"],
            # REGRESSION PARAMETERS
            f"location": [f"{self.spatial_dim_name}"],
            f"scale": [f"{self.spatial_dim_name}"],
            f"concentration": [f"{self.spatial_dim_name}"],
            
        }
    
    @property
    def variables(self):
        return list(self.dimensions.keys())