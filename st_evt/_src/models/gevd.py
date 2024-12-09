from jaxtyping import Float, Array
import equinox as eqx
import jax.numpy as jnp
import einx
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
    
    
class NonStationaryGEVD(eqx.Module):
    """
    NonStationaryGEVD is a probabilistic model for stationary Generalized Extreme Value Distribution (GEVD).
    Attributes:
    slope_prior : dist.Distribution
        Prior distribution for the location slope parameter.
    intercept_prior : dist.Distribution
        Prior distribution for the location intercept parameter.
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
    slope_prior: dist.Distribution
    intercept_prior: dist.Distribution
    scale_prior: dist.Distribution
    concentration_prior: dist.Distribution
    time_dim_name: str
    variable_name: str
    t0: float

    def __call__(
        self,
        t: Float[Array, "T"],
        *,
        y: Float[Array, "T"] | None = None,
        **kwargs,
    ):
        """
        Call method to define the probabilistic model.
        Parameters:
        -----------
        t : Float[Array, "T"] | None, optional
            time steps of shape (T,), where T is the number of timesteps.
        y : Float[Array, "T"] | None, optional
            Observed data array with shape (T,), where T is the number of timesteps.
            Default is None.
        **kwargs : dict
            Additional keyword arguments.
        Returns:
        --------
        y : Array
            The sampled or observed data array.
        """
        num_timesteps = t.shape[0]
        if y is not None:
            assert len(y.shape) == 1
            assert t.shape[0] == y.shape[0]
    
        plate_time = numpyro.plate(f"{self.time_dim_name}", num_timesteps, dim=-1)

    
        # LOCATION Regression PARAMETERs
        location_slope = numpyro.sample("location_slope", fn=self.slope_prior)
        location_intercept = numpyro.sample("location_intercept", fn=self.intercept_prior)
        
        # LOCATION PARAMETER
        location = numpyro.deterministic("location", location_slope * (t - self.t0) + location_intercept)
        
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
            # OBSERVATION VARIABLE
            f"{self.variable_name}": [f"{self.time_dim_name}"],
            # REGRESSION PARAMETERS
            f"location_slope": [],
            f"location_intercept": [],
            # LATENT VARIABLES
            f"location": [f"{self.time_dim_name}"],
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
    
    
class NonStationaryUnPooledGEVD(eqx.Module):
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
    slope_prior: dist.Distribution
    intercept_prior: dist.Distribution
    scale_prior: dist.Distribution
    concentration_prior: dist.Distribution
    spatial_dim_name: str
    time_dim_name: str
    variable_name: str
    t0: float

    def __call__(
        self,
        t: Float[Array, "T"],
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
        t : Float[Array, "T"] | None, optional
            time steps of shape (T,), where T is the number of timesteps.
        y : Float[Array, "T S"] | None, optional
            Observed data array with shape (T, S), where T is the number of timesteps and S is the number of spatial points.
            Default is None.
        num_spatial : int, optional
            Number of spatial points. Default is 1.
        **kwargs : dict
            Additional keyword arguments.
        Returns:
        --------
        y : Array
            The sampled or observed data array.
        """
        num_timesteps = t.shape[0]
        if y is not None:
            assert len(y.shape) == 2
            assert t.shape[0] == y.shape[0]
            num_spatial = y.shape[1]
        else:
            num_spatial = 1
    
        plate_time = numpyro.plate(f"{self.time_dim_name}", num_timesteps, dim=-2)
        plate_space = numpyro.plate(f"{self.spatial_dim_name}", num_spatial, dim=-1)
    
        with plate_space:
            # LOCATION Regression PARAMETERs
            location_slope = numpyro.sample("location_slope", fn=self.slope_prior)
            location_intercept = numpyro.sample("location_intercept", fn=self.intercept_prior)
            
            # LOCATION PARAMETER
            location = einx.multiply("S, T -> T S", location_slope, (t - self.t0))
            location = einx.add("T S, S -> T S", location, location_intercept)
            location = numpyro.deterministic("location", location)
            
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
            f"location": [f"{self.time_dim_name}", f"{self.spatial_dim_name}"],
            f"location_slope": [f"{self.spatial_dim_name}"],
            f"location_intercept": [f"{self.spatial_dim_name}"],
            f"scale": [f"{self.spatial_dim_name}"],
            f"concentration": [f"{self.spatial_dim_name}"],
            
        }
    
    @property
    def variables(self):
        return list(self.dimensions.keys())
    
    
class CoupledExponentialUnPooledGEVD(eqx.Module):
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
    loc_slope_prior: dist.Distribution
    loc_intercept_prior: dist.Distribution
    scale_slope_prior: dist.Distribution
    scale_intercept_prior: dist.Distribution
    concentration_prior: dist.Distribution
    spatial_dim_name: str
    time_dim_name: str
    variable_name: str
    t0: float

    def __call__(
        self,
        t: Float[Array, "T"],
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
        t : Float[Array, "T"] | None, optional
            time steps of shape (T,), where T is the number of timesteps.
        y : Float[Array, "T S"] | None, optional
            Observed data array with shape (T, S), where T is the number of timesteps and S is the number of spatial points.
            Default is None.
        num_spatial : int, optional
            Number of spatial points. Default is 1.
        **kwargs : dict
            Additional keyword arguments.
        Returns:
        --------
        y : Array
            The sampled or observed data array.
        """
        num_timesteps = t.shape[0]
        if y is not None:
            assert len(y.shape) == 2
            assert t.shape[0] == y.shape[0]
            num_spatial = y.shape[1]
        else:
            num_spatial = 1
    
        plate_time = numpyro.plate(f"{self.time_dim_name}", num_timesteps, dim=-2)
        plate_space = numpyro.plate(f"{self.spatial_dim_name}", num_spatial, dim=-1)
    
        with plate_space:
            # LOCATION Regression PARAMETERs
            location_slope = numpyro.sample("location_slope", fn=self.loc_slope_prior)
            location_intercept = numpyro.sample("location_intercept", fn=self.loc_intercept_prior)
            
            location = einx.multiply("S, T -> T S", location_slope, (t - self.t0))
            location = einx.divide("T S, S -> T S", location, location_intercept)
            location = einx.multiply("T S, S -> T S", jnp.exp(location), location_intercept)
            
            # LOCATION PARAMETER
            location = numpyro.deterministic("location",  location)
            
            # LOCATION Regression PARAMETERs
            scale_slope = numpyro.sample("scale_slope", fn=self.scale_slope_prior)
            scale_intercept = numpyro.sample("scale_intercept", fn=self.scale_intercept_prior)
            
            scale = einx.multiply("S, T -> T S", scale_slope, (t - self.t0))
            scale = einx.divide("T S, S -> T S", scale, scale_intercept)
            scale = einx.multiply("T S, S -> T S", jnp.exp(scale), location_intercept)
            
            # SCALE PARAMETER
            scale = numpyro.deterministic("scale",  scale)
            
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
            "location": [f"{self.time_dim_name}", f"{self.spatial_dim_name}"],
            "location_slope": [f"{self.spatial_dim_name}"],
            "location_intercept": [f"{self.spatial_dim_name}"],
            "scale": [f"{self.time_dim_name}", f"{self.spatial_dim_name}"],
            "scale_slope": [f"{self.spatial_dim_name}"],
            "scale_intercept": [f"{self.spatial_dim_name}"],
            "concentration": [f"{self.spatial_dim_name}"],
            
        }
    
    @property
    def variables(self):
        return list(self.dimensions.keys())