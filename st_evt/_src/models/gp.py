from typing import Callable, Dict, List
from jaxtyping import Array, Float
import equinox as eqx
import jax
import jax.numpy as jnp
import einx
from tinygp import kernels, GaussianProcess
import numpyro
import numpyro.distributions as dist
from tensorflow_probability.substrates.jax import distributions as tfd
from st_evt._src.models.scalar import ScalarModel


def get_kernel(kernel: str = "expsquared"):
    
    if kernel == "rbf":
        return kernels.ExpSquared()
    elif kernel == "matern32":
        return kernels.Matern32()
    elif kernel == "matern52":
        return kernels.Matern52()
    elif kernel == "rq":
        return kernels.RationalQuadratic()
    elif kernel == "expsinesq":
        return kernels.ExpSineSquared()
    elif kernel == "linear":
        return kernels.DotProduct()
    else:
        raise ValueError(f"Unrecognized kernel")


class SpatialGP(eqx.Module):
    spatial_coords: Array
    kernel: str 
    name: str
    slope_dist: dist.Distribution | None
    intercept_dist: dist.Distribution | None
    noise_dist: dist.Distribution | None
    link_function: Callable
    num_outputs: int
    spatial_dim_name: str
    jitter: float = eqx.field(static=True,)
    ard: bool = True

    @property
    def num_spatial_dims(self):
        return self.spatial_coords.shape[1]

    def __call__(self, *args, s: Float[Array, "S Ds"] | None = None, train: bool = False, noise: bool = True, **kwargs) -> Float[Array, "S"]:
        
        # MEAN FUNCTION
        if self.slope_dist is not None:
            weights: Float[Array, "Ds"] = numpyro.sample(f"{self.name}_mean_slope", fn=self.slope_dist,)
            # weights: Float[Array, "Ds"] = numpyro.sample(f"{self.name}_mean_slope", fn=self.slope_dist, sample_shape=(self.num_spatial_dims,))
            if s is not None and not train:
                mean_weights: Float[Array, "S"] = einx.dot("Ds, S Ds -> S", weights, s)
            else:
                mean_weights: Float[Array, "S"] = einx.dot("Ds, S Ds -> S", weights, self.spatial_coords)
                
        else:
            mean_weights = 0.0
        if self.intercept_dist is not None:
            intercept: Float[Array, ""] = numpyro.sample(f"{self.name}_mean_intercept", fn=self.intercept_dist,)
        else:
            intercept = 0.0
    

        mean_weights = numpyro.deterministic(f"{self.name}_mean", mean_weights)

        # KERNEL PARAMETERS
        if self.ard:
            with numpyro.plate("spatial_dims", size=self.num_spatial_dims):
                kscale = numpyro.sample(f"{self.name}_kernel_scale", fn=dist.HalfNormal(2.0))
        else:
            kscale = numpyro.sample(f"{self.name}_kernel_scale", fn=dist.HalfNormal(2.0))
        # kscale = numpyro.sample(f"{self.name}_kernel_scale", fn=dist.HalfNormal(2.0), sample_shape=(self.num_spatial_dims,))
        kvar = numpyro.sample(f"{self.name}_kernel_variance", fn=dist.HalfNormal(2.0))
        if self.noise_dist is not None and noise:
            knoise = numpyro.sample(f"{self.name}_kernel_noise", fn=self.noise_dist) + self.jitter
        else:
            knoise = self.jitter
        kernel = kvar.squeeze() * self.kernel
        
        # LATENT FUNCTION
        gp = GaussianProcess(kernel, self.spatial_coords/kscale, diag=knoise if noise else self.jitter, mean=0.0)
        f: Float[Array, "N"] = numpyro.sample(f"{self.name}_spatial_field_cond", gp.numpyro_dist())
    
        if s is not None and not train:
            _, gp = gp.condition(y=f, X_test=s/kscale, diag=knoise if noise else self.jitter,)
            f: Float[Array, "M"] = numpyro.sample(f"{self.name}_spatial_field", gp.numpyro_dist())
    
        f = mean_weights + intercept + f
    
        f = numpyro.deterministic(f"{self.name}", self.link_function(f))
        
        return f
    
    @property
    def dimensions(self):
        return {
            f"{self.name}": [f"{self.spatial_dim_name}"],
            f"{self.name}_cond": [f"{self.spatial_dim_name}_cond"],
            # REGRESSION PARAMETERS
            f"{self.name}_mean_intercept": [],
            f"{self.name}_mean_slope": ["spatial_dims"] if self.slope_dist is not None else [],
            f"{self.name}_mean": [f"{self.spatial_dim_name}"] if self.slope_dist is not None else [],
            # GP PARAMETERS
            f"{self.name}_kernel_variance": [],
            f"{self.name}_kernel_noise": [],
            f"{self.name}_kernel_scale": ["spatial_dims"] if self.ard else [],
            f"{self.name}_spatial_field": [f"{self.spatial_dim_name}"],
            f"{self.name}_spatial_field_cond": [f"{self.spatial_dim_name}_cond"],
            
        }


    @property
    def variables(self):
        return list(self.dimensions.keys())


class SpatioTemporalModel(eqx.Module):
    spatial_slope_model: SpatialGP
    spatial_intercept_model: SpatialGP
    name: str
    time_dim_name: str
    spatial_dim_name: str
    t0: float
    link_function: Callable

    def __call__(
        self,
        t: Float[Array, "T"],
        *args,
        s: Float[Array, "S Ds"] | None = None,
        train: bool = False,
        **kwargs
    ) -> Float[Array, "T S"]:

        # GP FUNCTION - WEIGHTS
        f_intercept = self.spatial_intercept_model(s=s, train=train, **kwargs)
        
        # GP FUNCTION - SLOPE
        f_slope = self.spatial_slope_model(s=s, train=train, **kwargs)

        # MULTIPLICATION
        f_spacetime = einx.dot("S, T -> T S", f_slope, t - self.t0)

        # ADDITION
        f = einx.add("T S, S -> T S", f_spacetime, f_intercept)
    
        f = numpyro.deterministic(f"{self.name}", self.link_function(f))
        
        return f
    
    @property
    def dimensions(self):

        dims = {}
        for idict in [self.spatial_slope_model.dimensions, self.spatial_intercept_model.dimensions]:
            dims.update(idict)
            
        dims[f"{self.name}"] = [f"{self.time_dim_name}", f"{self.spatial_dim_name}"]
        return dims
    
    @property
    def variables(self):
        return list(self.dimensions.keys())



class SpatioTemporalGEVD(eqx.Module):
    location_model: eqx.Module
    scale_model: eqx.Module
    concentration_model: eqx.Module
    variable_name: str
    time_dim_name: str
    spatial_dim_name: str
    
    def __call__(
        self,
        t: Float[Array, "T"],
        s: Float[Array, "S Ds"],
        *args,
        y: Float[Array, "T S"] | None = None,
        train: bool = True,
        **kwargs
        ):
        
        num_timesteps = t.shape[0]
        num_spatial = s.shape[0]
        
        plate_time = numpyro.plate("time", num_timesteps, dim=-2)
        plate_space = numpyro.plate("space", num_spatial, dim=-1)
        
        location = self.location_model(t=t, s=s, train=train, **kwargs)
        scale = self.scale_model(t=t, s=s, train=train, **kwargs)
        concentration = self.concentration_model(t=t, s=s, train=train, **kwargs)
        
        
        # create likelihood distribution
        with plate_time, plate_space:
            y_dist = tfd.GeneralizedExtremeValue(loc=location, scale=scale, concentration=concentration)
            y = numpyro.sample(f"{self.variable_name}", y_dist, obs=y)

        return y
    
    @property
    def dimensions(self):

        dims = {}
        for idict in [self.location_model.dimensions, self.scale_model.dimensions, self.concentration_model.dimensions]:
            dims.update(idict)

        dims[f"{self.variable_name}"] = [f"{self.time_dim_name}", f"{self.spatial_dim_name}"]
        return dims
    
    @property
    def variables(self):
        return list(self.dimensions.keys())


