from typing import List, Dict
from typing import Callable
import equinox as eqx
from jaxtyping import Float, Array
from jax.scipy.linalg import cholesky, solve_triangular
import jax.numpy as jnp
import numpyro.distributions as dist
import numpyro
import jax
import einx
from tinygp import kernels, transforms
from tensorflow_probability.substrates.jax import distributions as tfd


class LinearModel(eqx.Module):
    name: str
    batch_dim_name: str
    input_dim_name: str
    output_dim_name: str
    slope: Array
    intercept: Array
    
    @classmethod
    def init_from_numpyro_priors(
        cls,
        name: str,
        num_input_dims: int = 1,
        num_output_dims: int = 1,
        slope_prior: dist.Distribution = dist.Normal(0.0, 1.0),
        intercept_prior: dist.Distribution = dist.Normal(0.0, 1.0),
        batch_dim_name: str = "num_points",
        input_dim_name: str = "input_dims",
        output_dim_name: str = "output_dims"
        ):
        
        slope = numpyro.sample(f"{name}_slope", slope_prior, sample_shape=(num_input_dims, num_output_dims,))
        intercept = numpyro.sample(f"{name}_intercept", intercept_prior, sample_shape=(num_output_dims,))
        
        return cls(
            name=name,
            batch_dim_name=batch_dim_name,
            input_dim_name=input_dim_name,
            output_dim_name=output_dim_name,
            slope=slope, 
            intercept=intercept
            )        

    def __call__(self, x: Float[Array, "... Dx"], *args, **kwargs) -> Float[Array, "... Dz"]:
        mu = einx.dot("... Dx, Dx Dz -> ... Dz", x, self.slope)
        mu = einx.add("... Dz, Dz -> ... Dz", mu, self.intercept)
        mu = numpyro.deterministic(f"{self.name}", mu)
        return mu

    @property
    def num_input_dims(self):
        return self.slope.shape[-2]
    
    @property
    def num_output_dims(self):
        return self.intercept.shape[-1]
    
    @property
    def dimensions(self):
        return {
            f"{self.name}_intercept": [f"{self.output_dim_name}"],
            f"{self.name}_slope": [f"{self.input_dim_name}", f"{self.output_dim_name}"],
            f"{self.name}": [f"{self.batch_dim_name}", f"{self.output_dim_name}"],
        }

    @property
    def variables(self):
        return list(self.dimensions.keys())
    
    
class ExponentialModel(eqx.Module):
    name: str
    batch_dim_name: str
    input_dim_name: str
    output_dim_name: str
    slope: Array
    intercept: Array
    
    @classmethod
    def init_from_numpyro_priors(
        cls,
        name: str,
        num_input_dims: int = 1,
        num_output_dims: int = 1,
        slope_prior: dist.Distribution = dist.LogNormal(1.0, 0.5),
        intercept_prior: dist.Distribution = dist.LogNormal(1.0, 0.5),
        batch_dim_name: str = "num_points",
        input_dim_name: str = "input_dims",
        output_dim_name: str = "output_dims"
        ):
        
        slope = numpyro.sample(f"{name}_slope", slope_prior, sample_shape=(num_input_dims, num_output_dims,))
        intercept = numpyro.sample(f"{name}_intercept", intercept_prior, sample_shape=(num_output_dims,))
        
        
        return cls(
            name=name,
            batch_dim_name=batch_dim_name,
            input_dim_name=input_dim_name,
            output_dim_name=output_dim_name,
            slope=slope, 
            intercept=intercept
            )        

    def __call__(self, x: Float[Array, "... Dx"], *args, **kwargs) -> Float[Array, "... Dz"]:
        mu = einx.dot("... Dx, Dx Dz -> ... Dz", x, self.slope)
        mu = einx.multiply("... Dz, Dz -> ... Dz", jnp.exp(mu), self.intercept)
        mu = numpyro.deterministic(f"{self.name}", mu)
        return mu

    @property
    def num_input_dims(self):
        return self.slope.shape[-2]
    
    @property
    def num_output_dims(self):
        return self.intercept.shape[-1]
    
    @property
    def dimensions(self):
        return {
            f"{self.name}_intercept": [f"{self.output_dim_name}"],
            f"{self.name}_slope": [f"{self.input_dim_name}", f"{self.output_dim_name}"],     
            f"{self.name}": [f"{self.batch_dim_name}", f"{self.output_dim_name}"],     
        }

    @property
    def variables(self):
        return list(self.dimensions.keys())
    

class CoupledExponentialModel(eqx.Module):
    name: str
    batch_dim_name: str
    input_dim_name: str
    output_dim_name: str
    slope: Array
    intercept: Array
    
    @classmethod
    def init_from_numpyro_priors(
        cls,
        name: str,
        num_input_dims: int = 1,
        slope_prior: dist.Distribution = dist.LogNormal(1.0, 0.5),
        intercept_prior: dist.Distribution = dist.LogNormal(1.0, 0.5),
        batch_dim_name: str = "num_points",
        input_dim_name: str = "input_dims",
        output_dim_name: str = "output_dims"
        ):
        
        slope = numpyro.sample(f"{name}_slope", slope_prior, sample_shape=(num_input_dims, 2,))
        intercept = numpyro.sample(f"{name}_intercept", intercept_prior)
        
        
        return cls(
            name=name,
            batch_dim_name=batch_dim_name,
            input_dim_name=input_dim_name,
            output_dim_name=output_dim_name,
            slope=slope, 
            intercept=intercept
            )        

    def __call__(self, x: Float[Array, "... Dx"], *args, **kwargs) -> Float[Array, "... 2"]:
        mu = einx.dot("... Dx, Dx Dz -> ... Dz", x, self.slope)
        # print(mu.shape, self.intercept.shape)
        mu = mu / self.intercept # einx.divide("... Dz, Dz -> ... Dz", mu, self.intercept)
        mu = jnp.exp(mu) * self.intercept # einx.multiply("... Dz, Dz -> ... Dz", , self.intercept)
        mu = numpyro.deterministic(f"{self.name}", mu)
        return mu

    @property
    def num_input_dims(self):
        return self.slope.shape[-2]
    
    @property
    def num_output_dims(self):
        return self.intercept.shape[-1]
    
    @property
    def dimensions(self):
        return {
            f"{self.name}_intercept": [],
            f"{self.name}_slope": [f"{self.input_dim_name}", f"{self.output_dim_name}"],     
            f"{self.name}": [f"{self.batch_dim_name}", f"{self.output_dim_name}"],     
        }

    @property
    def variables(self):
        return list(self.dimensions.keys())
    

class ARDRBFKernel(eqx.Module):
    
    name: str
    input_dim_name: str
    num_input_dims: int
    scale: Float[Array, "D"]
    variance: Float[Array, ""]
    
    @classmethod
    def init_from_numpyro_priors(
        cls,
        name: str,
        num_input_dims: int = 1,
        scale_prior: dist.Distribution = dist.HalfNormal(2.0),
        variance_prior: dist.Distribution = dist.HalfNormal(2.0),
        input_dim_name: str = "input_dims",
        ):
        
        variance = numpyro.sample(f"{name}_variance", variance_prior, )
        scale = numpyro.sample(f"{name}_scale", scale_prior, sample_shape=(num_input_dims,))
        
        return cls(
            name=name,
            input_dim_name=input_dim_name,
            num_input_dims=num_input_dims,
            scale=scale, 
            variance=variance
            )


    def __call__(self, x, y) -> Float[Array, "N N"]:
        kernel = self.variance * transforms.Linear(
            1/self.scale, kernels.ExpSquared()
        )
        return kernel(x, y)

    @property
    def dimensions(self):
        return {
            f"{self.name}_variance": [],
            f"{self.name}_scale": [f"{self.input_dim_name}"],
        }

    @property
    def variables(self):
        return list(self.dimensions.keys())
    
class ARDRBFCholeskyKernel(eqx.Module):
    
    name: str
    input_dim_name: str
    num_input_dims: int
    diagonal_scale: Float[Array, "D"]
    off_diagonal_scale: Float[Array, "D"]
    variance: Float[Array, ""]
    
    @classmethod
    def init_from_numpyro_priors(
        cls,
        name: str,
        num_input_dims: int = 1,
        scale_prior: dist.Distribution = dist.HalfNormal(2.0),
        variance_prior: dist.Distribution = dist.HalfNormal(2.0),
        input_dim_name: str = "input_dims",
        ):
        
        variance = numpyro.sample(f"{name}_variance", variance_prior, )
        diagonal_scale = numpyro.sample(f"{name}_scale_diagonal", scale_prior, sample_shape=(num_input_dims,))
        off_diagonal_scale = numpyro.sample(f"{name}_scale_offdiagonal", scale_prior, sample_shape=(num_input_dims,))
        
        return cls(
            name=name,
            input_dim_name=input_dim_name,
            num_input_dims=num_input_dims,
            diagonal_scale=diagonal_scale, 
            off_diagonal_scale=off_diagonal_scale,
            variance=variance
            )


    def __call__(self, x, y) -> Float[Array, "N N"]:
        kernel = self.variance * transforms.Cholesky.from_parameters(
            diagonal=self.diagonal_scale,
            off_diagonal=self.off_diagonal_scale,
            kernel=kernels.ExpSquared()
        )
        return kernel(x, y)

    @property
    def dimensions(self):
        return {
            f"{self.name}_variance": [],
            f"{self.name}_scale_diagonal": [f"{self.input_dim_name}"],
            f"{self.name}_scale_offdiagonal": [f"{self.input_dim_name}"],
        }

    @property
    def variables(self):
        return list(self.dimensions.keys())
    

class VariationalGP(eqx.Module):
    x: Float[Array, "N D"]
    sample_dim_name: str
    num_latent_dims: int
    latent_dim_name: str
    jitter: float = eqx.field(static=True)
    mean: eqx.Module | None
    kernel: eqx.Module
    name: str
    link_function: Callable
    noise_level: dist.Distribution | None

    def __init__(
        self,
        x,
        mean,
        kernel,
        num_latent_dims: int,
        name: str = "model",
        sample_dim_name = "sample_dims",
        latent_dim_name = "latent_dims",
        jitter = 1e-5,
        noise_level: dist.Distribution = dist.HalfNormal(0.5),
        link_function = lambda x: x
    ):
        self.x = x
        self.sample_dim_name = sample_dim_name
        self.num_latent_dims = num_latent_dims
        self.latent_dim_name = latent_dim_name
        self.jitter = jitter
        self.mean = mean
        self.kernel = kernel
        self.name = name
        self.link_function = link_function
        self.noise_level = noise_level
    
    @property
    def num_points(self):
        return self.x.shape[0]
    
    def __call__(
        self,
        x: Float[Array, "M D"],
        *args,
        y: Float[Array, "M D"] | None = None,
        train: bool = True,
        include_noise: bool = False,
        **kwargs,
        ) -> Float[Array, "M D"]:
        
        if train is True:
            # Mean Function, μ: [N, Dz]
            if self.mean is not None:
                mean = self.mean(self.x)
            # Kernel Matrix, K_ff: [N, N]
            K_ff = self.kernel(self.x, self.x)
            K_ff += self.jitter * jnp.eye(self.num_points)
            if self.noise_level is not None:
                noise = numpyro.sample(f"{self.name}_noise", self.noise_level)
                K_ff += noise * jnp.eye(self.num_points)
            # Zero Mean, μ_0: [Dz, N]
            zero_loc = jnp.zeros((self.num_latent_dims,) + (self.num_points,))
            # Cholesky, L_ff: [N, N]
            L_ff = cholesky(K_ff, lower=True)
            # GP Distribution - MVN
            gp_dist = dist.MultivariateNormal(loc=zero_loc, scale_tril=L_ff).to_event(zero_loc.ndim - 1)
            f = numpyro.sample(f"{self.name}_field_cond", gp_dist)
        else:
            K_ff = self.kernel(self.x, self.x)
            K_ff += self.jitter * jnp.eye(self.num_points)
            if self.noise_level is not None and include_noise:
                noise = numpyro.sample(f"{self.name}_noise", self.noise_level)
                K_ff += noise * jnp.eye(self.num_points)
            zero_loc = jnp.zeros((self.num_latent_dims,) + (self.num_points,))
            L_ff = cholesky(K_ff, lower=True)
            f = numpyro.sample(f"{self.name}_field_cond", dist.MultivariateNormal(loc=zero_loc, scale_tril=L_ff).to_event(zero_loc.ndim - 1))

            if self.mean is not None:
                mean = self.mean(x)
            K_fs = self.kernel(self.x, x)
            K_ss = self.kernel(x, x)
            
            pack = einx.rearrange("Dz N, N M -> N (Dz + M)", f, K_fs)
            L_ff_inv = solve_triangular(L_ff, pack, lower=True)
            v_2D, W = einx.rearrange("N (Dz + M) -> N Dz, M N", L_ff_inv, Dz=self.num_latent_dims)
            loc = einx.dot("M N, N Dz -> Dz M", W, v_2D)
            Q_ss = einx.dot("M N, O N -> M O", W, W)
            cov = (K_ss - Q_ss) + self.jitter * jnp.eye(K_ss.shape[0])
            f = numpyro.sample(f"{self.name}_field", dist.MultivariateNormal(loc=loc, covariance_matrix=cov).to_event(loc.ndim - 1))
            
        f = einx.rearrange("Dz N -> N Dz", f)
        if self.mean is not None:
            f = einx.add("N Dz, N Dz -> N Dz", f, mean)
        f = numpyro.deterministic(f"{self.name}", self.link_function(f))

        return f
    
    @property
    def dimensions(self):
        
        dims= {
            f"{self.name}_field_cond": [f"{self.latent_dim_name}", f"{self.sample_dim_name}", ],
            f"{self.name}_field": [f"{self.latent_dim_name}", f"{self.sample_dim_name}", ],
            f"{self.name}": [f"{self.sample_dim_name}", f"{self.latent_dim_name}"],
        }
        if self.noise_level is not None:
            dims[f"{self.name}_noise"] = []
        for idict in [self.mean.dimensions, self.kernel.dimensions]:
            dims.update(idict)
        return dims

    @property
    def variables(self):
        return list(self.dimensions.keys())
    

class TemporalVariationalGP(VariationalGP):
    
    def __init__(
        self,
        t: Float[Array, "N D"],
        mean: eqx.Module,
        kernel: eqx.Module,
        num_latent_dims: int = 1,
        latent_dim_name: str = "latent_dim",
        sample_dim_name: str = "sample_dim",
        jitter: float = 1e-5,
        name: str = "gevd",
        noise_level: dist.Distribution = dist.HalfNormal(0.5),
        link_function: Callable = lambda x: x
        ):
        
        super().__init__(
            x=t,
            num_latent_dims=num_latent_dims,
            sample_dim_name=sample_dim_name,
            latent_dim_name=latent_dim_name,
            jitter=jitter,
            mean=mean,
            kernel=kernel,
            name=name,
            noise_level=noise_level,
            link_function=link_function,
            )
    
    def __call__(
        self,
        t: Float[Array, "T D"],
        *args,
        y: Float[Array, "T D"] | None = None,
        train: bool = True,
        include_noise: bool = False,
        **kwargs,
        ) -> Float[Array, "M D"]:
        return super().__call__(x=t, y=y, train=train, include_noise=include_noise, **kwargs)
    

class SpatialVariationalGP(VariationalGP):
    
    def __init__(
        self,
        s: Float[Array, "S D"],
        mean: eqx.Module,
        kernel: eqx.Module,
        num_latent_dims: int = 1,
        latent_dim_name: str = "latent_dim",
        sample_dim_name: str = "sample_dim",
        jitter: float = 1e-5,
        name: str = "gevd",
        noise_level: dist.Distribution = dist.HalfNormal(0.5),
        link_function: Callable = lambda x: x
        ):
        
        super().__init__(
            x=s,
            num_latent_dims=num_latent_dims,
            sample_dim_name=sample_dim_name,
            latent_dim_name=latent_dim_name,
            jitter=jitter,
            mean=mean,
            kernel=kernel,
            name=name,
            noise_level=noise_level,
            link_function=link_function
            )
    
    def __call__(
        self,
        s: Float[Array, "M Ds"],
        *args,
        y: Float[Array, "S Dy"] | None = None,
        train: bool = True,
        include_noise: bool = False,
        **kwargs,
        ) -> Float[Array, "M Dy"]:
        return super().__call__(x=s, y=y, train=train, include_noise=include_noise, **kwargs)

    # def prior_dist(self, *args, **kwargs):
    #     # Kernel Matrix, K_ff: [N, N]
    #     K_ff = self.kernel(self.x, self.x)
    #     K_ff += self.jitter * jnp.eye(self.num_points)
    #     # Zero Mean, μ_0: [Dz, N]
    #     zero_loc = jnp.zeros((self.num_output_dims,) + (self.num_points,))
    #     # Cholesky, L_ff: [N, N]
    #     L_ff = cholesky(K_ff, lower=True)
    #     # Identity, K_I_ff: [N, N]
    #     identity = jnp.eye(self.num_points)
    #     # GP Distribution - MVN
    #     gp_dist = dist.MultivariateNormal(loc=zero_loc, covariance_matrix=identity).to_event(zero_loc.ndim - 1)
        
    #     return L_ff, gp_dist

    # def posterior_predictive_dist(self, x: Float[Array, "M D"], *args, **kwargs):
    #     # Cross-Covariance Matrix, K_fs: [N, M]
    #     K_fs = self.kernel(self.x, x)
    #     # Test-Covariance Matrix, K_ss: [M, M]
    #     K_ss = self.kernel(x, x)
    #     # (inv(L_ff) @ Kf*).T, K_ss: [M, N]
    #     W = solve_triangular(L_ff, K_fs, lower=True).T
    #     pass


class TemporalGEVDLikelihood(eqx.Module):
    location: eqx.Module
    scale: eqx.Module
    concentration: eqx.Module

    def __call__(
        self,
        *args,
        t: Float[Array, "T Dt"],
        x: Float[Array, "T Dx"] | None = None,
        y: Float[Array, "T Dy"] | None = None,
        **kwargs
        ):
        
        assert len(t.shape) == 2
        num_timesteps = t.shape[0]
        
        if x is not None:
            assert len(x.shape) == 2
            assert x.shape[0] == t.shape[0]
            
        if y is not None:
            assert len(y.shape) == 2
            assert y.shape[0] == t.shape[0]
        
        if t is not None and y is not None:
            assert t.shape[0] == y.shape[0]
        
        plate_time = numpyro.plate("timesteps", size=num_timesteps, dim=-2)
        
        # get models
        location = self.location(t=t, x=x, y=y, **kwargs)
        scale = self.scale(t=t, x=x, y=y, **kwargs)
        concentration = self.concentration(t=t, x=x, y=y, **kwargs)
        obs_dist = tfd.GeneralizedExtremeValue(location, scale, concentration)
        
        with plate_time:
            
            # get nans if exists
            if y is not None:
                
                # get nans
                mask = jnp.isfinite(y)
                
                # mask distribution
                obs_dist = tfd.Masked(obs_dist, mask)

            # sample
            y = numpyro.sample("obs", obs_dist, obs=y)
        
        return y
    @property
    def dimensions(self):

        dims = {}
        for idict in [self.location.dimensions, self.scale.dimensions, self.concentration.dimensions]:
            dims.update(idict)

        dims["obs"] = ["time"]
        return dims



    @property
    def variables(self):
        return list(self.dimensions.keys())
    

class LocGEVDLikelihood(eqx.Module):
    location: eqx.Module
    scale_prior: dist.Distribution
    concentration_prior: dist.Distribution
    sample_dim_name: str

    def __call__(
        self,
        x: Float[Array, "N Dx"],
        *args,
        y: Float[Array, "N Dy"] | None = None,
        **kwargs
        ):
        
        assert len(x.shape) == 2
        num_samples = x.shape[0]
        
        if y is not None:
            assert len(y.shape) == 2
            assert y.shape[0] == x.shape[0]
        
        if x is not None and y is not None:
            assert x.shape[0] == y.shape[0]
        
        plate_time = numpyro.plate("timesteps", size=num_samples, dim=-2)
        
        # get models
        location = self.location(x=x, y=y, **kwargs)
        scale = numpyro.sample("scale", self.scale_prior)
        concentration = numpyro.sample("concentration", self.concentration_prior)
        obs_dist = tfd.GeneralizedExtremeValue(location, scale, concentration)
        
        with plate_time:
            
            # get nans if exists
            if y is not None:
                
                # get nans
                mask = jnp.isfinite(y)
                
                # mask distribution
                obs_dist = tfd.Masked(obs_dist, mask)

            # sample
            y = numpyro.sample("obs", obs_dist, obs=y)
        
        return y
    @property
    def dimensions(self):

        dims = {}
        for idict in [self.location.dimensions,]:
            dims.update(idict)
        dims["concentration"] = []
        dims["scale"] = []
        dims["obs"] = [f"{self.sample_dim_name}", f"variable"]
        return dims



    @property
    def variables(self):
        return list(self.dimensions.keys())
    

class GWR(eqx.Module):
    name: str
    intercept_model: eqx.Module
    slope_model: eqx.Module
    conditional_model: eqx.Module
    num_obs_dims: int
    num_cov_dims: int
    cond_variable_name: str
    spatial_dim_name: str
    
    
    def __call__(
        self,
        s: Float[Array, "S Ds"],
        x: Float[Array, "N Dx"],
        *args,
        y: Float[Array, "N S Dy"] | None = None,
        train: bool = True,
        **kwargs,
        ) -> Float[Array, "N S Dy"]:
        
        # apply GP Model - SLOPE
        intercept: Float[Array, "S Dy"] = self.intercept_model(s=s, y=y, train=train, **kwargs)
        
        # apply GP Model - Intercept
        slope: Float[Array, "S (Dy Dx)"] = self.slope_model(s=s, y=y, train=train, **kwargs)
        
        # Rearrange parameters
        slope = einx.rearrange("S (Dy Dx) -> S Dy Dx", slope, Dy=self.num_obs_dims, Dx=self.num_cov_dims)
        
        # Conditional Model
        x_pred = self.conditional_model(x=x, y=y, train=train, **kwargs)
        
        # multiplication
        z: Float[Array, "N Dy"] = einx.dot("S Dy Dx, N Dx -> N S Dy", slope, x_pred)
        # addition
        z: Float[Array, "N Dy"] = einx.add("N S Dy, S Dy -> N S Dy", z, intercept)
        
        z = numpyro.deterministic(f"{self.name}", z)
        
        return z
    
    @property
    def dimensions(self):

        dims = {}
        for idict in [self.slope_model.dimensions, self.intercept_model.dimensions]:
            dims.update(idict)
            
        dims[f"{self.name}"] = [f"{self.cond_variable_name}", f"{self.spatial_dim_name}", f"variable"]
        return dims



    @property
    def variables(self):
        return list(self.dimensions.keys())


class CoupledExpGWR(eqx.Module):
    name: str
    location_intercept_model: eqx.Module
    location_slope_model: eqx.Module
    scale_intercept_model: eqx.Module
    scale_slope_model: eqx.Module
    conditional_model: eqx.Module
    num_obs_dims: int
    num_cov_dims: int
    cond_variable_name: str
    spatial_dim_name: str
    
    
    def __call__(
        self,
        s: Float[Array, "S Ds"],
        x: Float[Array, "N Dx"],
        *args,
        y: Float[Array, "N S Dy"] | None = None,
        train: bool = True,
        **kwargs,
        ) -> Float[Array, "N S Dy"]:
        
        # apply GP Model - Intercepts
        location_intercept: Float[Array, "S Dy"] = self.location_intercept_model(s=s, y=y, train=train, **kwargs)
        scale_intercept: Float[Array, "S Dy"] = self.scale_intercept_model(s=s, y=y, train=train, **kwargs)
        
        # apply GP Model - Intercept
        location_slope: Float[Array, "S (Dy Dx)"] = self.location_slope_model(s=s, y=y, train=train, **kwargs)
        scale_slope: Float[Array, "S (Dy Dx)"] = self.scale_slope_model(s=s, y=y, train=train, **kwargs)
        
        # Rearrange parameters
        slope = einx.rearrange("S (Dy Dx) -> S Dy Dx", slope, Dy=self.num_obs_dims, Dx=self.num_cov_dims)
        
        # Conditional Model
        x_pred = self.conditional_model(x=x, y=y, train=train, **kwargs)
        
        # multiplication
        z: Float[Array, "N Dy"] = einx.dot("S Dy Dx, N Dx -> N S Dy", slope, x_pred)
        # addition
        z: Float[Array, "N Dy"] = einx.add("N S Dy, S Dy -> N S Dy", z, intercept)
        
        z = numpyro.deterministic(f"{self.name}", z)
        
        return z
    
    @property
    def dimensions(self):

        dims = {}
        for idict in [self.slope_model.dimensions, self.intercept_model.dimensions]:
            dims.update(idict)
            
        dims[f"{self.name}"] = [f"{self.cond_variable_name}", f"{self.spatial_dim_name}", f"variable"]
        return dims



    @property
    def variables(self):
        return list(self.dimensions.keys())


class LocSTGEVDLikelihood(eqx.Module):
    location: eqx.Module
    scale: dist.Distribution
    concentration_prior: dist.Distribution
    spatial_dim_name: str
    time_dim_name: str

    def __call__(
        self,
        s: Float[Array, "S Dx"],
        t: Float[Array, "T Dt"],
        *args,
        y: Float[Array, "T S Dy"] | None = None,
        time_batchsize: int = None,
        space_batchsize: int = None,
        **kwargs
        ) -> Float[Array, "T S Dy"]:
        
        num_time_steps = t.shape[0]
        num_spatial_steps = s.shape[0]
        
        if y is not None:
            assert len(y.shape) == 3
            assert y.shape[1] == s.shape[0]
            assert y.shape[0] == t.shape[0]
        
        
        plate_time = numpyro.plate("time_steps", size=num_time_steps, subsample_size=None, dim=-3)
        plate_space = numpyro.plate("spatial_steps", size=num_spatial_steps, subsample_size=None, dim=-2)
        
        # get models
        try:
            location = self.location(s=s, x=t, y=y, **kwargs)
        except TypeError:
            location = self.location(s=s, y=y, **kwargs)
        scale = self.scale(s=s, y=y, **kwargs)
        concentration = numpyro.sample("concentration", self.concentration_prior)
        obs_dist = tfd.GeneralizedExtremeValue(location, scale, concentration)
        
        with plate_time, plate_space:
            
            # get nans if exists
            if y is not None:
                
                # get nans
                mask = jnp.isfinite(y)
                
                # mask distribution
                obs_dist = tfd.Masked(obs_dist, mask)

            # sample
            y = numpyro.sample("obs", obs_dist, obs=y)
        
        return y
    @property
    def dimensions(self):

        dims = {}
        for idict in [self.location.dimensions, self.scale.dimensions]:
            dims.update(idict)
        dims["concentration"] = []
        dims["obs"] = [f"{self.time_dim_name}", f"{self.spatial_dim_name}", f"variable"]
        return dims



    @property
    def variables(self):
        return list(self.dimensions.keys())


class LocScaleGEVDLikelihood(eqx.Module):
    location_scale: eqx.Module
    concentration_prior: eqx.Module
    sample_dim_name: str

    def __call__(
        self,
        x: Float[Array, "N Dx"],
        *args,
        y: Float[Array, "N Dy"] | None = None,
        **kwargs
        ):
        
        assert len(x.shape) == 2
        num_samples = x.shape[0]
        
        if y is not None:
            assert len(y.shape) == 2
            assert y.shape[0] == x.shape[0]
        
        if x is not None and y is not None:
            assert x.shape[0] == y.shape[0]
        
        plate_time = numpyro.plate("timesteps", size=num_samples, dim=-2)
        
        # get models
        location_scale = self.location_scale(x=x, y=y, **kwargs)
        location, scale = location_scale[..., 0][..., None], location_scale[..., 1][..., None]
        concentration = numpyro.sample("concentration", self.concentration_prior)
        obs_dist = tfd.GeneralizedExtremeValue(location, scale, concentration)
        
        with plate_time:
            
            # get nans if exists
            if y is not None:
                
                # get nans
                mask = jnp.isfinite(y)
                
                # mask distribution
                obs_dist = tfd.Masked(obs_dist, mask)

            # sample
            y = numpyro.sample("obs", obs_dist, obs=y)
        
        return y
    @property
    def dimensions(self):

        dims = {}
        for idict in [self.location_scale.dimensions]:
            dims.update(idict)

        dims["concentration"] = []
        dims["obs"] = [f"{self.sample_dim_name}", f"variable"]
        return dims



    @property
    def variables(self):
        return list(self.dimensions.keys())


class Scalar(eqx.Module):
    prior_dist: dist.Distribution
    name: str

    def __call__(self, *args, **kwargs) -> Array:
        out = numpyro.sample(self.name, fn=self.prior_dist)
        return out

    @property
    def dimensions(self):
        return {self.name: []}

    @property
    def variables(self) -> List:
        return list(self.dimensions.keys())


class FixedScalar(eqx.Module):
    value: float
    name: str

    def __call__(self, *args, **kwargs) -> Array:
        out = numpyro.deterministic(self.name, value=self.value)
        return out

    @property
    def dimensions(self) -> Dict:
        return {self.name: []}

    @property
    def variables(self) -> List:
        return list(self.dimensions.keys())