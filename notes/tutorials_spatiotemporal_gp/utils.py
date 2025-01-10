from typing import Callable, Dict
from tinygp.kernels import Kernel
import equinox as eqx
from jaxtyping import Array, Float
from numpyro.distributions import Distribution
import numpyro.distributions as dist
from tinygp import kernels, transforms
from jax.scipy.linalg import cholesky, solve_triangular
import einx
import numpyro
import jax.numpy as jnp
import jax




class LinearModelv2(eqx.Module):
    name: str
    input_dim_name: str
    output_dim_name: str
    slope: Array
    intercept: Array

    def __call__(self, x: Float[Array, "... Dx"]) -> Float[Array, "... Dz"]:
        mu = einx.dot("... Dx, Dx Dz -> ... Dz", x, self.slope)
        mu = einx.add("... Dz, Dz -> ... Dz", mu, self.intercept)
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
        }

    @property
    def variables(self):
        return list(self.dimensions.keys())


class BatchedLinearModelv2(eqx.Module):
    name: str
    input_dim_name: str
    output_dim_name: str
    latent_dim_name: str
    slope: Float[Array, "B Dx Dy"]
    intercept: Float[Array, "B Dy"]

    def __call__(self, x: Float[Array, "... Dx"]) -> Float[Array, "... B Dz"]:
        mu = einx.dot("... Dx, B Dx Dz -> B ... Dz", x, self.slope)
        mu = einx.add("B ... Dz, B Dz -> B ... Dz", mu, self.intercept)
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
            f"{self.name}_intercept": [f"{self.latent_dim_name}", f"{self.output_dim_name}"],
            f"{self.name}_slope": [f"{self.latent_dim_name}", f"{self.input_dim_name}", f"{self.output_dim_name}"],          
        }

    @property
    def variables(self):
        return list(self.dimensions.keys())
    

class VariationalGP(eqx.Module):
    x: Float[Array, "N D"]
    num_output_dims: int
    jitter: float
    mean: eqx.Module
    kernel: eqx.Module
    
    @property
    def num_points(self):
        return self.x.shape[0]

    def prior_dist(self, *args, **kwargs):
        # Kernel Matrix, K_ff: [N, N]
        K_ff = self.kernel(self.x, self.x)
        K_ff += self.jitter * jnp.eye(self.num_points)
        # Zero Mean, μ_0: [Dz, N]
        zero_loc = jnp.zeros((self.num_output_dims,) + (self.num_points,))
        # Cholesky, L_ff: [N, N]
        L_ff = cholesky(K_ff, lower=True)
        # Identity, K_I_ff: [N, N]
        identity = jnp.eye(self.num_points)
        # GP Distribution - MVN
        gp_dist = dist.MultivariateNormal(loc=zero_loc, covariance_matrix=identity).to_event(zero_loc.ndim - 1)
        
        return L_ff, gp_dist

    def posterior_predictive_dist(self, x: Float[Array, "M D"], *args, **kwargs):
        # Cross-Covariance Matrix, K_fs: [N, M]
        K_fs = self.kernel(self.x, x)
        # Test-Covariance Matrix, K_ss: [M, M]
        K_ss = self.kernel(x, x)
        # (inv(L_ff) @ Kf*).T, K_ss: [M, N]
        W = solve_triangular(L_ff, K_fs, lower=True).T
        pass


class LinearModel(eqx.Module):
    
    name: str
    input_dim_name: str
    output_dim_name: str
    num_input_dims: int
    num_output_dims: int
    slope_prior: Distribution
    intercept_prior: Distribution

    def __init__(
        self,
        num_input_dims: int=1,
        num_output_dims: int=1,
        name: str = "mean",
        input_dim_name: str = "input_dim",
        output_dim_name: str = "output_dim",
        slope_prior: Distribution = dist.Normal(),
        intercept_prior: Distribution = dist.Normal(),
    ):
        self.name = name
        self.input_dim_name = input_dim_name
        self.output_dim_name = output_dim_name
        self.num_input_dims = num_input_dims
        self.num_output_dims = num_output_dims
        self.slope_prior = slope_prior
        self.intercept_prior = intercept_prior
            
    def __call__(self, x: Float[Array, "Dx"], params: Dict) -> Float[Array, "Dz"]:
        mu = einx.dot("Dx, Dx Dz -> Dz", x, params["slope"])
        mu = einx.add("Dz, Dz -> Dz", mu, params["intercept"])
        return mu

    @property
    def dimensions(self):
        return {
            f"{self.name}_intercept": [f"{self.output_dim_name}"],
            f"{self.name}_slope": [f"{self.input_dim_name}", f"{self.output_dim_name}"],          
        }

    @property
    def variables(self):
        return list(self.dimensions.keys())

    def init_numpyro_prior(self) -> Dict:
        params = dict()
        params["slope"] = numpyro.sample(f"{self.name}_slope", self.slope_prior, sample_shape=(self.num_input_dims, self.num_output_dims,))
        params["intercept"] = numpyro.sample(f"{self.name}_intercept", self.intercept_prior, sample_shape=(self.num_output_dims,))
        return params

    def init_numpyro_params(self) -> Callable:
        params = dict()
        init_value = jnp.zeros((self.num_input_dims, self.num_output_dims,))
        params["slope"] = numpyro.param(f"{self.name}_slope", init_value=init_value)
        init_value = jnp.zeros((self.num_output_dims,))
        params["intercept"] = numpyro.param(f"{self.name}_intercept", init_value=init_value)
        return params
    
    
class ExponentialModel(eqx.Module):
    
    name: str
    input_dim_name: str
    output_dim_name: str
    num_input_dims: int
    num_output_dims: int
    slope_prior: Distribution
    intercept_prior: Distribution

    def __init__(
        self,
        num_input_dims: int=1,
        num_output_dims: int=1,
        name: str = "mean",
        input_dim_name: str = "input_dim",
        output_dim_name: str = "output_dim",
        slope_prior: Distribution = dist.Normal(),
        intercept_prior: Distribution = dist.Normal(),
    ):
        self.name = name
        self.input_dim_name = input_dim_name
        self.output_dim_name = output_dim_name
        self.num_input_dims = num_input_dims
        self.num_output_dims = num_output_dims
        self.slope_prior = slope_prior
        self.intercept_prior = intercept_prior
            
    def __call__(self, x: Float[Array, "Dx"], params: Dict) -> Float[Array, "Dz"]:
        mu = einx.dot("Dx, Dx Dz -> Dz", x, params["slope"])
        mu = einx.multiply("Dz, Dz -> Dz", jnp.exp(mu), params["intercept"])
        return mu

    @property
    def dimensions(self):
        return {
            f"{self.name}_intercept": [f"{self.output_dim_name}"],
            f"{self.name}_slope": [f"{self.input_dim_name}", f"{self.output_dim_name}"],          
        }

    @property
    def variables(self):
        return list(self.dimensions.keys())

    def init_numpyro_prior(self) -> Dict:
        params = dict()
        params["slope"] = numpyro.sample(f"{self.name}_slope", self.slope_prior, sample_shape=(self.num_input_dims, self.num_output_dims,))
        params["intercept"] = numpyro.sample(f"{self.name}_intercept", self.intercept_prior, sample_shape=(self.num_output_dims,))
        return params

    def init_numpyro_params(self) -> Callable:
        params = dict()
        init_value = jnp.zeros((self.num_input_dims, self.num_output_dims,))
        params["slope"] = numpyro.param(f"{self.name}_slope", init_value=init_value)
        init_value = jnp.zeros((self.num_output_dims,))
        params["intercept"] = numpyro.param(f"{self.name}_intercept", init_value=init_value)
        return params
    

class CoupledExponentialModel(eqx.Module):
    
    name: str
    input_dim_name: str
    output_dim_name: str
    num_input_dims: int
    num_output_dims: int
    slope_prior: Distribution
    intercept_prior: Distribution

    def __init__(
        self,
        num_input_dims: int = 1,
        name: str = "mean",
        input_dim_name: str = "input_dim",
        output_dim_name: str = "output_dim",
        slope_prior: Distribution = dist.Normal(),
        intercept_prior: Distribution = dist.Normal(),
    ):
        self.name = name
        self.input_dim_name = input_dim_name
        self.output_dim_name = output_dim_name
        self.num_input_dims = num_input_dims
        self.num_output_dims = 2
        self.slope_prior = slope_prior
        self.intercept_prior = intercept_prior
            
    def __call__(self, x: Float[Array, "Dx"], params: Dict) -> Float[Array, "2"]:
        # split input
        mu, sigma = x[0], x[1]
        # do linear
        mu = params["loc_intercept"] + (params["loc_slope"] / params["loc_intercept"]) * x
        sigma = params["scale_intercept"] + (params["scale_slope"] / params["loc_intercept"]) * x
        out = jnp.concat([mu, sigma])
        return out

    @property
    def dimensions(self):
        return {
            f"{self.name}_loc_intercept": [],
            f"{self.name}_loc_slope": [],
            f"{self.name}_scale_intercept": [],
            f"{self.name}_scale_slope": [],   
        }

    @property
    def variables(self):
        return list(self.dimensions.keys())

    def init_numpyro_prior(self) -> Dict:
        params = dict()
        params["loc_slope"] = numpyro.sample(f"{self.name}_loc_slope", self.slope_prior,)
        params["loc_intercept"] = numpyro.sample(f"{self.name}_loc_intercept", self.intercept_prior, )
        params["scale_slope"] = numpyro.sample(f"{self.name}_scale_slope", self.slope_prior,)
        params["scale_intercept"] = numpyro.sample(f"{self.name}_scale_intercept", self.intercept_prior, )
        return params

    
    
class ARDKernel(eqx.Module):
    
    name: str
    input_dim_name: str
    num_input_dims: int
    scale_prior: Distribution
    variance_prior: Distribution

    def __init__(
        self,
        num_input_dims: int=1,
        name: str = "kernel",
        input_dim_name: str = "input_dim",
        scale_prior: Distribution = dist.HalfNormal(2.0),
        variance_prior: Distribution = dist.HalfNormal(2.0),
    ):
        self.name = name
        self.input_dim_name = input_dim_name
        self.num_input_dims = num_input_dims
        self.scale_prior = scale_prior
        self.variance_prior = variance_prior
            
    def __call__(self, x, y, params: Dict) -> Kernel:
        kernel = params["variance"] * transforms.Linear(
            1/params["scale"], kernels.ExpSquared()
        )
        return kernel(x, y)

    @property
    def dimensions(self):
        return {
            f"{self.name}_variance": [],
            f"{self.name}_scale": ["input_dims"],          
        }

    @property
    def variables(self):
        return list(self.dimensions.keys())

    def init_numpyro_prior(self) -> Dict:
        params = dict()
        params["variance"] = numpyro.sample(f"{self.name}_variance", self.variance_prior, )
        params["scale"] = numpyro.sample(f"{self.name}_scale", self.scale_prior, sample_shape=(self.num_input_dims,))
        return params

    def init_numpyro_params(self) -> Callable:
        params = dict()
        params["variance"] = numpyro.param(f"{self.name}_variance", init_value=1.0)
        init_value = 1.0 * jnp.ones((self.num_input_dims,))
        params["scale"] = numpyro.param(f"{self.name}_intercept", init_value=init_value)
        return params
    

class ARDLinearKernel(eqx.Module):
    
    name: str
    input_dim_name: str
    num_input_dims: int
    scale_prior: Distribution
    variance_prior: Distribution

    def __init__(
        self,
        num_input_dims: int=1,
        name: str = "kernel",
        input_dim_name: str = "input_dim",
        scale_prior: Distribution = dist.HalfNormal(2.0),
        variance_prior: Distribution = dist.HalfNormal(2.0),
    ):
        self.name = name
        self.input_dim_name = input_dim_name
        self.num_input_dims = num_input_dims
        self.scale_prior = scale_prior
        self.variance_prior = variance_prior
            
    def __call__(self, x, y, params: Dict) -> Kernel:
        kernel = params["variance"] * transforms.Linear(
            1/params["scale"], kernels.DotProduct()
        )
        return kernel(x, y)

    @property
    def dimensions(self):
        return {
            f"{self.name}_variance": [],
            f"{self.name}_scale": ["input_dims"],          
        }

    @property
    def variables(self):
        return list(self.dimensions.keys())

    def init_numpyro_prior(self) -> Dict:
        params = dict()
        params["variance"] = numpyro.sample(f"{self.name}_variance", self.variance_prior, )
        params["scale"] = numpyro.sample(f"{self.name}_scale", self.scale_prior, sample_shape=(self.num_input_dims,))
        return params

    def init_numpyro_params(self) -> Callable:
        params = dict()
        params["variance"] = numpyro.param(f"{self.name}_variance", init_value=1.0)
        init_value = 1.0 * jnp.ones((self.num_input_dims,))
        params["scale"] = numpyro.param(f"{self.name}_intercept", init_value=init_value)
        return params
    
class ARDKernelBatched(eqx.Module):
    
    name: str
    input_dim_name: str
    num_input_dims: int
    output_dim_name: str
    num_output_dims: int
    scale_prior: Distribution
    variance_prior: Distribution

    def __init__(
        self,
        num_input_dims: int = 1,
        num_output_dims: int = 1,
        name: str = "kernel",
        input_dim_name: str = "input_dim",
        output_dim_name: str = "output_dim",
        scale_prior: Distribution = dist.HalfNormal(2.0),
        variance_prior: Distribution = dist.HalfNormal(2.0),
    ):
        self.name = name
        self.input_dim_name = input_dim_name
        self.output_dim_name = output_dim_name
        self.num_input_dims = num_input_dims
        self.num_output_dims = num_output_dims
        self.scale_prior = scale_prior
        self.variance_prior = variance_prior
            
    def __call__(self, x, y, params: Dict) -> Kernel:
        
        def fn(params):
            kernel = params["variance"] * transforms.Linear(
                1/params["scale"], kernels.ExpSquared()
            )
            K = kernel(x, y)
            return K
        
        K = jax.vmap(fn)(params)

        return K

    @property
    def dimensions(self):
        return {
            f"{self.name}_variance": [f"{self.output_dim_name}"],
            f"{self.name}_scale": [f"{self.output_dim_name}", f"{self.input_dim_name}"],          
        }

    @property
    def variables(self):
        return list(self.dimensions.keys())

    def init_numpyro_prior(self) -> Dict:
        params = dict()
        params["variance"] = numpyro.sample(f"{self.name}_variance", self.variance_prior, sample_shape=(self.num_output_dims,))
        params["scale"] = numpyro.sample(f"{self.name}_scale", self.scale_prior, sample_shape=(self.num_output_dims, self.num_input_dims,))
        return params

    def init_numpyro_params(self) -> Callable:
        params = dict()
        params["variance"] = numpyro.param(f"{self.name}_variance", init_value=1.0)
        init_value = 1.0 * jnp.ones((self.num_input_dims,))
        params["scale"] = numpyro.param(f"{self.name}_intercept", init_value=init_value)
        return params