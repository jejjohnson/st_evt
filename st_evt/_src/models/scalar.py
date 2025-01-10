from typing import Callable, Dict, List
from jaxtyping import Array, Float
import equinox as eqx
import einx
from tinygp import kernels, GaussianProcess
import numpyro
import numpyro.distributions as dist
from tensorflow_probability.substrates.jax import distributions as tfd


class ScalarModel(eqx.Module):
    prior_dist: dist.Distribution
    name: str
    
    def __call__(self, *args, **kwargs):
        
        out = numpyro.sample(f"{self.name}", fn=self.prior_dist)
        
        return out
    
    @property
    def dimensions(self) -> Dict:
        return {str(self.name): []}

    @property
    def variables(self) -> List:
        return list(self.dimensions.keys())