from pathlib import Path
from jaxtyping import Array, Float
from typing import List, Optional
import numpy as np
import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tensorflow_probability.substrates.jax import distributions as tfd


def plot_qq_plot_gevd_manual(
    location: Array,
    scale: Array,
    concentration: Array,
    observations: Array,
    figures_save_dir: str | None = None
    ):
    y_min = observations.min()
    y_max = observations.max()

    diff = 0.10 * (y_max - y_min)
    y_domain = np.linspace(y_min - diff, y_max + diff, 100)

    fn_quantile = lambda x: tfd.GeneralizedExtremeValue(
        loc=jnp.asarray(location.squeeze()),
        scale=jnp.asarray(scale.squeeze()),
        concentration=jnp.asarray(concentration.squeeze()),
    ).quantile(x)

    xp = np.linspace(0, 1, 100)
    # calculate model quantiles
    model_quantile = np.asarray(jax.vmap(fn_quantile)(xp))
    # calculate empirical quantiles
    empirical_quantile = np.quantile(observations, xp)

    mq_cl, mq_mu, mq_cu = np.quantile(model_quantile, q=[0.025, 0.5, 0.975], axis=1)

    fig, ax = plt.subplots()
    ax.scatter(mq_mu, empirical_quantile, color="tab:blue", s=30.0, zorder=3, label="Median")
    ax.scatter(mq_cu, empirical_quantile, color="gray", s=10.0, alpha=0.4, zorder=3, label="CI Params")

    ax.scatter(mq_cl, empirical_quantile, color="gray", s=10.0, alpha=0.4, zorder=3)
    ax.plot(y_domain, y_domain, color="black", label="Ideal")

    ax.set(
        xlim=[y_domain[0], y_domain[-1]],
        ylim=[y_domain[0], y_domain[-1]],
        xlabel="Theoretical Quantiles",
        ylabel="Observed Quantiles"
        
    )
    ax.set_aspect('equal', 'box')
    ax.grid(True, which="both", linestyle='--', linewidth='0.5', color='gray')
    ax.minorticks_on()
    plt.legend()
    plt.tight_layout()
    if figures_save_dir is not None:
        fig.savefig(Path(figures_save_dir).joinpath(f"qqplot.png")) 
        plt.close()
    else:
        return fig, ax