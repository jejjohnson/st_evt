import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
import arviz as az


def plot_locationscale_return_regression(
    locations,
    x_axis: str = "time",
    scales = None,
    returns = None,
    observations = None,
    observations_window: bool=True
):

    fig, ax = plt.subplots(figsize=(10,5))
    
    
    ymin = locations.min().values
    ymin -= 0.01 * ymin
    if returns is not None:
        ymax = returns.max().values
    else:
        ymax = locations.max().values
    ymax += 0.01 * ymax
    
    if returns is not None:
        returns.sel(quantile=0.5, ).plot(
            ax=ax, color="tab:blue", linewidth=5, label="100-Year Return Level, $R_{100}(t)$"
        )
        ax.fill_between(
            returns[x_axis], 
            returns.sel(quantile=0.025),
            returns.sel(quantile=0.975),
            alpha=0.3, color="tab:blue",
            # label="95% Confidence Interval",
            label="",
        )
    
    locations.sel(quantile=0.5, method="nearest").plot(
        ax=ax, color="tab:red", linewidth=5, label=r"Location, $\mu(t)$",
    )
    
    ax.fill_between(
        locations[x_axis], 
        locations.sel(quantile=0.025, method="nearest"),
        locations.sel(quantile=0.975, method="nearest"),
        alpha=0.3, color="tab:red",
        label="",
    )
    
    if scales is not None:
        (locations.sel(quantile=0.5) + scales.sel(quantile=0.5)).plot(
            ax=ax, color="tab:green", linewidth=5, label=r"Scale, $\mu(t) + \sigma(t)$",
        )

        ax.fill_between(
            locations[x_axis], 
            locations.sel(quantile=0.025) + scales.sel(quantile=0.025),
            locations.sel(quantile=0.975) + scales.sel(quantile=0.975),
            alpha=0.3, color="tab:green",
            # label="95% Confidence Interval",
            label="",
        )
    
    if observations is not None:
        observations.plot.scatter(
            ax=ax,
            x=x_axis,
            s=10.0, 
            zorder=5, 
            color="black",
            marker="o",
            label="Observations",
        )

        if observations_window:
            ax.fill_betweenx(
                np.linspace(observations.min(), observations.max(), 10), 
                observations[x_axis].min(),
                observations[x_axis].max(),
                alpha=0.3, color="black",
                linestyle="dashed", 
                label="Observation Window",
                zorder=5,
            )
    
    
    ax.set(
        title="",
        xlabel="",
        ylabel=r"2m Max Temperature [°C]",
        xlim=[-0.25, 2.0],
        # yscale="log"
    )
    
    ax.grid(True, linestyle='--', linewidth='0.5', color='gray')
    ax.minorticks_on()
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    plt.legend()
    plt.tight_layout()

    return fig, ax
