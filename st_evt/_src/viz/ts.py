from typing import Optional
import matplotlib.pyplot as plt
import arviz as az


def plot_scatter_ts(
    ds_bm,
    x_variable: str="time",
    figures_save_dir: Optional[str]=None,
    ax=None,
    **kwargs
):
    if ax is not None:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(figsize=(10,5))

    pts = ds_bm.plot(ax=ax, x=x_variable, zorder=3, color="tab:red", linestyle="", marker="o", markersize=kwargs.get("markersize", 10.0))
    ax.set(
        xlabel="Time [Years]",
        title=f"",
    )
    ax.grid(True, linestyle='--', linewidth='0.5', color='gray')
    ax.minorticks_on()
    plt.tight_layout()
    return fig, ax, pts


def plot_histogram(ds, ax=None, figures_save_dir: Optional[str]=None):
    if ax is not None:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(figsize=(10,5))
    ds.plot.hist(ax=ax, bins=20, linewidth=4, density=False, fill=False)
    ax.set(
        ylabel="Number of Observations",
        title=f"",
    )
    ax.grid(True, linestyle='--', linewidth='0.5', color='gray')
    ax.minorticks_on()
    plt.tight_layout()
    return fig, ax
    
    
def plot_density(ds, ax=None, figures_save_dir: Optional[str]=None):
    if ax is not None:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(figsize=(10,5))
    ds.plot.hist(
        ax=ax,
        density=True, bins=20, linewidth=1, 
        color="black",
        label="Histogram", zorder=3,
        alpha=0.25, 
    )
    
    plot_kwargs = {
        "color": f"black",
        "linewidth": 4,
        "label": "KDE Fit"
    }
    az.plot_kde(ds.values.ravel(), ax=ax, plot_kwargs=plot_kwargs)
    
    # sns.kdeplot(
    #     np.asarray(ds.values.ravel()), 
    #     ax=ax,
    #     color=f"black",
    #     linewidth=5, label="KDE Fit"
    # )
    ax.grid(True, linestyle='--', linewidth='0.5', color='gray')
    ax.minorticks_on()
    ax.set(
        ylabel=r"Probability Density, $p(y)$",
        title=f"",
    )
    plt.tight_layout()
    return fig, ax

