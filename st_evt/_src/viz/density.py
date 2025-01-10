import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
import arviz as az


def plot_density_multiple(
    entries,
    log_bins: bool = False,
):
    fig, ax = plt.subplots()
    for ientry in entries:
        icolor = str(ientry["color"])
        ivalues = ientry["values"]
        ilinestyle = ientry.get("linestyle", "-")
        ivalue_units = ientry.get("values_units", "")
        iperiod = ientry.get("period", "")
        

        metric_median = np.nanmedian(ivalues.ravel())
        
        if log_bins:
            hist, bins = np.histogram(ivalues.ravel(), bins=20)
            bins = np.logspace(np.log10(0.01),np.log10(bins[-1]),len(bins))
        else:
            bins = 20

        
        ax.hist(
            ientry["values"].ravel(),
            density=True, bins=bins, linewidth=3, 
            color=icolor,
            # label=f"{iname}", 
            label=f"", 
            zorder=3,
            alpha=0.25
        )


        plot_kwargs = {
            "color": icolor,
            "linewidth": 4,
            "label": f"{iperiod}",
            "linestyle": ilinestyle,
        }
        az.plot_kde(ivalues.ravel(), ax=ax, plot_kwargs=plot_kwargs)
        ax.scatter(
            metric_median,
            0.0,
            color=icolor,
            clip_on=False,
            zorder=10,
            marker="o",
            edgecolor="black",
            label=f"Median: {metric_median:.2f} {ivalue_units}"
        )



    ax.set(
        
        ylabel="Probability Density Function",
        title="",
    )

    ax.grid(True, linestyle='--', linewidth='0.5', color='gray')
    ax.minorticks_on()
    return fig, ax


def plot_periods(
    entries,
    log_bins: bool = False,
):
    fig, ax = plt.subplots()
    for ientry in entries:
        icolor = ientry["color"]
        ivalues = ientry["values"]
        ivalue_units = ientry.get("values_units", "")
        iperiod = ientry["period"]
        

        metric_median = np.nanmedian(ivalues.ravel())
        
        if log_bins:
            hist, bins = np.histogram(ivalues.ravel(), bins=20)
            bins = np.logspace(np.log10(0.01),np.log10(bins[-1]),len(bins))
        else:
            bins = 20

        
        ax.hist(
            ientry["values"].ravel(),
            density=True, bins=bins, linewidth=3, 
            color=f"tab:{icolor}",
            # label=f"{iname}", 
            label=f"", 
            zorder=3,
            alpha=0.25
        )


        plot_kwargs = {
            "color": f"tab:{icolor}",
            "linewidth": 4,
            "label": f"{iperiod}"
        }
        az.plot_kde(ivalues.ravel(), ax=ax, plot_kwargs=plot_kwargs)
        ax.scatter(
            metric_median,
            0.0,
            color=f"tab:{icolor}",
            clip_on=False,
            zorder=10,
            marker="o",
            edgecolor="black",
            label=f"Median: {metric_median:.2f} {ivalue_units}"
        )



    ax.set(
        
        ylabel="Probability Density Function",
        title="",
    )

    ax.grid(True, linestyle='--', linewidth='0.5', color='gray')
    ax.minorticks_on()
    return fig, ax

def plot_periods_diff(
    entries,
    log_bins: bool = False,
):
    fig, ax = plt.subplots()
    for ientry in entries:
        icolor = ientry["color"]
        ivalues = ientry["values"]
        ivalue_units = ientry.get("values_units", "")
        ilabel = ientry["label"]

        if log_bins:
            hist, bins = np.histogram(ivalues.ravel(), bins=20)
            bins = np.logspace(np.log10(0.01),np.log10(bins[-1]),len(bins))
        else:
            bins = 20

        
        ivalues.plot.hist(
            ax=ax,
            density=True, bins=25, linewidth=3, 
            color=f"{icolor}",
            label="",
            zorder=3,
            alpha=0.25
        )
        

        plot_kwargs = {
            "color": icolor,
            "linewidth": 4,
            "label": f"{ilabel}"
        }
        az.plot_kde(ivalues.values.ravel(), ax=ax, plot_kwargs=plot_kwargs)

        metric_lb, metric_median, metric_ub = ivalues.quantile(q=[0.025, 0.5, 0.975])
        ax.scatter(
            metric_median,
            0.0,
            color=icolor,
            clip_on=False,
            zorder=10,
            marker="o",
            edgecolor="black",
            label=f"Median: {metric_median:.2f} ({metric_lb:.2f}, {metric_ub:.2f})" + f" {ivalue_units}"
        )



    ax.set(
        
        ylabel="Probability Density Function",
        title="",
    )

    ax.grid(True, linestyle='--', linewidth='0.5', color='gray')
    ax.minorticks_on()
    return fig, ax