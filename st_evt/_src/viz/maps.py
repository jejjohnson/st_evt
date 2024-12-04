
import xarray as xr
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt


def plot_spain(da: xr.DataArray, vmin: float | None = None, vmax: float | None = None, region: str = "mainland", **kwargs):


    fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': ccrs.PlateCarree()})
    if region == "mainland":
        ax.set_extent([-10, 4, 35, 45], crs=ccrs.PlateCarree())
    elif region == "canaries":
        ax.set_extent([-18.5, -13, 27.5, 29.5], crs=ccrs.PlateCarree())
    elif region == "all":
        pass
    else:
        pass
    
    pts = da.to_dataset().plot.scatter(
        ax=ax,
        x="lon", y="lat", hue=da.name,
        s=kwargs.get("s", 100.0),
        cmap=kwargs.get("cmap", "viridis"),
        c=kwargs.get("c", "Red"),
        vmin=vmin,
        vmax=vmax,
        marker="o",
        edgecolors="black",
        linewidths=kwargs.get("linewidths", 1.5),
        zorder=3,
        norm=kwargs.get("norm", None),
        add_colorbar=False,
    )
    cbar = fig.colorbar(pts, ax=ax, orientation="horizontal", pad=0.1, fraction=0.06)
    ax.set(title="")
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.1, color='k', alpha=1, 
                      linestyle='--')
    # ax.set(title=variable)
    ax.add_feature(cf.COASTLINE, linewidth=2)
    ax.add_feature(cf.BORDERS, linewidth=2)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12} 
    plt.tight_layout()
    
    return fig, ax, cbar


def plot_spain_mesh(da: xr.DataArray, vmin: float | None = None, vmax: float | None = None, region: str = "mainland", **kwargs):


    fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': ccrs.PlateCarree()})
    if region == "mainland":
        ax.set_extent([-10, 4, 35, 45], crs=ccrs.PlateCarree())
    elif region == "canaries":
        ax.set_extent([-18.5, -13, 27.5, 29.5], crs=ccrs.PlateCarree())
    elif region == "all":
        pass
    else:
        pass
    

    pts = da.plot.pcolormesh(
        ax=ax,
        x="lon", y="lat",
        # s=kwargs.get("s", 100.0),
        cmap=kwargs.get("cmap", "viridis"),
        # c=kwargs.get("c", "Red"),
        vmin=vmin,
        vmax=vmax,
        norm=kwargs.get("norm", None),
        edgecolors=kwargs.get("edgecolors", "black"),
        linewidths=kwargs.get("linewidths", 1.5),
        zorder=-10,
        add_colorbar=False,
    )
    
    cbar = fig.colorbar(pts, ax=ax, orientation="horizontal", pad=0.1, fraction=0.06)
    ax.set(title="")
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.1, color='k', alpha=1, 
                      linestyle='--')
    # ax.set(title=variable)
    ax.add_feature(cf.COASTLINE, linewidth=2)
    ax.add_feature(cf.BORDERS, linewidth=2)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12} 
    plt.tight_layout()
    
    return fig, ax, cbar