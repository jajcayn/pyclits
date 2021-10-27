"""
Basic plotting.
"""

import matplotlib.pyplot as plt
import numpy as np
from cartopy.crs import Mercator, PlateCarree, Robinson
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER


class GeoPlot:
    """
    Helper class for geographical map plots.
    """

    def __init__(
        self,
        lats,
        lons,
        whole_world=False,
        colormesh=False,
        projection=None,
    ):
        self.lats = lats
        self.lons = lons
        self.whole_world = whole_world
        if projection is None:
            self.projection = Robinson() if whole_world else Mercator()
        else:
            self.projection = projection

        self.colormesh = colormesh

    def _set_up_map(self):
        self.axis = plt.axes(projection=self.projection)
        if self.whole_world:
            self.axis.set_global()

    def _add_map_lines(self):
        """
        Add coastlines, gridlines, and countries' boundaries to the map.
        """
        self.axis.coastlines(resolution="10m")
        gl = self.axis.gridlines(draw_labels=True, linestyle="--")
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.top_labels = False
        country = NaturalEarthFeature(
            category="cultural",
            name="admin_0_boundary_lines_land",
            scale="10m",
            facecolor="none",
            edgecolor="k",
            linewidth=1.2,
        )
        self.axis.add_feature(country)

    def plot(
        self,
        data,
        vmin=None,
        vmax=None,
        cmap="viridis",
        cbar=True,
        cbar_label="",
        symmetric_plot=False,
    ):
        assert data.shape[0] == self.lats.shape[0]
        assert data.shape[1] == self.lons.shape[0]

        self._set_up_map()
        self._add_map_lines()

        vmax = vmax or np.nanmax(data)
        vmin = vmin or np.nanmin(data)
        if symmetric_plot:
            if np.abs(vmax) > np.abs(vmin):
                vmin = -vmax
            else:
                vmax = -vmin
        levels = np.linspace(vmin, vmax, 41)
        if self.colormesh:
            cs = plt.pcolormesh(
                self.lons,
                self.lats,
                data,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                transform=PlateCarree(),
            )
        else:
            cs = plt.contourf(
                self.lons,
                self.lats,
                data,
                levels=levels,
                cmap=cmap,
                transform=PlateCarree(),
            )
        cbar = plt.colorbar(
            cs, ticks=levels[::4], pad=0.07, shrink=0.8, fraction=0.05
        )
        cbar.ax.set_yticklabels(np.around(levels[::4], decimals=2), size=15)
        cbar.set_label(cbar_label, rotation=90, size=18)
