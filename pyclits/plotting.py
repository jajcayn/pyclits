"""
Basic plotting.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap


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
    ):
        self.lats = lats
        self.lons = lons
        self.whole_world = whole_world

        if whole_world:
            self._set_whole_world_basemap()
        else:
            self._set_local_basemap()

        self.colormesh = colormesh

    def _set_whole_world_basemap(self):
        m = Basemap(projection="robin", lon_0=0, resolution="l")
        self.m = m

    def _draw_whole_world_map(self):
        self.m.drawparallels(
            np.arange(-90, 90, 30),
            linewidth=1.2,
            labels=[1, 0, 0, 0],
            color="#222222",
            size=15,
        )
        self.m.drawmeridians(
            np.arange(-180, 180, 60),
            linewidth=1.2,
            labels=[0, 0, 0, 1],
            color="#222222",
            size=15,
        )
        self.m.drawcoastlines(linewidth=1.6, color="#333333")
        self.m.drawcountries(linewidth=1.1, color="#333333")

    def _set_local_basemap(self):
        m = Basemap(
            projection="merc",
            llcrnrlat=self.lats[0],
            urcrnrlat=self.lats[-1],
            llcrnrlon=self.lons[0],
            urcrnrlon=self.lons[-1],
            resolution="i",
        )
        self.m = m

    def _draw_local_map(self):
        draw_lats = np.arange(
            np.around(self.lats[0] / 5, decimals=0) * 5,
            np.around(self.lats[-1] / 5, decimals=0) * 5,
            10,
        )
        draw_lons = np.arange(
            np.around(self.lons[0] / 5, decimals=0) * 5,
            np.around(self.lons[-1] / 5, decimals=0) * 5,
            20,
        )
        self.m.drawparallels(
            draw_lats,
            linewidth=1.2,
            labels=[1, 0, 0, 0],
            color="#222222",
            size=15,
        )
        self.m.drawmeridians(
            draw_lons,
            linewidth=1.2,
            labels=[0, 0, 0, 1],
            color="#222222",
            size=15,
        )
        self.m.drawcoastlines(linewidth=1.0, color="#333333")
        self.m.drawcountries(linewidth=0.7, color="#333333")

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

        if self.whole_world:
            self._draw_whole_world_map()
        else:
            self._draw_local_map()

        x, y = self.m(*np.meshgrid(self.lons, self.lats))

        vmax = vmax or np.nanmax(data)
        vmin = vmin or np.nanmin(data)
        if symmetric_plot:
            if np.abs(vmax) > np.abs(vmin):
                vmin = -vmax
            else:
                vmax = -vmin
        levels = np.linspace(vmin, vmax, 41)
        if self.colormesh:
            cs = self.m.pcolormesh(x, y, data, vmin=vmin, vmax=vmax, cmap=cmap)
        else:
            cs = self.m.contourf(x, y, data, levels=levels, cmap=cmap)
        cbar = plt.colorbar(
            cs, ticks=levels[::4], pad=0.07, shrink=0.8, fraction=0.05
        )
        cbar.ax.set_yticklabels(np.around(levels[::4], decimals=2), size=15)
        cbar.set_label(cbar_label, rotation=90, size=18)
