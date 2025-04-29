from matplotlib import pyplot as plt
import numpy as np
from qc_eval.plotting.plot_core import PlotCore
from typing import Optional
from qc_eval.plotting.plotting_parameters import DefaultPlotting


class Heatmap(PlotCore):
    cmap: str

    def __init__(self, x: np.ndarray,
                 y: np.ndarray,
                 z: np.ndarray = None,
                 title: Optional[str] = None,
                 x_axis: Optional[str] = None,
                 y_axis: Optional[str] = None,
                 z_axis: Optional[str] = None,
                 cmap: Optional[str] = None):
        super().__init__(x, y, z, title, x_axis, y_axis, z_axis, cmap)
        if x.ndim != 1:
            raise ValueError(
                f"The x data needs a dimension of 1."
            )
        if y.ndim != 1:
            raise ValueError(
                f"The y data needs a dimension of 1."
            )
        if z.shape == (x.size, y.size):
            self.z = self.z.transpose()
        elif z.shape != (y.size, x.size):
            raise ValueError(
                f"The z data needs the shape of (x.size, y.size)."
            )
        self.cmap = self.cmap or DefaultPlotting.cmap.value

    def plot(self) -> plt.Figure:
        self._init_default_params()
        fig, ax = plt.subplots()

        im = ax.imshow(self.z, origin='lower', cmap=self.cmap,
                       aspect='equal')

        ax.set_xticks(np.arange(len(self.x)))
        ax.set_xticklabels(self.x)
        ax.set_yticks(np.arange(len(self.y)))
        ax.set_yticklabels(self.y)

        cbar = fig.colorbar(im, ax=ax)
        if self.z_axis:
            cbar.set_label(self.z_axis)

        if self.title:
            ax.set_title(self.title)
        if self.x_axis:
            ax.set_xlabel(self.x_axis)
        if self.y_axis:
            ax.set_ylabel(self.y_axis)

        return fig
