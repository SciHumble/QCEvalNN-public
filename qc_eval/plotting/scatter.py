from qc_eval.plotting import PlotCore
from matplotlib import pyplot as plt
import numpy as np
from typing import Optional, Any
from matplotlib.patches import Wedge

# Import your color schemes.
from qc_eval.present_data.color_coding import POOLING_COLORS, \
    CONVOLUTION_COLORS


class Scatter(PlotCore):
    def __init__(self, x: np.ndarray, y: np.ndarray,
                 z: Optional[np.ndarray] = None, title: Optional[str] = None,
                 x_axis: Optional[str] = None, y_axis: Optional[str] = None,
                 z_axis: Optional[str] = None, cmap: Optional[str] = None,
                 labels: Optional[list[Any]] = None,
                 layer_configuration: Optional[list[tuple[int, int]]] = None,
                 x_ticks: Optional[np.ndarray] = None,
                 x_ticklabels: Optional[np.ndarray] = None):
        """
        Initialize the Scatter plot.

        The labels parameter is expected to be a list of tuples,
        where each tuple is (pooling_index, convolution_index) corresponding
        to the desired colors from POOLING_COLORS and CONVOLUTION_COLORS.
        If layer_configuration is provided (not None), composite markers will be used.
        """
        super().__init__(x, y, z, title, x_axis, y_axis, z_axis, cmap)
        if labels:
            if self.x.shape[0] != len(labels):
                raise ValueError(
                    "The number of labels must match the shape of "
                    "x and y data."
                )
            self.labels = labels
        else:
            # Default to (1, 1) for all points if no label is provided.
            self.labels = None
        self.layer_configuration = layer_configuration

        self.x_ticks, self.x_ticklabels = x_ticks, x_ticklabels


    @staticmethod
    def _draw_composite_marker(ax, x, y, pooling_idx: int,
                               convolution_idx: int,
                               size: float = 0.1, border: bool = False):
        """
        Draws a composite marker at (x, y) by splitting a circular marker
        into two halves. The left half (from 90° to 270°) uses the pooling color,
        and the right half (from 270° to 450°) uses the convolution color.

        Args:
            ax: The matplotlib Axes to draw on.
            x, y: Coordinates of the marker center.
            pooling_idx: An integer key (1-4) for selecting the pooling color.
            convolution_idx: An integer key (1-6) for selecting the convolution color.
            size: The radius of the marker.
            border: If True, draws an edge around the marker.
        """
        pooling_color = POOLING_COLORS.get(pooling_idx, POOLING_COLORS[1])
        convolution_color = CONVOLUTION_COLORS.get(convolution_idx,
                                                   CONVOLUTION_COLORS[1])

        # Configure border settings.
        edgecolor = 'black' if border else 'none'
        linewidth = 1 if border else 0

        # Draw left half: covers angles 90° to 270°.
        left_half = Wedge(center=(x, y), r=size, theta1=90, theta2=270,
                          facecolor=convolution_color, edgecolor=edgecolor,
                          lw=linewidth)
        # Draw right half: covers angles 270° to 450°.
        right_half = Wedge(center=(x, y), r=size, theta1=270, theta2=450,
                           facecolor=pooling_color, edgecolor=edgecolor,
                           lw=linewidth)

        ax.add_patch(left_half)
        ax.add_patch(right_half)

    def plot(self) -> plt.Figure:
        self._init_default_params()
        fig, ax = plt.subplots()
        self._set_ax(ax)

        if self.x_ticks is not None:
            ax.set_xticks(self.x_ticks)
        if self.x_ticklabels is not None:
            ax.set_xticklabels(self.x_ticklabels)

        # Determine an appropriate marker size based on data range.
        x_range = self.x.max() - self.x.min()
        y_range = self.y.max() - self.y.min()
        marker_size = x_range * 0.025  # adjust this factor as needed

        if self.layer_configuration is not None:
            ax.set_ylim((self.y.min()-y_range*0.1, self.y.max()+y_range*0.1))
            ax.set_xlim((self.x.min()-x_range*0.1, self.x.max()+x_range*0.1))
            # Use composite markers.
            for xi, yi, layer in zip(self.x, self.y, self.layer_configuration):
                if isinstance(layer, tuple) and len(layer) == 2:
                    convolution_idx, pooling_idx = layer
                else:
                    convolution_idx, pooling_idx = 1, 1
                self._draw_composite_marker(ax, xi, yi, pooling_idx,
                                            convolution_idx,
                                            size=marker_size, border=False)
        else:
            # Use standard scatter markers.
            if self.labels:
                labels = self.labels
            else:
                labels = [None] * self.x.shape[0]
            for xi, yi, label in zip(self.x, self.y, labels):
                plot_params = self._get_plot_params(xi, yi, label)
                pcm = ax.scatter(**plot_params)
                if self.z is not None:
                    fig.colorbar(pcm, ax=ax, **self._get_colorbar_params())
            if any(lab is not None for lab in labels):
                ax.legend()

        return fig


if __name__ == "__main__":
    scatter = Scatter(x=np.random.rand(2, 16),
                      y=np.random.rand(2, 16),
                      labels=["Random 1", "Random 2"])
    fig = scatter.plot()
    fig.set_size_inches(*scatter.get_figsize(0.9))
    plt.show()
