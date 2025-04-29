import numpy as np
from typing import Optional, Any
from qc_eval.plotting.plot_core import PlotCore
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Import the color coding scheme.
from qc_eval.present_data.color_coding import POOLING_COLORS, \
    CONVOLUTION_COLORS


class Histogram(PlotCore):
    """
    A child class of PlotCore for creating histogram plots.

    Attributes:
        x (np.ndarray): Data array to be histogrammed.
        y (np.ndarray): Bin specification for the histogram (number of bins or
        bin edges).
        z (Optional[np.ndarray]): Weights for each entry in x, if applicable.
        x_ticks (Optional[np.ndarray]): Custom ticks for the x-axis.
        y_ticks (Optional[np.ndarray]): Custom ticks for the y-axis.
        layer_configuration (Optional[Any]): If provided, expected as a tuple
        (convolution_idx, pooling_idx) to apply a composite diagonal color
        coding scheme uniformly to all bins.
    """

    def __init__(self,
                 x: np.ndarray,
                 y: Optional[np.ndarray] = None,
                 z: Optional[np.ndarray] = None,
                 title: Optional[str] = None,
                 x_axis: Optional[str] = None,
                 y_axis: Optional[str] = None,
                 z_axis: Optional[str] = None,
                 cmap: Optional[str] = None,
                 x_ticks: Optional[np.ndarray] = None,
                 y_ticks: Optional[np.ndarray] = None,
                 layer_configuration: Optional[tuple[int, int]] = None):
        super().__init__(x, y, z, title, x_axis, y_axis, z_axis, cmap)
        self.x_ticks = x_ticks
        self.y_ticks = y_ticks
        self.layer_configuration = layer_configuration

    def plot(self) -> plt.Figure:
        """
        Creates a histogram plot based on the provided x data.
        If layer_configuration is provided (a tuple (pooling_idx, convolution_idx)),
        each bar is drawn with a composite diagonal split using the specified colors.
        Otherwise, a standard histogram is drawn.

        Returns:
            plt.Figure: A matplotlib Figure object containing the histogram.
        """
        # Initialize the default style parameters.
        self._init_default_params()
        fig, ax = plt.subplots()

        # Determine the bins.
        bins = self.y if self.y is not None else 'auto'
        weights = self.z if self.z is not None else None

        if self.layer_configuration is None:
            # Standard histogram
            ax.hist(self.x, bins=bins, weights=weights, color='blue',
                    edgecolor='black', alpha=0.7)
        else:
            # Compute the histogram counts and bin edges.
            counts, bin_edges = np.histogram(self.x, bins=bins,
                                             weights=weights)

            # Extract pooling and convolution indices from the layer configuration.
            try:
                convolution_idx, pooling_idx = self.layer_configuration
            except Exception:
                convolution_idx, pooling_idx = 1, 1  # default values if parsing fails

            # Retrieve the composite colors.
            diag_color2 = POOLING_COLORS.get(pooling_idx, POOLING_COLORS[1])
            diag_color1 = CONVOLUTION_COLORS.get(convolution_idx,
                                                 CONVOLUTION_COLORS[1])

            # Draw each histogram bin with a diagonal split using the chosen colors.
            for i in range(len(counts)):
                left_edge = bin_edges[i]
                right_edge = bin_edges[i + 1]
                height = counts[i]

                # Define the four corners of the bar.
                bottom_left = (left_edge, 0)
                bottom_right = (right_edge, 0)
                top_right = (right_edge, height)
                top_left = (left_edge, height)

                # Split the rectangle diagonally from bottom-left to top-right.
                triangle1 = Polygon([bottom_left, bottom_right, top_right],
                                    closed=True, facecolor=diag_color1,
                                    edgecolor='none')
                triangle2 = Polygon([bottom_left, top_right, top_left],
                                    closed=True, facecolor=diag_color2,
                                    edgecolor='none')

                ax.add_patch(triangle1)
                ax.add_patch(triangle2)

            ax.set_xlim(bin_edges[0], bin_edges[-1])
            ax.set_ylim(0, max(counts) * 1.1)

        # Apply custom ticks if provided.
        if self.x_ticks is not None:
            ax.set_xticks(self.x_ticks)
        if self.y_ticks is not None:
            ax.set_yticks(self.y_ticks)

        # Set the axis labels and title using the helper method from the parent class.
        self._set_ax(ax)
        return fig


if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd
    import ast
    import torch

    df = pd.read_csv(Path(__file__).parent.parent / "data" / "evaluation" /
                     "quantum_results_V1-0-5.csv")

    predictions = df.loc[0, "predictions"]
    predictions = eval(predictions, {"tensor": torch.tensor})
    labels = df.loc[0, "labels"]
    labels = ast.literal_eval(labels)
    x_value = np.array([pred[lab].detach().cpu().numpy()
                        for pred, lab in zip(predictions, labels)])
    hist = Histogram(x=x_value,
                     y=None,
                     x_axis="Prediction Probability",
                     y_axis="Count",
                     layer_configuration=(1, 2))
    hist.plot().show()
