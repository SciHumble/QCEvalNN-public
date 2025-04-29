from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Union
from matplotlib import pyplot as plt
from pathlib import Path


class PlotCore(ABC):
    default_style_file: Path = Path(__file__).parent / "default.mplstyle"

    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 z: Optional[np.ndarray] = None,
                 title: Optional[str] = None,
                 x_axis: Optional[str] = None,
                 y_axis: Optional[str] = None,
                 z_axis: Optional[str] = None,
                 cmap: Optional[str] = None):
        self.x = x
        self.y = y
        self.z = z
        self.title = title
        self.x_axis = self._make_string_pretty(x_axis)
        self.y_axis = self._make_string_pretty(y_axis)
        self.z_axis = self._make_string_pretty(z_axis)

        # https://matplotlib.org/stable/users/explain/colors/colormaps.html
        self.cmap = cmap

    @abstractmethod
    def plot(self) -> plt.Figure:
        """
        This method should create a matplotlib plot, but should not create an
        output.
        Examples:
            plotter = PlotCore(**input_dict)
            fig = plotter.plot()
            # fig.savefig(file_location)
            plt.show() # or plt.savefig(file_location)
        Returns:
            Figure of the plot
        """
        pass

    def _init_default_params(self):
        plt.style.use(self.default_style_file)

    def _set_ax(self, ax: plt.Axes) -> None:
        if self.cmap is not None:
            plt.set_cmap(self.cmap)
        if self.title is not None:
            ax.set_title(self.title)
        if self.x_axis is not None:
            ax.set_xlabel(self.x_axis)
        if self.y_axis is not None:
            ax.set_ylabel(self.y_axis)

    def _get_plot_params(self, x, y, label: Optional[str] = None) -> dict:
        value = {"x": x, "y": y}
        if label:
            value["label"] = label
        if self.z is not None:
            value["c"] = self.z
            if self.cmap is not None:
                value["cmap"] = self.cmap
        return value

    def _get_colorbar_params(self) -> dict:
        value = dict()
        if self.z_axis is not None:
            value["label"] = self.z_axis
        return value

    @staticmethod
    def get_figsize(width: float,
                    ratio: Optional[tuple[float, float]] = None,
                    textwidth_in_inch: float = 5.9
                    ) -> tuple[float, float]:
        """
        Calculate figure size based on LaTeX textwidth.

        Args:
            width: Fraction of text width (e.g., 0.9 or 0.45).
            ratio: (width, height) ratio. Default is (1, 0.6).
            textwidth_in_inch: Width of LaTeX textblock in inches.

        Returns:
            (figure width, figure height) in inches
        """
        if ratio is None:
            ratio = (1, 0.6)
        else:
            ratio = (1, ratio[1] / ratio[0])

        def width_mult(x: float) -> float:
            return x * width * textwidth_in_inch

        return tuple(map(width_mult, ratio))

    @staticmethod
    def _make_string_pretty(value: Union[None, str]) -> Union[None, str]:
        if value is not None:
            value = " ".join(value.split("_"))
            value.capitalize()
        return value
