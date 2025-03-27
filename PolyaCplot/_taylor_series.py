import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
from ._main import streamplot

from typing import Optional, Callable
from matplotlib.figure import Figure
from matplotlib.widgets import Slider

import matplotlib
matplotlib.use('Qt5Agg')


def taylorPoly(f_expr, var, n_terms):
    poly = 0
    for k in range(n_terms + 1):
        term = f_expr.diff(var, k).subs(var, 0) / sp.factorial(k) * var**k
        poly += term
    return sp.simplify(poly)


def maclaurin_series(
        f_expr: sp.Expr | Callable[[np.ndarray], np.ndarray],
        z: sp.Symbol,
        fig: Figure,
        ax: Optional[plt.Axes] = None,
        x_range: tuple[float, float] = (-5, 5),
        y_range: tuple[float, float] = (-5, 5),
        density: int | float = 25,
        colormap: str = "plasma",
        n_init: int = 5,
        n_min: int = 3,
        n_max: int = 20,
        n_step: int = 1
):
    if n_min < 2:
        raise ValueError("Minimum number of terms in Taylor series should be at least 2.")

    if ax is None:
        ax = plt.gca()

    taylor_expr = taylorPoly(f_expr, z, n_init)
    streamplot(taylor_expr, z, ax, x_range=x_range, y_range=y_range, density=density, colormap=colormap)

    ax_slider = plt.axes([0.1, 0.02, 0.8, 0.05])
    slider = Slider(ax_slider, 'n', n_min , n_max, valinit=n_init, valstep=n_step)

    def update(val):
        n = int(slider.val)
        ax.clear()
        new_taylor_expr = taylorPoly(f_expr, z, n)
        streamplot(new_taylor_expr, z, ax, x_range=x_range, y_range=y_range, density=density, colormap=colormap)
        fig.canvas.draw_idle()

    slider.on_changed(update)
