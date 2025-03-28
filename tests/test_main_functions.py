import PolyaCplot as pc
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def test_1(
        x_range=(-10, 10),
        y_range=(-10, 10),
        density=20,
        colormap="plasma"
):
    z = sp.symbols('z')
    f_expr = 1 / z ** 2

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25)

    plt.sca(ax_left)
    pc.streamplot(f_expr, z, x_range=x_range, y_range=y_range, density=density, colormap=colormap)
    ax_left.set_title("Функция 1/z^2")

    a_init = 1
    f_2 = 1 / (z ** 2 - a_init ** 2)
    plt.sca(ax_right)

    pc.streamplot(f_2, z, x_range=x_range, y_range=y_range, density=density, colormap=colormap)
    ax_right.set_title("функция 1/(z^2 - a^2)")

    ax_slider = plt.axes((0.25, 0.1, 0.65, 0.03))
    slider = Slider(ax_slider, 'a', 0, 10, valinit=a_init, valstep=1)

    def update(val):
        a = int(slider.val)
        ax_right.clear()
        plt.sca(ax_right)
        f_new = 1 / (z ** 2 - a ** 2)
        pc.streamplot(f_new, z, x_range=x_range, y_range=y_range, density=density, colormap=colormap)
        ax_right.set_title("функция 1/(z ** 2 - a^2)")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def test_2(
        x_range=(-10, 10),
        y_range=(-10, 10),
        density=20,
        colormap="plasma"
):
    z = sp.symbols('z')
    f_expr = 1 / z ** 3

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25)

    plt.sca(ax_left)
    pc.streamplot(f_expr, z, x_range=x_range, y_range=y_range, density=density, colormap=colormap)
    ax_left.set_title("Функция 1/z^3")

    a_init = 1
    f_2 = 1 / (z ** 3 - a_init ** 3)
    plt.sca(ax_right)

    pc.streamplot(f_2, z, x_range=x_range, y_range=y_range, density=density, colormap=colormap)
    ax_right.set_title("функция 1/(z^3 - a^3)")

    ax_slider = plt.axes((0.25, 0.1, 0.65, 0.03))
    slider = Slider(ax_slider, 'a', 0, 10, valinit=a_init, valstep=1)

    def update(val):
        a = int(slider.val)
        ax_right.clear()
        plt.sca(ax_right)
        f_new = 1 / (z ** 3 - a ** 3)
        pc.streamplot(f_new, z, x_range=x_range, y_range=y_range, density=density, colormap=colormap)
        ax_right.set_title("функция 1/(z ** 3 - a^3)")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def test_3():
    z = sp.symbols('z')
    f_expr = z ** 2

    fig, ax = plt.subplots()

    x_range = (-5, 5)
    y_range = (-5, 5)

    pc.deformedCoordinateGrid(f_expr, z, ax, x_range=x_range, y_range=y_range, density=25)

    pc.streamplot(f_expr, z, ax, x_range=x_range, y_range=y_range)

    plt.show()


if __name__ == '__main__':
    test_3()
