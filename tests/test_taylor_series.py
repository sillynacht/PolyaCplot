import sympy as sp
import matplotlib.pyplot as plt
import PolyaCplot as pc


def test_cos_series():
    z = sp.symbols('z')
    f_expr = sp.cos(z)
    fig, ax = plt.subplots()
    pc.taylor_series(f_expr, z, fig, ax, x_range=(-5, 5), y_range=(-5, 5), n_init=5, n_min=2, n_max=20)
    plt.show()


def test_sin_series():
    z = sp.symbols('z')
    f_expr = sp.sin(z)
    fig, ax = plt.subplots()
    pc.taylor_series(f_expr, z, fig, ax, x_range=(-5, 5), y_range=(-5, 5), n_init=5, n_min=2, n_max=20)
    plt.show()


def test_exp_series():
    z = sp.symbols('z')
    f_expr = sp.exp(z)
    fig, ax = plt.subplots()
    pc.taylor_series(f_expr, z, fig, ax, x_range=(-5, 5), y_range=(-5, 5), n_init=5, n_min=2, n_max=20)
    plt.show()


if __name__ == "__main__":
    test_exp_series()
