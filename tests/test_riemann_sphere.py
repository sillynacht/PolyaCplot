import PolyaCplot as pc
import sympy as sp


if __name__ == '__main__':
    z = sp.symbols('z')
    f_expr = z ** 2

    pc.riemannVectorplot(f_expr, z, density=40, colormap="plasma", max_magnitude=1000.0)
