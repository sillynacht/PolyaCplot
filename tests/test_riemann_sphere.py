import PolyaCplot as pc
import sympy as sp


if __name__ == '__main__':
    z = sp.symbols('z')
    f_expr = (z ** 2 - 1)
    pc.riemannVectorplot(f_expr, z, density=50, colormap="plasma", R_split=1.0, max_magnitude=1000.0)
