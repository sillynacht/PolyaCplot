import PolyaCplot as pc
import numpy as np
import matplotlib.pyplot as plt


# gray to improve visibility on github's dark background
_gray = "#969696"
style = {
    "text.color": _gray,
    "axes.labelcolor": _gray,
    "axes.edgecolor": _gray,
    "xtick.color": _gray,
    "ytick.color": _gray,
}
plt.style.use(style)


args = [
    #
    ("z1.png", lambda z: z**1, (-2, +2), (-2, +2)),
    ("z2.png", lambda z: z**2, (-2, +2), (-2, +2)),
    ("z3.png", lambda z: z**3, (-2, +2), (-2, +2)),
    #
    ("1z.png", lambda z: 1 / z, (-2.0, +2.0), (-2.0, +2.0)),
    ("1z2.png", lambda z: 1 / z**2, (-2.0, +2.0), (-2.0, +2.0)),
    ("1z3.png", lambda z: 1 / z**3, (-2.0, +2.0), (-2.0, +2.0)),
    # möbius
    ("moebius1.png", lambda z: (z + 1) / (z - 1), (-5, +5), (-5, +5)),
    (
        "moebius2.png",
        lambda z: (z + 1.5 - 0.5j) * (1.5 - 0.5j) / (z - 1.5 + 0.5j) * (-1.5 + 0.5j),
        (-5, +5),
        (-5, +5),
    ),
    (
        "moebius3.png",
        lambda z: (-1.0j * z) / (1.0j * z + 1.5 - 0.5j),
        (-5, +5),
        (-5, +5),
    ),
    #
    # roots of unity
    ("z6+1.png", lambda z: z**6 + 1, (-1.5, 1.5), (-1.5, 1.5)),
    ("z6-1.png", lambda z: z**6 - 1, (-1.5, 1.5), (-1.5, 1.5)),
    ("z-6+1.png", lambda z: z ** (-6) + 1, (-1.5, 1.5), (-1.5, 1.5)),
    #
    ("zz.png", lambda z: z**z, (-3, +3), (-3, +3)),
    ("1zz.png", lambda z: (1 / z) ** z, (-3, +3), (-3, +3)),
    ("z1z.png", lambda z: z ** (1 / z), (-3, +3), (-3, +3)),
    #
    ("root2.png", np.sqrt, (-2, +2), (-2, +2)),
    ("root3.png", lambda x: x ** (1 / 3), (-2, +2), (-2, +2)),
    ("root4.png", lambda x: x**0.25, (-2, +2), (-2, +2)),
    #
    ("log.png", np.log, (-2, +2), (-2, +2)),
    ("exp.png", np.exp, (-3, +3), (-3, +3)),
    ("exp2.png", np.exp2, (-3, +3), (-3, +3)),
    #
    # non-analytic functions
    ("re.png", np.real, (-2, +2), (-2, +2)),
    # ("abs.png", np.abs, (-2, +2), (-2, +2)),
    ("z-absz.png", lambda z: z / np.abs(z), (-2, +2), (-2, +2)),
    ("conj.png", np.conj, (-2, +2), (-2, +2)),
    #
    # essential singularities
    ("exp1z.png", lambda z: np.exp(1 / z), (-1, +1), (-1, +1)),
    ("zsin1z.png", lambda z: z * np.sin(1 / z), (-0.6, +0.6), (-0.6, +0.6)),
    ("cos1z.png", lambda z: np.cos(1 / z), (-0.6, +0.6), (-0.6, +0.6)),
    #
    ("exp-z2.png", lambda z: np.exp(-(z**2)), (-3, +3), (-3, +3)),
    ("11z2.png", lambda z: 1 / (1 + z**2), (-3, +3), (-3, +3)),
    #
    ("exp1z1.png", lambda z: np.exp(1 / z) / (1 + np.exp(1 / z)), (-1, 1), (-1, 1)),
    #
    # generating function of fibonacci sequence
    ("fibonacci.png", lambda z: 1 / (1 - z * (1 + z)), (-5.0, +5.0), (-5.0, +5.0)),
    #
    ("sin.png", np.sin, (-5, +5), (-5, +5)),
    ("cos.png", np.cos, (-5, +5), (-5, +5)),
    ("tan.png", np.tan, (-5, +5), (-5, +5)),
    #
    ("sec.png", lambda z: 1 / np.cos(z), (-5, +5), (-5, +5)),
    ("csc.png", lambda z: 1 / np.sin(z), (-5, +5), (-5, +5)),
    ("cot.png", lambda z: 1 / np.tan(z), (-5, +5), (-5, +5)),
    #
    ("sinh.png", np.sinh, (-5, +5), (-5, +5)),
    ("cosh.png", np.cosh, (-5, +5), (-5, +5)),
    ("tanh.png", np.tanh, (-5, +5), (-5, +5)),
    #
    ("sech.png", lambda z: 1 / np.cosh(z), (-5, +5), (-5, +5)),
    ("csch.png", lambda z: 1 / np.sinh(z), (-5, +5), (-5, +5)),
    ("coth.png", lambda z: 1 / np.tanh(z), (-5, +5), (-5, +5)),
    #
    ("arcsin.png", np.arcsin, (-2, +2), (-2, +2)),
    ("arccos.png", np.arccos, (-2, +2), (-2, +2)),
    ("arctan.png", np.arctan, (-2, +2), (-2, +2)),
    #
    ("arcsinh.png", np.arcsinh, (-2, +2), (-2, +2)),
    ("arccosh.png", np.arccosh, (-2, +2), (-2, +2)),
    ("arctanh.png", np.arctanh, (-2, +2), (-2, +2)),
    #
    ("sinz-z.png", lambda z: np.sin(z) / z, (-7, +7), (-7, +7)),
    ("cosz-z.png", lambda z: np.cos(z) / z, (-7, +7), (-7, +7)),
    ("tanz-z.png", lambda z: np.tan(z) / z, (-7, +7), (-7, +7)),
    #
    # hyperbolic functions
    (
        "tanh-sinh.png",
        lambda z: np.tanh(np.pi / 2 * np.sinh(z)),
        (-2.5, +2.5),
        (-2.5, +2.5),
    ),
    (
        "sinh-sinh.png",
        lambda z: np.sinh(np.pi / 2 * np.sinh(z)),
        (-2.5, +2.5),
        (-2.5, +2.5),
    ),
    (
        "exp-sinh.png",
        lambda z: np.exp(np.pi / 2 * np.sinh(z)),
        (-2.5, +2.5),
        (-2.5, +2.5),
    ),
    # logistic regression:
    ("sigmoid.png", lambda z: 1.0 / (1.0 + np.exp(-z)), (-10, +10), (-10, +10))
]


for filename, fun, x, y in args:
    plt.figure(figsize=(6, 4))
    pc.streamplot(fun, x_range=x, y_range=y, density=25, colormap="plasma")
    plt.savefig("images/" + filename, transparent=True, bbox_inches="tight")
    plt.close()
