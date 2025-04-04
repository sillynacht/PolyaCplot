import sympy as sp
import matplotlib.pyplot as plt
import PolyaCplot as pc
import imageio.v2 as imageio
from PIL import Image
import shutil
import os


_gray = "#969696"
style = {
    "text.color": _gray,
    "axes.labelcolor": _gray,
    "axes.edgecolor": _gray,
    "xtick.color": _gray,
    "ytick.color": _gray,
}
plt.style.use(style)


def cos_series_gif():
    os.makedirs("gifs/frames", exist_ok=True)

    frames = []
    n_min, n_max = 3, 20

    for n in range(n_min, n_max + 1):
        plt.clf()
        plt.figure(figsize=(6, 4))
        z = sp.Symbol("z", complex=True)
        f_expr = sp.cos(z)

        func = pc.taylor_poly(f_expr, z, n)

        pc.streamplot(func, z, x_range=(-5, 5), y_range=(-5, 5), density=25)
        plt.title("maclaurin series of cos(z)")
        plt.legend()

        frame_path = f"gifs/frames/frame_{n}.png"
        plt.savefig(frame_path, transparent=True, bbox_inches="tight")
        plt.close()

        frames.append(Image.open(frame_path))

    frames[0].save(
        "gifs/cos_taylor.gif",
        save_all=True,
        append_images=frames[1:],
        duration=500,
        loop=0,
        disposal=2,
    )

    shutil.rmtree("gifs/frames")


def exp_series_gif_with_orig():
    os.makedirs("gifs/frames", exist_ok=True)

    frames = []
    n_min, n_max = 3, 20

    for n in range(n_min, n_max + 1):
        fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize=(12, 6))
        z = sp.Symbol("z", complex=True)
        f_expr = sp.exp(z)

        pc.streamplot(f_expr, z, ax_1, x_range=(-5, 5), y_range=(-5, 5))

        ax_1.set_title("streamplot of exp(z)")

        func = pc.taylor_poly(f_expr, z, n)
        pc.streamplot(func, z, ax_2, x_range=(-5, 5), y_range=(-5, 5))

        ax_2.set_title(f"maclaurin series of exp(z) with n={n} terms")

        frame_path = f"gifs/frames/frame_{n}.png"
        plt.savefig(frame_path)
        plt.close()

        frames.append(imageio.imread(frame_path))

    gif_path = "gifs/exp_taylor_with_orig.gif"
    imageio.mimsave(gif_path, frames, fps=2)

    shutil.rmtree("gifs/frames")


# trigonometric functions
def cos_series_gif_with_orig():
    os.makedirs("gifs/frames", exist_ok=True)

    frames = []
    n_min, n_max = 3, 20

    for n in range(n_min, n_max + 1):
        fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize=(12, 6))
        z = sp.Symbol("z", complex=True)
        f_expr = sp.cos(z)

        pc.streamplot(f_expr, z, ax_1, x_range=(-5, 5), y_range=(-5, 5))

        ax_1.set_title("streamplot of cos(z)")

        func = pc.taylor_poly(f_expr, z, n)
        pc.streamplot(func, z, ax_2, x_range=(-5, 5), y_range=(-5, 5))

        ax_2.set_title(f"maclaurin series of cos(z) with n={n} terms")

        frame_path = f"gifs/frames/frame_{n}.png"
        plt.savefig(frame_path)
        plt.close()

        frames.append(imageio.imread(frame_path))

    gif_path = "gifs/cos_taylor_with_orig.gif"
    imageio.mimsave(gif_path, frames, fps=2)

    shutil.rmtree("gifs/frames")


def sin_series_gif_with_orig():
    os.makedirs("gifs/frames", exist_ok=True)

    frames = []
    n_min, n_max = 3, 20

    for n in range(n_min, n_max + 1):
        fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize=(12, 6))
        z = sp.Symbol("z", complex=True)
        f_expr = sp.sin(z)

        pc.streamplot(f_expr, z, ax_1, x_range=(-5, 5), y_range=(-5, 5))

        ax_1.set_title("streamplot of sin(z)")

        func = pc.taylor_poly(f_expr, z, n)
        pc.streamplot(func, z, ax_2, x_range=(-5, 5), y_range=(-5, 5))

        ax_2.set_title(f"maclaurin series of sin(z) with n={n} terms")

        frame_path = f"gifs/frames/frame_{n}.png"
        plt.savefig(frame_path)
        plt.close()

        frames.append(imageio.imread(frame_path))

    gif_path = "gifs/sin_taylor_with_orig.gif"
    imageio.mimsave(gif_path, frames, fps=2)

    shutil.rmtree("gifs/frames")


def tan_series_gif_with_orig():
    os.makedirs("gifs/frames", exist_ok=True)

    frames = []
    n_min, n_max = 3, 20

    for n in range(n_min, n_max + 1):
        fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize=(12, 6))
        z = sp.Symbol("z", complex=True)
        f_expr = sp.tan(z)

        pc.streamplot(f_expr, z, ax_1, x_range=(-float(sp.pi * sp.sqrt(2)) / 4, float(sp.pi * sp.sqrt(2)) / 4), y_range=(-float(sp.pi * sp.sqrt(2)) / 4, float(sp.pi * sp.sqrt(2)) / 4))

        ax_1.set_title("streamplot of tan(z)")

        func = pc.taylor_poly(f_expr, z, n)
        pc.streamplot(func, z, ax_2, x_range=(-float(sp.pi * sp.sqrt(2)) / 4, float(sp.pi * sp.sqrt(2)) / 4), y_range=(-float(sp.pi * sp.sqrt(2)) / 4, float(sp.pi * sp.sqrt(2)) / 4))

        ax_2.set_title(f"maclaurin series of tan(z) with n={n} terms")

        frame_path = f"gifs/frames/frame_{n}.png"
        plt.savefig(frame_path)
        plt.close()

        frames.append(imageio.imread(frame_path))

    gif_path = "gifs/tan_taylor_with_orig.gif"
    imageio.mimsave(gif_path, frames, fps=2, disposal=2)

    shutil.rmtree("gifs/frames")


def geom_series_gif_with_orig():
    os.makedirs("gifs/frames", exist_ok=True)

    frames = []
    n_min, n_max = 3, 20

    for n in range(n_min, n_max + 1):
        fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize=(12, 6))
        z = sp.Symbol("z", complex=True)
        f_expr = 1 / (1 - z)

        pc.streamplot(f_expr, z, ax_1, x_range=(-float(sp.sqrt(2)) / 2, float(sp.sqrt(2)) / 2), y_range=(-float(sp.sqrt(2)) / 2, float(sp.sqrt(2)) / 2))

        ax_1.set_title("streamplot of 1 / (1 - z)")

        func = pc.taylor_poly(f_expr, z, n)
        pc.streamplot(func, z, ax_2, x_range=(-float(sp.sqrt(2)) / 2, float(sp.sqrt(2)) / 2), y_range=(-float(sp.sqrt(2)) / 2, float(sp.sqrt(2)) / 2))

        ax_2.set_title(f"geometric series with n={n} terms")

        frame_path = f"gifs/frames/frame_{n}.png"
        plt.savefig(frame_path)
        plt.close()

        frames.append(imageio.imread(frame_path))

    gif_path = "gifs/geom_taylor_with_orig.gif"
    imageio.mimsave(gif_path, frames, fps=2)

    shutil.rmtree("gifs/frames")


if __name__ == "__main__":
    cos_series_gif()
