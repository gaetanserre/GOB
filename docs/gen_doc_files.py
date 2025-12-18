import sys

sys.path.append("../")
import inspect
import gob.optimizers as go
import gob.metrics as gm
import gob.benchmarks as gb
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


matplotlib.rcParams.update({"font.size": 9, "text.usetex": True})


def create_dir(path: Path):
    if not path.exists():
        path.mkdir(exist_ok=True, parents=True)


def create_file(path: Path, content: str):
    with open(path, "w") as f:
        f.write(content)


def generate_index(path: Path, name: str, files: list[Path]):
    files = "\n   ".join(files)
    file_content = (
        f"{name}\n"
        f"{''.join(['='] * len(name))}\n\n"
        ".. toctree::\n"
        "   :maxdepth: 1\n\n"
        f"   {files}"
    )
    create_file(path / "index.rst", file_content)


if __name__ == "__main__":

    wd = Path("./source")

    # Optimizers

    create_dir(wd / "optimizers")
    optimizers = inspect.getmembers(go, inspect.isclass)
    files = []
    for name, opt in optimizers:
        path_opt = Path(inspect.getsourcefile(opt))

        # Get subpackage names
        parts = path_opt.parts
        i = parts.index("optimizers")
        subpackages = list(parts[i + 1 : -1])
        opt = opt([])

        file_content = (
            f"{opt}\n"
            f"{''.join(['='] * len(str(opt)))}\n\n"
            f".. automodule:: gob.optimizers.{".".join(subpackages)}.{name}\n"
            "   :members:\n"
            "   :show-inheritance:\n"
            "   :undoc-members:\n"
        )
        path = wd / f"optimizers/gob.{name}.rst"
        create_file(path, file_content)
        files.append(path.stem)
    generate_index(wd / "optimizers", "Optimizers", files)

    # Metrics

    create_dir(wd / "metrics")
    metrics = inspect.getmembers(gm, inspect.isclass)
    files = []
    for name, met in metrics:
        met = met(None, None)

        file_content = (
            f"{met}\n"
            f"{''.join(['='] * len(str(met)))}\n\n"
            f".. automodule:: gob.metrics.{name.lower()}\n"
            "   :members:\n"
            "   :show-inheritance:\n"
            "   :undoc-members:\n"
        )
        path = wd / f"metrics/gob.{name}.rst"
        create_file(path, file_content)
        files.append(path.stem)
    generate_index(wd / "metrics", "Metrics", files)

    # Benchmarks

    create_dir(wd / "benchmarks")
    benchmarks = inspect.getmembers(gb, inspect.isclass)
    files = []
    for name, ben in benchmarks:
        if name == "PyGKLS":
            seed = 125794
            ben = ben(2, 30, [-10, 10], -10, gen=seed)
            x = np.linspace(-10, 10, 500)
            y = np.linspace(-10, 10, 500)
            X, Y = np.meshgrid(x, y)
            Z = np.array(
                [ben(np.array([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))]
            ).reshape(X.shape)
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

            ax.plot_surface(X, Y, Z, linewidth=0.2, edgecolors="white", cmap="coolwarm")
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            ax.set_box_aspect(None, zoom=0.8)
            xticks = ax.xaxis.get_major_ticks()
            yticks = ax.yaxis.get_major_ticks()
            for i in range(0, len(xticks), 2):
                xticks[i].label1.set_visible(False)
            for i in range(0, len(yticks), 2):
                yticks[i].label1.set_visible(False)
            plt.savefig(
                wd / f"benchmarks/graphs/{name}.png",
                dpi=400,
                bbox_inches="tight",
                transparent=True,
            )
            plt.close()

            file_content = (
                f"{ben}\n"
                f"{''.join(['='] * len(str(ben)))}\n\n"
                f".. image:: graphs/{name}.png\n"
                "   :width: 550px\n"
                "   :height: 550px\n"
                "   :align: center\n\n"
                "Uses the `pyGKLS <https://pypi.org/project/gkls/>`_ package to generate random test functions, with control over their geometry. \n\n"
                f".. automodule:: gob.benchmarks.{name.lower()}\n"
                "   :members:\n"
                "   :show-inheritance:\n"
                "   :undoc-members:\n"
            )
            create_file(wd / f"benchmarks/gob.{name}.rst", file_content)
        else:
            ben = ben()

            # Generate graphs

            create_dir(wd / "benchmarks/graphs")

            bounds = ben.visual_bounds
            x = np.linspace(bounds[0][0], bounds[0][1], 500)
            y = np.linspace(bounds[1][0], bounds[1][1], 500)
            X, Y = np.meshgrid(x, y)
            Z = np.array(
                [ben(np.array([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))]
            ).reshape(X.shape)
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.plot_surface(X, Y, Z, linewidth=0.2, edgecolors="white", cmap="coolwarm")
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            ax.set_box_aspect(None, zoom=0.8)
            xticks = ax.xaxis.get_major_ticks()
            yticks = ax.yaxis.get_major_ticks()
            for i in range(0, len(xticks), 2):
                xticks[i].label1.set_visible(False)
            for i in range(0, len(yticks), 2):
                yticks[i].label1.set_visible(False)
            plt.savefig(
                wd / f"benchmarks/graphs/{name}.png",
                dpi=400,
                bbox_inches="tight",
                transparent=True,
            )
            plt.close()

            file_content = (
                f"{ben}\n"
                f"{''.join(['='] * len(str(ben)))}\n\n"
                f".. image:: graphs/{name}.png\n"
                "   :width: 550px\n"
                "   :height: 550px\n"
                "   :align: center\n\n"
                f".. automodule:: gob.benchmarks.{name.lower()}\n"
                "   :members:\n"
                "   :show-inheritance:\n"
                "   :undoc-members:\n"
            )
        path = wd / f"benchmarks/gob.{name}.rst"
        create_file(path, file_content)
        files.append(path.stem)
    generate_index(wd / "benchmarks", "Benchmarks", files)
