import sys

sys.path.append("../")
import inspect
import gob.optimizers as go
import gob.metrics as gm
import gob.benchmarks as gb
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["text.usetex"] = True


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

    create_dir(Path("./source"))

    wd = Path("./source")

    # Optimizers

    create_dir(wd / "optimizers")
    optimizers = inspect.getmembers(go, inspect.isclass)
    files = []
    for name, opt in optimizers:
        opt = opt([])

        file_content = (
            f"{opt}\n"
            f"{''.join(['='] * len(str(opt)))}\n\n"
            f".. automodule:: gob.optimizers.{name}\n"
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
            ben = ben(1, 1, [-1, 1], -1)
            file_content = (
                f"{ben}\n"
                f"{''.join(['='] * len(str(ben)))}\n\n"
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

            x = np.linspace(-10, 10, 500)
            y = np.linspace(-10, 10, 500)
            X, Y = np.meshgrid(x, y)
            Z = np.array(
                [ben(np.array([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))]
            ).reshape(X.shape)
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

            ax.plot_surface(X, Y, Z, cmap="coolwarm")
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            plt.savefig(
                wd / f"benchmarks/graphs/{name}.png",
                dpi=300,
                bbox_inches="tight",
                transparent=True,
            )
            plt.close()

            file_content = (
                f"{ben}\n"
                f"{''.join(['='] * len(str(met)))}\n\n"
                f".. image:: graphs/{name}.png\n"
                "   :width: 500px\n"
                "   :height: 500px\n"
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
