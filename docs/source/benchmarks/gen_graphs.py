import sys

sys.path.append("../../..")
import inspect
import gob.benchmarks as gb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["text.usetex"] = True

benchmark_functions = inspect.getmembers(gb, inspect.isclass)

x = np.linspace(-10, 10, 500)
y = np.linspace(-10, 10, 500)
X, Y = np.meshgrid(x, y)

for name, f in benchmark_functions:
    if name == "PyGKLS":
        continue
    print(f"Generating graph for {name}...")
    f = f()
    Z = np.array(
        [f(np.array([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))]
    ).reshape(X.shape)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    ax.plot_surface(X, Y, Z, cmap="coolwarm")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    plt.savefig(f"graphs/{name}.png", dpi=300, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"Graph for {name} saved.")
