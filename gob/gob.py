#
# Created in 2024 by Gaëtan Serré
#
import numpy as np
import inspect

import gob.optimizers as go
import gob.benchmarks as gb
import gob.metrics as gm

from .benchmarks.create_bounds import create_bounds

from .utils import print_table_by_metric
from .utils import print_competitive_ratios
from .utils import print_blue, print_dark_green

from .benchmarks import PyGKLS


class GOB:
    """
    Global Optimization Benchmarks.
    """

    def __init__(self, optimizers, benchmarks, metrics, bounds=None, options={}):
        """
        Initialize the benchmarking tool.

        Parameters
        ----------
        optimizers : List tuple(str | class, dict) | List str | List Object
            The optimizers to use. A tuple is (name | class, dict of keyword arguments).

        benchmarks : List str | Object
            The benchmarks to use.

        metrics : List str | Object
            The metrics to use.

        bounds : array-like of shape (n_benchmark, n_variables, 2)
            The bounds of the search space.

        options : dict of keyword arguments for the metrics
            {name: dict of keyword arguments}
        """
        if bounds is None:
            bounds = create_bounds(len(benchmarks), -1, 1, 2)

        self.bounds = bounds

        self.options = options
        self.optimizers = optimizers
        self.benchmarks = benchmarks
        self.metrics = metrics

        self.count_gkls = 1

    def parse_optimizer(self, optimizer, bounds):
        """
        Parse the optimizer.

        Parameters
        ----------
        optimizer : tuple(str | class, dict) | str | Object
            The optimizer to use.
        bounds : array-like of shape (n_variables, 2)
            The bounds of the search space.

        Returns
        -------
        Optimizer
            Instance of the optimizer.
        """

        def get_class_by_name(name):
            optimizers = inspect.getmembers(go, inspect.isclass)
            for _, opt in optimizers:
                if str(opt([])) == name:
                    return opt
            raise ValueError(f"Unknown optimizer: {name}")

        if isinstance(optimizer, tuple):
            name, options = optimizer
            if isinstance(name, str):
                return get_class_by_name(name)(bounds=bounds, **options)
            else:
                return name(bounds=bounds, **options)
        elif isinstance(optimizer, str):
            return get_class_by_name(optimizer)(bounds=bounds)
        else:
            return optimizer

    def parse_benchmark(self, benchmark):
        """
        Parse the benchmark.

        Parameters
        ----------
        benchmark : str
            The benchmark to use.

        Returns
        -------
        Benchmark
            Instance of the benchmark.
        """
        if isinstance(benchmark, str):
            benchmarks = inspect.getmembers(gb, inspect.isclass)
            for name, bench in benchmarks:
                if name == "PyGKLS":
                    continue
                if str(bench()) == benchmark:
                    return bench()
            raise ValueError(f"Unknown benchmark: {benchmark}")

        elif isinstance(benchmark, PyGKLS):
            benchmark.name = f"{benchmark} n°{self.count_gkls}"
            self.count_gkls += 1
            return benchmark
        else:
            return benchmark

    def parse_metric(self, metric, benchmark, bounds, options={}):
        """
        Parse the metric.

        Parameters
        ----------
        metric : str | Object
            The metric to use.
        benchmark : Benchmark
            The benchmark function.
        bounds : array-like of shape (n_variables, 2)
            The bounds of the search space.
        options : dict of keyword arguments
            The options for the metric.

        Returns
        -------
        List Metric
            Instance of the metric.
        """
        if isinstance(metric, str):
            metrics = inspect.getmembers(gm, inspect.isclass)
            for _, met in metrics:
                if str(met(None, None)) == metric:
                    return met(benchmark, bounds, **options)
            raise ValueError(f"Unknown metric: {metric}")
        else:
            return metric

    @staticmethod
    def _print_approx(sols, f, n_runs):
        """
        Print the approximate minimum.

        Parameters
        ----------
        sols : List float
            The list of approximations of the minimum.
        f : float
            The true minimum.
        n_runs : int
            The number of runs.
        """
        mean, std = np.mean(sols), np.std(sols)
        if n_runs == 1:
            print(f"Minimum: {sols[0]:.6f}, True minimum: {f}")
        else:
            print(f"Minimum: {mean:.4f} ± {std:.4f}, True minimum: {f}")

    def competitive_ratio(self, res_dict, min_dict):
        """
        Compute the competitive ratio: :math:`\\frac{1}{|F|}\\sum\\limits_{f \\in F} \\frac{\\text{approx}(f)}{\\min_x f(x)}`.

        Parameters
        ----------
        res_dict : dict
            The results dictionary.
        min_dict : dict
            The minimum values of the benchmarks.
        benchmarks : List Object
            The benchmarks.

        Returns
        -------
        dict
            A dict of competitive ratios.
        """
        optimizer_names = list(res_dict.values())[0].keys()
        ratios = {}
        approxs = {}
        for optimizer_name in optimizer_names:
            for bench_name, bench_dict in res_dict.items():
                if bench_name not in approxs:
                    approxs[bench_name] = []
                approxs[bench_name].append(
                    np.abs(
                        bench_dict[optimizer_name]["Approx"]["mean"]
                        - min_dict[bench_name]
                    )
                )
        for optimizer_name in optimizer_names:
            ratio = 0
            for bench_name, bench_dict in res_dict.items():
                approx = np.abs(
                    bench_dict[optimizer_name]["Approx"]["mean"] - min_dict[bench_name]
                )
                best = np.min(approxs[bench_name])
                ratio += min(100, (approx + 1e-10) / (best + 1e-10))
            ratios[optimizer_name] = ratio / len(self.benchmarks)
        return ratios

    @staticmethod
    def _get_duplicate_name(bench_dict, name, duplicates_name):
        if name in bench_dict:
            if name not in duplicates_name:
                duplicates_name[name] = 2
            else:
                duplicates_name[name] += 1
            return f"{name} ({duplicates_name[name]})"
        else:
            return name

    def run(self, n_runs=1, verbose=0):
        """
        Run the benchmark.

        Parameters
        ----------
        n_runs : int
            The number of runs to perform.
        verbose : int
            The verbosity level.
        """
        res_dict = {}
        min_dict = {}
        for i, benchmark in enumerate(self.benchmarks):
            bench_dict = {}
            benchmark = self.parse_benchmark(benchmark)
            duplicates_name = {}
            for optimizer in self.optimizers:
                opt_dict = {}
                sols = []
                optimizer_ = self.parse_optimizer(optimizer, self.bounds[i])
                name_optimizer_ = self._get_duplicate_name(
                    bench_dict, str(optimizer_), duplicates_name
                )
                for nr in range(n_runs):
                    sol = optimizer_.minimize(benchmark)[1]
                    if verbose > 1:
                        print_blue(
                            f"Run {nr + 1} done for {name_optimizer_} on {benchmark}. Result: {sol}"
                        )
                    sols.append(sol)
                    if nr < n_runs - 1:
                        optimizer_ = self.parse_optimizer(optimizer, self.bounds[i])
                opt_dict["Approx"] = {"mean": np.mean(sols), "std": np.std(sols)}
                for metric in self.metrics:
                    metric = self.parse_metric(
                        metric, benchmark, self.bounds[i], self.options.get(metric, {})
                    )
                    m = metric(sols)
                    opt_dict[str(metric)] = m
                bench_dict[name_optimizer_] = opt_dict
                if verbose:
                    print_dark_green(f"Done for {name_optimizer_} on {benchmark}.")
            res_dict[str(benchmark)] = bench_dict
            min_dict[str(benchmark)] = benchmark.min
        if verbose:
            print_table_by_metric(res_dict)
            print_competitive_ratios(self.competitive_ratio(res_dict, min_dict))
        return res_dict
