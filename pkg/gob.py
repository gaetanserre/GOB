#
# Created in 2024 by Gaëtan Serré
#
import numpy as np
from .create_bounds import create_bounds

from .benchmarks import Square
from .benchmarks import Ackley

from .metrics import Proportion

from .utils import print_table_by_metric
from .utils import print_competitive_ratios
from .utils import print_blue


class GOB:
    """
    Global Optimization Benchmarks.
    """

    def __init__(self, optimizers, benchmarks, metrics, bounds=None, options={}):
        """
        Initialize the benchmarking tool.

        Parameters
        ----------
        optimizers : List str | Object
            The optimizers to use.

        benchmarks : List str | Object
            The benchmarks to use.

        metrics : List str | Object
            The metrics to use.

        bounds : array_like of shape (n_benchmark, n_variables, 2)
            The bounds of the search space.

        **kwargs : dict of keyword arguments for the optimizers or the metrics
            {name_optimizer: dict of keyword arguments}
        """
        if bounds is None:
            bounds = create_bounds(len(benchmarks), -1, 1, 2)
        self.bounds = bounds

        self.options = options
        self.optimizers = optimizers
        self.benchmarks = benchmarks
        self.metrics = metrics

    def parse_optimizer(self, optimizer, bounds, options={}):
        """
        Parse the optimizer.

        Parameters
        ----------
        optimizer : str | Object
            The optimizer to use.
        bounds : array_like of shape (n_variables, 2)
            The bounds of the search space.

        **kwargs : dict of keyword arguments
            {name_optimizer: dict of keyword arguments}

        Returns
        -------
        Optimizer
            Instance of the optimizer.
        """
        if isinstance(optimizer, str):
            match optimizer:
                case "PRS":
                    from .optimizers import PRS

                    return PRS(bounds=bounds, **options)
                case "GD":
                    from .optimizers import GD

                    return GD(bounds=bounds, **options)
                case "CMA-ES":
                    from .optimizers import CMA_ES

                    return CMA_ES(bounds=bounds, **options)

                case "AdaLIPO_P":
                    from .optimizers import AdaLIPO_P

                    return AdaLIPO_P(bounds=bounds, **options)

                case "SBS":
                    from .optimizers import SBS

                    return SBS(bounds=bounds, **options)

                case _:
                    raise ValueError(f"Unknown optimizer: {optimizer}")
        else:
            return optimizer

    @staticmethod
    def parse_benchmark(benchmark):
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
            match benchmark:
                case "Square":
                    return Square()
                case "Ackley":
                    return Ackley()
                case _:
                    raise ValueError(f"Unknown benchmark: {benchmark}")
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
        bounds : array_like of shape (n_variables, 2)
            The bounds of the search space.

        Returns
        -------
        List Metric
            Instance of the metric.
        """
        if isinstance(metric, str):
            match metric:
                case "Proportion":
                    return Proportion(benchmark, bounds, **options)
                case _:
                    raise ValueError(f"Unknown metric: {metric}")
        else:
            return metric

    @staticmethod
    def print_approx(sols, f, n_runs):
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

    def competitive_ratio(self, res_dict):
        """
        Compute the competitive ratio: 1 / |F| * sum_{f in F} (approx(f) / min(f)).

        Parameters
        ----------
        res_dict : dict
            The results dictionary.
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
                approxs[bench_name] = bench_dict[optimizer_name]["Approx"]["mean"]
        for optimizer_name in optimizer_names:
            ratio = 0
            for bench_name, bench_dict in res_dict.items():
                approx = bench_dict[optimizer_name]["Approx"]["mean"]
                best = np.min(approxs[bench_name])
                ratio += min(100, (approx + 1e-10) / (best + 1e-10))
            ratios[optimizer_name] = ratio / len(self.benchmarks)
        return ratios

    def run(self, n_runs=1, verbose=False):
        """
        Run the benchmark.

        Parameters
        ----------
        n_runs : int
            The number of runs to perform.
        verbose : bool
            Whether to print the results.
        """
        res_dict = {}
        for i, benchmark in enumerate(self.benchmarks):
            bench_dict = {}
            benchmark = self.parse_benchmark(benchmark)
            for optimizer in self.optimizers:
                opt_dict = {}
                optimizer = self.parse_optimizer(
                    optimizer, self.bounds[i], self.options.get(optimizer, {})
                )
                sols = []
                for _ in range(n_runs):
                    sol = optimizer.minimize(benchmark)
                    sols.append(sol)
                opt_dict["Approx"] = {"mean": np.mean(sols), "std": np.std(sols)}
                for metric in self.metrics:
                    metric = self.parse_metric(
                        metric, benchmark, self.bounds[i], self.options.get(metric, {})
                    )
                    m = metric(sols)
                    opt_dict[str(metric)] = m
                bench_dict[str(optimizer)] = opt_dict
            res_dict[str(benchmark)] = bench_dict
            print_blue(f"Done for {benchmark}.")
        if verbose:
            print_table_by_metric(res_dict)
            print_competitive_ratios(self.competitive_ratio(res_dict))
