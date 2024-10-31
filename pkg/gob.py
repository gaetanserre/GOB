#
# Created in 2024 by Gaëtan Serré
#
import numpy as np
from .benchmarks import Square
from .optimizers import PRS
from .metrics import Proportion
from .create_bounds import create_bounds


class GOB:
    """
    Global Optimization Benchmarks.
    """

    def __init__(
        self, optimizer, benchmark, metric, bounds=create_bounds(-1, 1, 2), options={}
    ):
        """
        Initialize the benchmarking tool.

        Parameters
        ----------
        optimizer : str | Class
            The optimizer to use.

        benchmark : str | Class
            The benchmark to use.

        metric : str | Class
            The metric to use.

        bounds : array_like of shape (n_variables, 2)
            The bounds of the search space.

        **kwargs : dict of keyword arguments
            {name_optimizer: dict of keyword arguments}
        """

        self.bounds = bounds

        if isinstance(optimizer, str):
            self.optimizer = self.parse_optimizer(optimizer, options.get(optimizer, {}))
        else:
            self.optimizer = optimizer

        if isinstance(benchmark, str):
            self.benchmark = self.parse_benchmark(benchmark)
        else:
            self.benchmark = benchmark

        if isinstance(metric, str):
            self.metric = self.parse_metric(metric)
        else:
            self.metric = metric

    def parse_optimizer(self, optimizer, options={}):
        """
        Parse the optimizer.

        Parameters
        ----------
        optimizer : str
            The optimizer to use.

        **kwargs : dict of keyword arguments
            {name_optimizer: dict of keyword arguments}

        Returns
        -------
        Optimizer
            An instance of the optimizer.
        """
        match optimizer:
            case "PRS":
                return PRS(bounds=self.bounds, **options)
            case _:
                raise ValueError(f"Unknown optimizer: {optimizer}")

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
        Function
            An instance of the benchmark.
        """
        match benchmark:
            case "Square":
                return Square()
            case _:
                raise ValueError(f"Unknown benchmark: {benchmark}")

    def parse_metric(self, metric):
        """
        Parse the metric.

        Parameters
        ----------
        metric : str
            The metric to use.

        Returns
        -------
        Metric
            An instance of the metric.
        """
        match metric:
            case "Proportion":
                return Proportion(self.benchmark, self.bounds, 0.99)
            case _:
                raise ValueError(f"Unknown metric: {metric}")

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
        sols = []
        for _ in range(n_runs):
            sol = self.optimizer.minimize(self.benchmark)
            sols.append(sol)

        mean, std = np.mean(sols), np.std(sols)
        if verbose:
            if n_runs == 1:
                print(f"Minimum: {sols[0]:.6f}, True minimum: {self.benchmark.min}")
            else:
                print(
                    f"Minimum: {mean:.4f} ± {std:.4f}, True minimum: {self.benchmark.min}"
                )
        metric = self.metric(sols)
        print(metric)
        return sols
