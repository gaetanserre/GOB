#
# Created in 2024 by Gaëtan Serré
#

from prettytable import PrettyTable


def print_purple(*text):
    """
    Print text in purple.

    Parameters
    ----------
    text : str
        The text to print.
    """
    print("\033[95m", end="")
    print(*text)
    print("\033[0m", end="")


def print_blue(*text):
    """
    Print text in blue.

    Parameters
    ----------
    text : str
        The text to print.
    """
    print("\033[94m", end="")
    print(*text)
    print("\033[0m", end="")


def print_table_per_benchmark(res_dict):
    """
    Print the results of the optimization for each benchmark.

    Parameters
    ----------
    res_dict : dict
        The results of the optimization of the form {"Benchmark name": {"Optimizer name": {"Metric name": ...}}}
    """
    print("")
    for benchmark_name, optim_res in res_dict.items():
        print_purple(f"Results for {benchmark_name}:")
        metric_names = list(list(optim_res.values())[0].keys())
        tab = PrettyTable(["Optimizer"] + metric_names)
        for opt_name, metric_dict in optim_res.items():
            tab.add_row(
                [opt_name] + [metric_dict[metric_name] for metric_name in metric_names]
            )
        print(tab)

        """ tab = PrettyTable(["Optimizer", "Metric", "Value"])
        for opt_name, metric_dict in optim_res.items():
            for metric_name, metric_value in metric_dict.items():
                tab.add_row([opt_name, metric_name, metric_value])
        print(tab) """
