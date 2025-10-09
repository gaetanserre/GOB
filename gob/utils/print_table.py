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


def print_dark_green(*text):
    """
    Print text in dark green.

    Parameters
    ----------
    text : str
        The text to print.
    """
    print("\033[32m", end="")
    print(*text)
    print("\033[0m", end="")


def print_table_by_benchmark(res_dict):
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
            score = []
            for metric_name in metric_names:
                if metric_name == "Approx":
                    score.append(
                        f"{metric_dict[metric_name]['mean']:.4f} ± {metric_dict[metric_name]['std']:.4f}"
                    )
                else:
                    score.append(f"{metric_dict[metric_name]:.4f}")
            tab.add_row([opt_name] + score)
        print(tab)


def print_table_by_metric(res_dict):
    """
    Print the results of the optimization for each metric.

    Parameters
    ----------
    res_dict : dict
        The results of the optimization of the form {"Benchmark name": {"Optimizer name": {"Metric name": ...}}}
    """
    metric_names = list(list(list(res_dict.values())[0].values())[0].keys())
    print("")
    for metric_name in metric_names:
        print_purple(f"Results for {metric_name}:")
        tab = PrettyTable(["Optimizer"] + list(res_dict.keys()))
        names_opt = list(list(res_dict.values())[0].keys())
        for name_opt in names_opt:
            score = []
            for benchmark_name in res_dict:
                if metric_name == "Approx":
                    score.append(
                        f"{res_dict[benchmark_name][name_opt][metric_name]['mean']:.4f} ± {res_dict[benchmark_name][name_opt][metric_name]['std']:.4f}"
                    )
                else:
                    score.append(
                        f"{res_dict[benchmark_name][name_opt][metric_name]:.4f}"
                    )
            tab.add_row([name_opt] + score)
        print(tab)


def print_competitive_ratios(ratios):
    print_purple("Competitive ratios:")
    tab = PrettyTable(["Optimizer", "Competitive ratio"])
    for opt_name, ratio in ratios.items():
        tab.add_row([opt_name, f"{ratio:.4f}"])
    print(tab)
