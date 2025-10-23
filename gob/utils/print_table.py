#
# Created in 2024 by Gaëtan Serré
#

from prettytable.colortable import ColorTable, Themes
import numpy as np


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


def transform_power_of_ten(v):
    n = 0
    while np.abs(v * 10**n) <= 1:
        n += 1
    return f"{int(v * 10**n)}e^{{-{n}}}"


def transform_number(v):
    if np.abs(v) >= 0.001:
        return f"{v:.3f}"
    elif 0 < np.abs(v):
        return f"{transform_power_of_ten(v)}"
    else:
        return "0"


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
        tab = ColorTable(["Optimizer"] + metric_names, theme=Themes.LAVENDER)
        for opt_name, metric_dict in optim_res.items():
            score = []
            for metric_name in metric_names:
                if metric_name == "Approx":
                    mean = transform_number(metric_dict[metric_name]["mean"])
                    std = transform_number(metric_dict[metric_name]["std"])
                    score.append(f"${mean} \\pm {std}$")
                else:
                    score.append(f"{metric_dict[metric_name]:.4f}")
            tab.add_row([opt_name] + score)
        print(tab)


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
        tab = ColorTable(["Optimizer"] + metric_names, theme=Themes.LAVENDER)
        for opt_name, metric_dict in optim_res.items():
            score = []
            for metric_name in metric_names:
                if metric_name == "Approx":
                    mean = transform_number(metric_dict[metric_name]["mean"])
                    std = transform_number(metric_dict[metric_name]["std"])
                    score.append(f"${mean} \\pm {std}$")
                else:
                    score.append(f"{metric_dict[metric_name]:.4f}")
            tab.add_row([opt_name] + score)
        print(tab)


def print_table_by_metric_latex(res_dict):
    """
    Print the results of the optimization for each metric in LaTeX format.

    Parameters
    ----------
    res_dict : dict
        The results of the optimization of the form {"Benchmark name": {"Optimizer name": {"Metric name": ...}}}
    """
    metric_names = list(list(list(res_dict.values())[0].values())[0].keys())
    print("")
    for metric_name in metric_names:
        print_purple(f"Results for {metric_name}:")
        tab = ColorTable(theme=Themes.LAVENDER)
        tab.add_column("Benchmark", list(res_dict.keys()))
        names_opt = list(list(res_dict.values())[0].keys())
        for name_opt in names_opt:
            score = []
            for benchmark_name in res_dict:
                if metric_name == "Approx":
                    means = [
                        res_dict[benchmark_name][name_opt_]["Approx"]["mean"]
                        for name_opt_ in names_opt
                    ]
                    best_mean = min(means)
                    mean = transform_number(
                        res_dict[benchmark_name][name_opt][metric_name]["mean"]
                    )
                    std = transform_number(
                        res_dict[benchmark_name][name_opt][metric_name]["std"]
                    )
                    if (
                        res_dict[benchmark_name][name_opt][metric_name]["mean"]
                        == best_mean
                    ):
                        mean = f"\\mathbf{{{mean}}}"
                    score.append(f"${mean} \\pm {std}$")
                else:
                    score.append(
                        f"{res_dict[benchmark_name][name_opt][metric_name]:.4f}"
                    )
            tab.add_column(name_opt, score)
        print(tab.get_formatted_string("latex"))


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
        tab = ColorTable(theme=Themes.LAVENDER)
        tab.add_column("Benchmark", list(res_dict.keys()))
        names_opt = list(list(res_dict.values())[0].keys())
        for name_opt in names_opt:
            score = []
            for benchmark_name in res_dict:
                if metric_name == "Approx":
                    mean = transform_number(
                        res_dict[benchmark_name][name_opt][metric_name]["mean"]
                    )
                    std = transform_number(
                        res_dict[benchmark_name][name_opt][metric_name]["std"]
                    )
                    score.append(f"{mean} ± {std}")
                else:
                    score.append(
                        f"{res_dict[benchmark_name][name_opt][metric_name]:.4f}"
                    )
            tab.add_column(name_opt, score)
        print(tab)


def print_competitive_ratios(ratios):
    print_purple("Competitive ratios:")
    tab = ColorTable(["Optimizer", "Competitive ratio"], theme=Themes.LAVENDER)
    for opt_name, ratio in ratios.items():
        tab.add_row([opt_name, f"{ratio:.4f}"])
    print(tab)
