#
# Created in 2024 by Gaëtan Serré
#

from prettytable.colortable import ColorTable, Themes
from prettytable import VRuleStyle
import numpy as np
from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu


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


def _significancy(res_dict_benchmark, best_optim_name):
    sols = res_dict_benchmark[best_optim_name]["Approx"]["all"]
    p_values = []
    for optim_name, optim_dict in res_dict_benchmark.items():
        if optim_name != best_optim_name:
            sols_other = optim_dict["Approx"]["all"]
            _, p_value = mannwhitneyu(sols, sols_other, alternative="two-sided")
            p_values.append(p_value)
    pvals_corr = multipletests(p_values, method="holm")[1]
    return (np.array(pvals_corr) < 0.05).all(), np.max(pvals_corr)


def format_latex_table(tab_string):
    lines = tab_string.splitlines()
    # Remove first and last vlines
    fst_line = list(lines[0])
    fst_line[-2] = ""
    # \begin{tabular}{| -> 16
    fst_line[16] = ""
    lines[0] = "".join(fst_line)

    # Add thick hlines
    lines[1] = r"\Xhline{2\arrayrulewidth}"
    lines.insert(3, r"\Xhline{2\arrayrulewidth}")
    lines[-2] = r"\Xhline{2\arrayrulewidth}"
    return "\n".join(lines)


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
        tab.add_column(r"\textbf{Benchmark}", list(res_dict.keys()))
        names_opt = list(list(res_dict.values())[0].keys())
        p_values = {}
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
                    """ std = transform_number(
                        res_dict[benchmark_name][name_opt][metric_name]["std"]
                    ) """
                    if (
                        res_dict[benchmark_name][name_opt][metric_name]["mean"]
                        == best_mean
                    ):
                        mean = f"\\mathbf{{{mean}}}"
                        if benchmark_name in p_values:
                            significancy = True
                        else:
                            count_best = sum(1 for m in means if m == best_mean)
                            if count_best > 1:
                                significancy = True
                                p_values[benchmark_name] = "N/A"
                            else:
                                significancy, p_val = _significancy(
                                    res_dict[benchmark_name], name_opt
                                )
                                p_values[benchmark_name] = f"${p_val:.3f}$"
                        if significancy:
                            color = r"\cellcolor{cell-gray}"
                            mean = f"{color} {mean}"
                    score.append(f"${mean}$")
                else:
                    score.append(
                        f"{res_dict[benchmark_name][name_opt][metric_name]:.4f}"
                    )
            tab.add_column(r"\textbf{" + name_opt + "}", score)
        if metric_name == "Approx":
            print(p_values)
            p_values = [p_values[bm] for bm in res_dict]
            tab.add_column(r"\textbf{p-value}", p_values)

        latex_table = tab.get_formatted_string(
            "latex", vrules=VRuleStyle.ALL, border=True, format=True
        )
        print(format_latex_table(latex_table))


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
