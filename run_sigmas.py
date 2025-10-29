from gob import GOB
import inspect
import gob.benchmarks as gb
import numpy as np
from scipy.stats import invgamma


def print_avg_rank(res_dict):
    bms = list(res_dict.keys())
    optims_names = list(res_dict[bms[0]].keys())
    avg_ranks = {name: [] for name in optims_names}
    for bm in bms:
        optim_mean = []
        for optim_name in res_dict[bm]:
            optim_mean.append((optim_name, res_dict[bm][optim_name]["Approx"]["mean"]))
        optim_mean = sorted(optim_mean, key=lambda x: x[1])
        print(f"Ranks for {bm}:")
        for rank, (optim_name, mean) in enumerate(optim_mean):
            print(f"  Rank {rank + 1}: {optim_name} with mean {mean:.4f}")
            avg_ranks[optim_name].append(rank + 1)
        print("")
    print("Average ranks over all benchmarks:")
    latex_str = "Avg Ranks & "
    for optim_name in optims_names:
        avg_rank = np.mean(avg_ranks[optim_name])
        print(f"  {optim_name}: {avg_rank:.2f}")
        latex_str += f"{avg_rank:.2f} & "
    latex_str = latex_str[:-2] + " \\\\"
    print("\nLaTeX format:")
    print(latex_str)


def augment_dimensions(bounds, target_dim):
    bounds_dim0 = bounds[0]
    return np.array([bounds_dim0] * target_dim)


if __name__ == "__main__":
    benchmarks = [gb.PyGKLS(2, 50, [-100, 100], -100, smoothness="ND", gen=42)]
    bounds = [gb.create_bounds(2, -99, 99)]
    multi_modal_benchmarks = [
        gb.Ackley(),
        gb.Deb(),
        gb.Levy(),
        gb.Michalewicz(),
        gb.Rastrigin(),
        gb.Styblinskitang(),
    ]
    for bm in multi_modal_benchmarks:
        benchmarks.append(bm)
        bounds.append(bm.visual_bounds)
    multi_modal_benchmarks_names = [str(bm) for bm in multi_modal_benchmarks]

    for name, obj in inspect.getmembers(gb, inspect.isclass):
        if name != "PyGKLS":
            b = obj()
            if str(b) not in multi_modal_benchmarks_names:
                benchmarks.append(obj())
                bounds.append(benchmarks[-1].visual_bounds)

    n_particles = 150
    iter = 300
    a = 50
    sigma = lambda: invgamma.rvs(a=a, scale=a + 1)
    thetas = np.linspace(1 / 4, 2, 6)
    print(f"Thetas: {thetas}")
    optimizers = [
        ("SBS", {"n_particles": n_particles, "iter": iter, "sigma": 1 / n_particles**2})
    ] + [
        (
            "SBS-RKHS",
            {
                "n_particles": n_particles,
                "iter": iter,
                "sigma": sigma,
                "theta": theta,
            },
        )
        for theta in thetas
    ]

    gob = GOB(
        optimizers,
        benchmarks,
        ["Proportion"],
        bounds=bounds,
    )
    res_dict = gob.run(n_runs=20, verbose=1, latex_table=True)
    print_avg_rank(res_dict)
    print(res_dict)
