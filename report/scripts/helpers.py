import numpy as np
import matplotlib.pyplot as plt

from ip_mcmc import MCMCSampler


FIGURE_DIR = "/home/david/fs20/thesis/code/report/figures/"
DATA_DIR = "/home/david/fs20/thesis/code/report/data/"


def store_figure(name):
    """
    Store a figure in the figures directory.
    Assumes there is an active pyplot-Plot and clears it after
    """
    plt.savefig(FIGURE_DIR + name + ".svg", format='svg')
    plt.clf()
    print("Saved figure", name)


def load_or_compute(name, function, args):
    """
    Attempt to load data, if not possible compute it.

    Attempt to np.load name, if not found use function(*args)
    to compute it, save the computed values under name and
    return them.
    """
    try:
        res = np.load(DATA_DIR + name + ".npy")
        print("Loaded existing results for", name)
    except FileNotFoundError:
        print("Computing", name)
        res = function(*args)
        np.save(DATA_DIR + name, res)
    return res


def autocorrelation(samples, tau_max, avg_over):
    assert tau_max * avg_over <= len(samples[0, :]), (
        "Not enough samples to average {tau_max=} over {avg_over} subseries")

    n_vars = len(samples[:, 0])
    ac = np.zeros((n_vars, tau_max))
    for i in range(avg_over):
        for var in range(n_vars):
            vals = samples[var, i*tau_max:(i+1)*tau_max]
            ac[var, :] += MCMCSampler.autocorr(vals)
    ac /= avg_over

    return ac
