import ot  # Wasserstein distance

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


def autocorrelation(samples, tau_max):
    avg_over = int(len(samples[0, :]) / tau_max)
    assert avg_over > 0, ("Not enough samples to compute autocorrelation"
                          "with specified length")

    n_vars = len(samples[:, 0])
    ac = np.zeros((n_vars, tau_max))
    for i in range(avg_over):
        for var in range(n_vars):
            vals = samples[var, i*tau_max:(i+1)*tau_max]
            ac[var, :] += MCMCSampler.autocorr(vals)
    ac /= avg_over

    return ac


def wasserstein_distance(d1, d2, intervals):
    """Compute the Wasserstein-distance between d1, d2

    d1 and d2 are densities (assumed to be normalized)
    supported on the given intervals
    """
    assert d1.shape == d2.shape, ""
    assert len(d1.shape) == intervals.shape[0], ""
    assert intervals.shape[1] == 2, ""

    # create metric
    nd = len(d1.shape)
    ax_ticks = np.meshgrid(*[np.linspace(start=intervals[i, 0],
                                         stop=intervals[i, 1],
                                         num=d1.shape[i])
                             for i in range(nd)], indexing='ij')

    grid = np.stack([ticks.flatten() for ticks in ax_ticks], axis=1)
    M = ot.dist(grid, grid, metric='euclidean')

    return ot.emd2(d1.flatten(), d2.flatten(), M)


def test_wasserstein_distance():
    # two 1D histograms with 5 bins
    n = 5
    x_1 = np.arange(n).reshape((n, 1))
    M1 = ot.dist(x_1, x_1, metric='euclidean')

    # two delta peaks
    d_11 = [1,0,0,0,0]
    d_12 = [0,0,1,0,0]
    assert ot.emd2(d_11, d_12, M1) == 2, ""

    # two 2D histograms
    n1, n2 = (5, 8)
    xv, yv = np.meshgrid(np.arange(n1), np.arange(n2), indexing='ij')
    grid = np.stack((xv.flatten(), yv.flatten()), axis=1)
    M = ot.dist(grid, grid, metric='euclidean')

    # from all corners to bottom left
    d_21 = np.zeros((n1, n2))
    d_21[0,0] = .25
    d_21[0, -1] = .25
    d_21[-1, 0] = .25
    d_21[-1, -1] = .25
    d_22 = np.zeros((n1, n2))
    d_22[0, -1] = 1

    assert np.isclose(ot.emd2(d_21.flatten(), d_22.flatten(), M),
                      ((n1-1) + (n2-1) + np.sqrt((n1-1)**2 + (n2-1)**2))/4), ""

    # using my function
    n1, n2, n3 = (3, 4, 6)
    intervals = np.array([[-1, 1],  # nice integer spacing
                          [0, 3],
                          [-5, 0]])

    d_31 = np.zeros((n1, n2, n3))
    d_31[0,0,0] = 1
    d_32 = np.zeros((n1, n2, n3))
    d_32[2,2,2] = 1

    assert np.isclose(wasserstein_distance(d_31, d_32, intervals), np.sqrt(12)), ""


test_wasserstein_distance()
