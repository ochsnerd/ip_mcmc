import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

import sys
sys.path.append("/home/david/fs20/thesis/code/report/scripts/")
from helpers import (store_figure,
                     load_or_compute,
                     autocorrelation,
                     wasserstein_distance,
                     DATA_DIR)

sys.path.append("/home/david/fs20/thesis/code/report/scripts/burgers")
from rusanov import RusanovFVM


class FVMObservationOperator:
    """
    Observation operator for MCMC based on the time-evolution
    of a FVM
    """
    def __init__(self, IC, prior_means, integrator, measurement_operator):
        """
        Write requirements here

        IC: given u as argument, return callable used as initial condition for
            the integrator.
        prior_means: mean values of the prior placed on u,
            i.e. integrate IC(prior_means + u).
        integrator: given a callable as initial condition, integrate it
            and return the end-state.
        measurement_operator: given the end-state returned by the integrator,
            measure the observales.
        """
        self.IC = IC
        self.u_0 = prior_means
        self.integrator = integrator
        self.meas_op = measurement_operator

    def __call__(self, u):
        return self.meas_op(self.integrator(self.IC(self.u_0 + u)))


class PerturbedRiemannIC:
    """
    f(x) = 1 + delta_1 if x < sigma
           delta_2     if x > sigma
    """
    def __init__(self, params):
        """
        params[0]: delta_1
        params[1]: delta_2
        params[2]: sigma
        """
        self.left = 1 + params[0]
        self.right = params[1]
        self.jump = params[2]

    def __call__(self, x):
        if x < self.jump:
            return self.left
        return self.right


class RusanovMCMC:
    """
    Rusanov FVM integrator for use in the MCMC algorithm

    All parameters except the IC are set during initialization
    """
    def __init__(self, flux, flux_prime, domain, N, T):
        self.FVM = RusanovFVM(flux, flux_prime, domain, N)
        self.T = T

    def __call__(self, IC):
        # u as in "final state u", not "MCMC state u"
        u, t = self.FVM.integrate(IC, self.T)
        assert t >= self.T, "Did not reach final time during integration"
        return u


class Measurer:
    """
    Measure around points
    """
    def __init__(self, measurement_points, measurement_interval, x_values):
        # x_values: evenly spaced points where the given values are located
        self.n_meas = len(measurement_points)

        self.n_x_vals = len(x_values)
        self.dx = x_values[1] - x_values[0]
        m_p = np.asarray(measurement_points, dtype=np.float)
        self.left_limits = np.searchsorted(x_values,
                                           m_p - measurement_interval / 2,
                                           side='left')
        self.right_limits = np.searchsorted(x_values,
                                            m_p + measurement_interval / 2,
                                            side='left')

    def __call__(self, values):
        assert len(values) == self.n_x_vals, "Provided values don't match x_vals"

        m = np.empty_like(self.left_limits, dtype=np.float)
        for i in range(self.n_meas):
            left = self.left_limits[i]
            right = self.right_limits[i]
            m[i] = 10 * np.trapz(values[left:right], dx=self.dx)

        return m


class BurgersEquation:
    @staticmethod
    def flux(w):
        return .5 * w * w

    @staticmethod
    def flux_prime(w):
        return w

    @staticmethod
    def riemann_shock_pos(w_l, w_r, s_0, T):
        """Return the location of the shock.

        Return the position of the shock after time T
        of a Riemann problem with initial conditions
        w(x, 0) = w_l  if x < s_0
                  w_r  if x > s_0
        """
        assert w_l > w_r, "doesn't work for rarefactions"
        return s_0 + 0.5 * (w_r**2 - w_l**2) / (w_r - w_l) * T


def len_burn_in(x):
    """Return the index where burn-in has finished

    Burn-in is considered finished once a moving average of the
    components of u stop changing significantly"""
    def moving_avg(y, l):
        res = np.cumsum(y, dtype=float)
        res[l:] = res[l:] - res[:-l]
        return res[l - 1:] / l

    avg_window = 50
    accepted_change = 0.03

    n_vars = len(x[:, 0])
    n_avgs = len(x[0, :]) - avg_window + 1

    avgs = np.empty((n_vars, n_avgs))
    for var in range(n_vars):
        avgs[var, :] = moving_avg(x[var, :], avg_window)

    # indicate significant changes
    means = np.mean(x, axis=1)
    avgs_changed = np.empty((n_avgs - 1, ), dtype=bool)
    for i in range(0, n_avgs - 1):
        avgs_changed[i] = any(abs((avgs[v, i] - avgs[v, i + 1]) / means[v]) > accepted_change
                              for v in range(n_vars))

    # require at least avg_window + 1 consecutive significant changes
    for i in range(len(avgs_changed) - avg_window - 2, 0, -1):
        if all(avgs_changed[i: i + avg_window + 1]):
            return i

    return len(x[0, :]) - 1


def uncorrelated_sample_spacing(x):
    """Return the required spacing for samples to be uncorrelated"""
    # find length of sequence where autocorrelation becomes 0 for the first
    # time. The shorter the sequence where that happens, the better
    # (value will be more exact since we can average over more subsequences)
    tau = 10
    while (True):
        if 2 > int(len(x[0, :]) / tau):
            # never decorrelate
            return len(x)

        tau = int(tau * 1.5)
        ac = autocorrelation(x, tau)
        avg_ac = np.mean(ac, axis=0)
        idx = np.argwhere(avg_ac <= 0.001)

        if len(idx) > 0:
            return idx[0][0]


def clean_samples(x):
    """Purge burn_in and correlated samples"""
    x_ = np.copy(x)
    x_ = x_[:, len_burn_in(x_):]
    x_ = x_[:, ::uncorrelated_sample_spacing(x_)]

    return x_


def show_chain(chain, burn_in, sample_interval):
    names = ("delta_1", "delta_2", "sigma_0")

    prior_means = (1.5, 0.25, -0.5)
    prior_std_dev = 0.25

    for i in range(len(chain[0, :])):
        chain[:, i] += prior_means

    # chain-evolution
    measurement_lims = ((-0.55, -0.45),
                        (-0.3, -0.2),
                        (0.2, 0.3),
                        (0.45, 0.55),
                        (0.7, 0.8))

    for i in range(3):
        plt.plot(chain[i, :], label=names[i])

    for l, r in measurement_lims:
        plt.axhspan(l, r, facecolor='r', alpha=0.3)

    shock_locs = np.zeros_like(chain[0, :])
    rarefactions = []
    for i in range(len(shock_locs)):
        try:
            shock_locs[i] = BurgersEquation.riemann_shock_pos(chain[0, i] + 1,
                                                              chain[1, i],
                                                              chain[2, i],
                                                              1)
        except AssertionError:
            rarefactions += [i]
    plt.plot(shock_locs, color='r')
    if rarefactions:
        print(f"{len(rarefactions)} rarefactions during sampling")
        for i in rarefactions:
            plt.axvline(i, color='r', alpha=0.05)

    plt.legend()
    plt.show()

    # autocorrelation
    ac = autocorrelation(chain[:, burn_in:], 75)
    for i in range(3):
        plt.plot(ac[i, :], label=names[i])
    plt.axhline(0, color='k', linestyle='dashed')

    plt.legend()
    plt.show()

    # densities
    samples = chain[:, burn_in:]
    samples = samples[:, ::sample_interval]
    intervals = [(-2, 2)] * 3
    priors = [norm(loc=mu, scale=prior_std_dev)
              for mu in prior_means]
    fig, plts = plt.subplots(1, 3, figsize=(20, 10))

    plot_info = zip(intervals,
                    (0.025, -0.025, -0.02),
                    names,
                    priors,
                    plts)

    for i, (interval, true_val, name, prior, ax) in enumerate(plot_info):
        x_range = np.linspace(*interval, num=500)
        ax.plot(x_range, [prior.pdf(x) for x in x_range])
        ax.hist(samples[i, :], density=True, color='b')
        ax.axvline(true_val, c='r')
        ax.set_title(f"Posterior for {name}")
        ax.set(xlabel=name, ylabel="Probability")

    plt.show()

    print(f"Chain length: {len(chain[0, :])}\nUsed samples: {len(samples[0, :])}")
