import numpy as np
import matplotlib.pyplot as plt

from ip_mcmc import (MCMCSampler,
                     pCNAccepter, CountedAccepter,
                     pCNProposer,
                     EvolutionPotential,
                     GaussianDistribution)

import sys
sys.path.append("/home/david/fs20/thesis/code/report/scripts/")
from rusanov import RusanovFVM
from helpers import store_figure, load_or_compute, autocorrelation


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
        # x_values evenly spaced points where the given values are located
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


def main():
    rng = np.random.default_rng(2)

    def flux(u):
        return .5 * u*u

    def flux_prime(u):
        return u

    # simulation parameters
    domain = (-1, 1)
    N_gridpoints = 200
    T_end = 1

    # initial condition
    delta_1 = 0.025
    delta_2 = -0.025
    sigma = -0.02
    ground_truth = [delta_1, delta_2, sigma]
    true_IC = PerturbedRiemannIC(ground_truth)

    integrator = RusanovMCMC(flux, flux_prime, domain, N_gridpoints, T_end)

    # observables
    meas_points = [-0.5, -0.25, 0.25, 0.5, 0.75]
    meas_interval = 0.1

    # exclude ghost cells from x-values
    measurer = Measurer(meas_points, meas_interval, integrator.FVM.x[1:-1])

    # observe ground truth
    y = measurer(integrator(true_IC))

    # noise
    noise_beta = 0.05
    noise = GaussianDistribution(mean=np.zeros_like(y),
                                 covariance=noise_beta**2 * np.identity(len(y)))

    # prior
    prior_means = np.array([1.5, 0.25, -0.5])  # delta_1, delta_2, sigma
    gamma = 0.25
    prior_covariance = gamma**2 * np.identity(3)
    # centered prior
    prior = GaussianDistribution(np.zeros_like(prior_means), prior_covariance)

    observation_operator = FVMObservationOperator(PerturbedRiemannIC,
                                                  prior_means,
                                                  integrator,
                                                  measurer)

    potential = EvolutionPotential(observation_operator,
                                   y,
                                   noise)

    prop_beta = 0.25
    proposer = pCNProposer(beta=prop_beta, prior=prior)
    accepter = CountedAccepter(pCNAccepter(potential=potential))

    sampler = MCMCSampler(proposer, accepter, rng)

    u_0 = np.zeros(3)
    n_samples = 1100
    burn_in = 0  # manually
    sample_interval = 1

    samples_full = load_or_compute(f"burgers_samples_n={n_samples}_b={prop_beta}",
                                   sampler.run,
                                   (u_0, n_samples, burn_in, sample_interval))

    samples_full = samples_full.T
    # Add pertubations to means
    for i in range(len(samples_full[0, :])):
        samples_full[:, i] += prior_means

    # do burn_in=100 and sample_interval=5 after the fact
    samples = samples_full[:, 100:]
    samples = samples[:, ::5]

    # plot densities
    fig, plts = plt.subplots(1, 3, figsize=(20, 10))

    priors = [GaussianDistribution(mu, np.sqrt(sigma_sq))
              for mu, sigma_sq in zip(prior_means, np.diag(prior_covariance))]
    intervals = [(-2, 2)] * 3
    names = ["delta_1", "delta_2", "sigma"]

    plot_info = zip(priors,
                    intervals,
                    ground_truth,
                    names,
                    plts)

    for i, (prior, interval, true_val, name, ax) in enumerate(plot_info):
        ax.hist(samples[i, :], density=True)
        x_range = np.linspace(*interval)
        ax.plot(x_range, [prior(x) for x in x_range])
        ax.axvline(true_val, c='r')
        ax.set_title(f"Prior and posterior for {name}")
        ax.set(xlabel=name, ylabel="Probability")

    fig.suptitle("Posteriors and priors")
    store_figure(f"burgers_densities")

    # autocorrelation
    ac = autocorrelation(samples_full, 100, 10)
    for i in range(3):
        plt.plot(ac[i, :], label=names[i])
    plt.title("Autocorrelation")
    plt.xlabel("Lag")
    plt.legend()
    store_figure(f"burgers_ac_b={prop_beta}")

    show_chain_evolution(samples_full, prop_beta, integrator, measurer, names, ground_truth)
    show_setup(true_IC, integrator, measurer)


def show_chain_evolution(samples, prop_beta, integrator, measurer, names, ground_truth):
    def shock_location(d1, d2, s):
        return s + 0.5 * (d2**2 - d1**2 - 2*d1 - 1) / (d1 + d2 - 1)

    x_vals = integrator.FVM.x[1:-1]
    measurement_lims = zip(measurer.left_limits, measurer.right_limits)
    for l_idx, r_idx in measurement_lims:
        plt.axhspan(x_vals[l_idx], x_vals[r_idx], facecolor='r', alpha=0.3)

    for a in ground_truth:
        plt.axhline(a, color='k')

    for i in range(3):
        plt.plot(samples[i, :], label=names[i])

    shock_locs = [shock_location(*samples[:, i]) for i in range(len(samples[0,:]))]
    plt.plot(shock_locs, label="Shock location")
    plt.ylim(-2, 2)
    plt.title("Chain evolution")
    plt.legend()
    store_figure(f"burgers_chain_b={prop_beta}")


def show_setup(IC, integrator, measurer):
    unperturbed_IC = PerturbedRiemannIC([0] * 3)

    x_vals = integrator.FVM.x[1:-1]

    unperturbed_u_start = [unperturbed_IC(x) for x in x_vals]
    perturbed_u_start = [IC(x) for x in x_vals]
    unperturbed_u_end = integrator(unperturbed_IC)
    perturbed_u_end = integrator(IC)

    measurement_lims = zip(measurer.left_limits, measurer.right_limits)

    plt.plot(x_vals, unperturbed_u_end, 'k--', label="0,1 Riemann problem")
    plt.plot(x_vals, unperturbed_u_start, 'k--')
    plt.plot(x_vals, perturbed_u_end, 'b')
    plt.plot(x_vals, perturbed_u_start, 'b', label="ground truth for MCMC")
    for left_idx, right_idx in measurement_lims:
        plt.axvspan(x_vals[left_idx], x_vals[right_idx], facecolor='g', alpha=0.5)

    plt.legend()
    plt.title("Setup, at T = 0 and T = 1")
    plt.xlabel("x")
    plt.ylabel("u")

    store_figure("burgers_setup")


if __name__ == '__main__':
    main()
