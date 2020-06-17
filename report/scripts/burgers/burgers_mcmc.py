import numpy as np
import matplotlib.pyplot as plt

from ip_mcmc import (MCMCSampler,
                     pCNAccepter, CountedAccepter,
                     pCNProposer,
                     EvolutionPotential,
                     GaussianDistribution)

import sys
sys.path.append("/home/david/fs20/thesis/code/report/scripts/burgers")
from utilities import (FVMObservationOperator,
                       PerturbedRiemannIC,
                       RusanovMCMC,
                       Measurer,
                       store_figure,
                       load_or_compute,
                       autocorrelation)


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
