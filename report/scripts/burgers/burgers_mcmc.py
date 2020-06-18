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
                       BurgersEquation,
                       autocorrelation)


class Settings:
    """'Static' class to collect the settings for a MCMC simulation"""
    # 'Attributes' that derive from other attributes need to be impleneted
    # using a getter-method, so that they get updated when the thing
    # they depend on changes.
    class Simulation:
        class IC:
            names = ["delta_1", "delta_2", "sigma"]
            delta_1 = 0.025
            delta_2 = -0.025
            sigma = -0.02
            ground_truth = [delta_1, delta_2, sigma]

        domain = (-1, 1)
        N_gridpoints = 200
        T_end = 1
        flux = BurgersEquation.flux
        flux_prime = BurgersEquation.flux_prime

        @staticmethod
        def get_xvals():
            return RusanovMCMC(None, None,
                               Settings.Simulation.domain,
                               Settings.Simulation.N_gridpoints,
                               0).FVM.x[1:-1]

    class Measurement:
        points = [-0.5, -0.25, 0.25, 0.5, 0.75]
        interval = 0.1

    class Noise:
        mean = np.array([0] * 5)
        std_dev = 0.05
        covariance = std_dev**2 * np.identity(5)

        @staticmethod
        def get_distribution():
            return GaussianDistribution(Settings.Noise.mean,
                                        Settings.Noise.covariance)

    class Prior:
        mean = np.array([1.5,    # delta_1
                         0.25,   # delta_2
                         -0.5])  # sigma
        std_dev = 0.25
        covariance = std_dev**2 * np.identity(len(mean))

        @staticmethod
        def get_distribution():
            return GaussianDistribution(Settings.Prior.mean,
                                        Settings.Prior.covariance)

    class Sampling:
        beta = 0.25
        u_0 = np.zeros(3)
        N = 1100
        burn_in = 100
        sample_interval = 5

    @staticmethod
    def filename():
        return f"burgers_n={Settings.Sampling.N}_b={Settings.Sampling.beta}"


def create_integrator():
    return RusanovMCMC(Settings.Simulation.flux,
                       Settings.Simulation.flux_prime,
                       Settings.Simulation.domain,
                       Settings.Simulation.N_gridpoints,
                       Settings.Simulation.T_end)


def create_measurer():
    return Measurer(Settings.Measurement.points,
                    Settings.Measurement.interval,
                    Settings.Simulation.get_xvals())


def create_mcmc_sampler():
    rng = np.random.default_rng(2)

    # Proposer
    prior = Settings.Prior.get_distribution()
    proposer = pCNProposer(Settings.Sampling.beta, prior)

    # Accepter
    integrator = create_integrator()
    measurer = create_measurer()
    IC_true = PerturbedRiemannIC(Settings.Simulation.IC.ground_truth)
    observation_operator = FVMObservationOperator(PerturbedRiemannIC,
                                                  Settings.Prior.mean,
                                                  integrator,
                                                  measurer)

    ground_truth = measurer(integrator(IC_true))
    noise = Settings.Noise.get_distribution()
    potential = EvolutionPotential(observation_operator,
                                   ground_truth,
                                   noise)
    accepter = CountedAccepter(pCNAccepter(potential))

    return MCMCSampler(proposer, accepter, rng)


def main():
    sampler = create_mcmc_sampler()

    samples_full = load_or_compute(Settings.filename(),
                                   sampler.run,
                                   (Settings.Sampling.u_0,
                                    Settings.Sampling.N,
                                    0,
                                    1))

    samples_full = samples_full.T
    # Add pertubations to means
    for i in range(len(samples_full[0, :])):
        samples_full[:, i] += Settings.Prior.mean

    # do burn_in and sample_interval after the fact
    samples = samples_full[:, Settings.Sampling.burn_in:]
    samples = samples[:, ::Settings.Sampling.sample_interval]

    # plot densities
    fig, plts = plt.subplots(1, 3, figsize=(20, 10))

    priors = [GaussianDistribution(mu, Settings.Prior.std_dev)
              for mu in Settings.Prior.mean]

    intervals = [(-2, 2)] * 3

    plot_info = zip(priors,
                    intervals,
                    Settings.Simulation.IC.ground_truth,
                    Settings.Simulation.IC.names,
                    plts)

    for i, (prior, interval, true_val, name, ax) in enumerate(plot_info):
        ax.hist(samples[i, :], density=True)
        x_range = np.linspace(*interval, num=300)
        ax.plot(x_range, [prior(x) for x in x_range])
        ax.axvline(true_val, c='r')
        ax.set_title(f"Prior and posterior for {name}")
        ax.set(xlabel=name, ylabel="Probability")

    fig.suptitle("Posteriors and priors")
    store_figure(f"burgers_densities")

    # autocorrelation
    ac = autocorrelation(samples_full, int(Settings.Sampling.N / 10), 10)
    for i in range(3):
        plt.plot(ac[i, :], label=Settings.Simulation.IC.names[i])
    plt.title("Autocorrelation")
    plt.xlabel("Lag")
    plt.legend()
    store_figure(Settings.filename() + "_ac")

    show_chain_evolution(samples_full)
    show_setup()


def show_chain_evolution(samples):
    def shock_location(d1, d2, s):
        return BurgersEquation.riemann_shock_pos(1 + d1, d2, s, Settings.Simulation.T_end)

    x_vals = Settings.Simulation.get_xvals()
    measurer = create_measurer()
    measurement_lims = zip(measurer.left_limits, measurer.right_limits)
    for l_idx, r_idx in measurement_lims:
        plt.axhspan(x_vals[l_idx], x_vals[r_idx], facecolor='r', alpha=0.3)

    for a in Settings.Simulation.IC.ground_truth:
        plt.axhline(a, color='k')

    for i in range(3):
        plt.plot(samples[i, :], label=Settings.Simulation.IC.names[i])

    shock_locs = [shock_location(*samples[:, i]) for i in range(len(samples[0, :]))]
    plt.plot(shock_locs, label="Shock location")
    plt.ylim(-2, 2)
    plt.title("Chain evolution")
    plt.legend()
    store_figure(Settings.filename() + "_chain")


def show_setup():
    integrator = create_integrator()
    actual_IC = PerturbedRiemannIC(Settings.Simulation.IC.ground_truth)
    unperturbed_IC = PerturbedRiemannIC([0] * 3)

    x_vals = Settings.Simulation.get_xvals()

    unperturbed_u_start = [unperturbed_IC(x) for x in x_vals]
    perturbed_u_start = [actual_IC(x) for x in x_vals]
    unperturbed_u_end = integrator(unperturbed_IC)
    perturbed_u_end = integrator(actual_IC)

    measurer = create_measurer()
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
    plt.ylabel("w")

    store_figure("burgers_setup")


if __name__ == '__main__':
    main()
