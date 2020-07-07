import numpy as np
import matplotlib.pyplot as plt

from ip_mcmc import (MCMCSampler,
                     pCNAccepter, CountedAccepter,
                     ConstSteppCNProposer,
                     EvolutionPotential,
                     GaussianDistribution,
                     ConstStepStandardRWProposer,
                     VarStepStandardRWProposer,
                     StandardRWAccepter)

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
            names = ["delta_1", "delta_2", "sigma_0"]
            delta_1 = 0.025
            delta_2 = -0.025
            sigma = -0.02
            ground_truth = [delta_1, delta_2, sigma]

        domain = (-1, 1)
        N_gridpoints = 128
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
        step = 0.25
        u_0 = np.zeros(3)
        N = 5000
        burn_in = 250
        sample_interval = 20
        rng = np.random.default_rng(2)

    @staticmethod
    def filename():
        return f"burgers_RW_n={Settings.Sampling.N}_b={Settings.Sampling.step}"


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
    # Proposer
    prior = Settings.Prior.get_distribution()
    # proposer = ConstSteppCNProposer(Settings.Sampling.step, prior)
    # proposer = ConstStepStandardRWProposer(Settings.Sampling.step, prior)
    proposer = VarStepStandardRWProposer(Settings.Sampling.step, prior)

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
    # accepter = CountedAccepter(pCNAccepter(potential))
    accepter = CountedAccepter(StandardRWAccepter(potential, prior))

    return MCMCSampler(proposer, accepter, Settings.Sampling.rng)


class PWLinear:
    """linearly decrease delta until burn_in is finished, then keep it constant"""
    def __init__(self, start_delta, end_delta, len_burn_in):
        self.d_s = start_delta
        self.d_e = end_delta
        self.l = len_burn_in

        self.slope = (start_delta - end_delta) / len_burn_in

    def __call__(self, i):
        if i > self.l:
            return self.d_e
        return self.d_s - self.slope * i

    def __repr__(self):
        """For filename"""
        return f"pwl_{self.d_s}_{self.d_e}_{self.l}"


def main():
    stepsize = PWLinear(0.1, 0.001, Settings.Sampling.burn_in)
    steps = [stepsize]
    burn_ins = [Settings.Sampling.burn_in]
    sample_intervals = [Settings.Sampling.sample_interval]

    for step, burn_in, sample_interval in zip(steps, burn_ins, sample_intervals):
        Settings.Sampling.step = step
        Settings.Sampling.burn_in = burn_in
        Settings.Sampling.sample_interval = sample_interval

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

        # samples = clean_samples(samples_full)
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
        store_figure(Settings.filename() + "_densities")

        # autocorrelation
        # samples_burned = samples_full[:, len_burn_in(samples_full):]
        ac = autocorrelation(samples_full[:, Settings.Sampling.burn_in:], 100)
        for i in range(3):
            plt.plot(ac[i, :], label=Settings.Simulation.IC.names[i])

        plt.axhline(0, color='k', alpha=0.5)

        plt.title("Autocorrelation")
        plt.xlabel("Lag")
        plt.legend()
        store_figure(Settings.filename() + "_ac")

        show_chain_evolution_and_step(samples_full)


def show_chain_evolution(samples):
    # measurement indicators
    x_vals = Settings.Simulation.get_xvals()
    measurer = create_measurer()
    measurement_lims = zip(measurer.left_limits, measurer.right_limits)
    for l_idx, r_idx in measurement_lims:
        plt.axhspan(x_vals[l_idx], x_vals[r_idx], facecolor='r', alpha=0.3)

    # ground truth lines
    for a in Settings.Simulation.IC.ground_truth:
        plt.axhline(a, color='k')

    # actual chain values
    for i in range(3):
        plt.plot(samples[i, :], label=Settings.Simulation.IC.names[i])

    # shock locations
    shock_locs = [BurgersEquation.riemann_shock_pos(samples[0, i] + 1,
                                                    samples[1, i],
                                                    samples[2, i],
                                                    Settings.Simulation.T_end)
                  for i in range(len(samples[0, :]))]
    plt.plot(shock_locs, label="Shock location")

    # computed characteristics
    # l_b = len_burn_in(samples)
    # if l_b == len(samples[0, :]):
    #     tau = "burn-in not finished"
    # else:
    #     tau = uncorrelated_sample_spacing(samples[:, l_b:])
    # plt.text(0, 1.6, f"burn-in: {l_b}\ndecorrelation-length: {tau}")

    # characteristics
    plt.text(0, 1.6, (f"burn-in: {Settings.Sampling.burn_in}\n"
                      f"sampling-interval: {Settings.Sampling.sample_interval}\n"
                      f"usable samples: {(Settings.Sampling.N - Settings.Sampling.burn_in)/Settings.Sampling.sample_interval}"))

    plt.ylim(-1, 1.8)
    plt.title("Chain evolution")
    plt.legend()
    store_figure(Settings.filename() + "_chain")


def show_chain_evolution_and_step(samples):
    fig, (parameters, shock, steps) = plt.subplots(3, 1, figsize=(15, 10),
                                                   gridspec_kw={'height_ratios': [3, 3, 1]})

    # ground truth lines
    for a in Settings.Simulation.IC.ground_truth:
        parameters.axhline(a, linestyle='dashed', color='k')

    # actual chain values
    for i in range(3):
        parameters.plot(samples[i, :], label=Settings.Simulation.IC.names[i])
    parameters.set_ylim(-.8, 1.6)
    parameters.set_xlim(0, Settings.Sampling.N)
    parameters.legend()
    parameters.set_title("Parameters")

    # measurement indicators
    x_vals = Settings.Simulation.get_xvals()
    measurer = create_measurer()
    measurement_lims = zip(measurer.left_limits, measurer.right_limits)
    for l_idx, r_idx in measurement_lims:
        shock.axhspan(x_vals[l_idx], x_vals[r_idx], facecolor='r', alpha=0.3)

    # shock locations
    shock_locs = [BurgersEquation.riemann_shock_pos(samples[0, i] + 1,
                                                    samples[1, i],
                                                    samples[2, i],
                                                    Settings.Simulation.T_end)
                  for i in range(len(samples[0, :]))]
    shock.plot(shock_locs, color='r')
    shock.set_ylim(Settings.Simulation.domain)
    shock.set_xlim(0, Settings.Sampling.N)
    shock.set_title("Shock locations and measurement intervals")

    # beta evolution
    step_vals = np.array([Settings.Sampling.step(i) for i in range(len(samples[0, :]))])
    steps.plot(step_vals, color='k')
    steps.set_ylim(0, Settings.Sampling.step.d_s)
    steps.set_xlim(0, Settings.Sampling.N)
    steps.set_title("Step size")

    fig.suptitle("Chain evolution")
    store_figure(Settings.filename() + "_chain_beta")


if __name__ == '__main__':
    main()
