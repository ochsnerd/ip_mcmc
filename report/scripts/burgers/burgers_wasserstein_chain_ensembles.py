import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from scipy.stats import norm

from ip_mcmc import (EnsembleManager,
                     MCMCSampler,
                     pCNAccepter,
                     StandardRWAccepter,
                     ConstrainAccepter,
                     CountedAccepter,
                     ConstSteppCNProposer,
                     EvolutionPotential,
                     GaussianDistribution,
                     ConstStepStandardRWProposer,
                     VarStepStandardRWProposer)

import sys
sys.path.append("/home/david/fs20/thesis/code/report/scripts/burgers")
from utilities import (FVMObservationOperator,
                       PerturbedRiemannIC,
                       RusanovMCMC,
                       Measurer,
                       store_figure,
                       load_or_compute,
                       BurgersEquation,
                       autocorrelation,
                       wasserstein_distance,
                       DATA_DIR)


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


def is_valid_IC(u):
    """Return True if u is a valid initial condition.

    Namely it returns False if the initial shock location is
    outside of the domain.
    """
    s = u[2] + Settings.Prior.mean[2]
    eps = 0.1
    return (s > Settings.Simulation.domain[0] + eps and
            s < Settings.Simulation.domain[1] - eps)


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
            ground_truth = np.array([delta_1, delta_2, sigma])

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
        mean = np.array([0.25,    # delta_1
                         0.25,   # delta_2
                         -0.25])  # sigma
        std_dev = 0.25
        covariance = std_dev**2 * np.identity(len(mean))

        @staticmethod
        def get_distribution():
            return GaussianDistribution(Settings.Prior.mean,
                                        Settings.Prior.covariance)

    class Sampling:
        step = PWLinear(0.05, 0.001, 250)
        u_0 = np.zeros(3)
        N = 5000
        burn_in = 250
        sample_interval = 20
        rng = np.random.default_rng(2)

    @staticmethod
    def filename():
        return f"burgers_EP_n={Settings.Sampling.N}_h={Settings.Simulation.N_gridpoints}"


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
    accepter = ConstrainAccepter(CountedAccepter(StandardRWAccepter(potential, prior)), is_valid_IC)

    return MCMCSampler(proposer, accepter)


def create_data(ensemble_size,
                varied_quantity_setter,
                varied_quantity_getter,
                varied_quantity_values,
                varied_quantity_reference):
    """Create an ensemble and reference with the specified varied quantity

    ensemble_size: int
    varied_quantity_setter: callable
        Provides access to the global Settings and is used the change the
        varied_quantity in a way that the create_mcmc_sampler works correctly
        (i.e. sees the updated values)
    varied_quantity_getter: callable
        Gives access to the varied quantity, used so that the state of the
        Settings can be restored
    varied_quantity_values: list
        List elements are arguments to varied_quantity_setter
    varied_quantity: scalar
        Argument to varied_quantity_setter
    """
    # Change the global Settings variable, not the local one in function scope
    original_value = varied_quantity_getter()

    rngs = [np.random.default_rng(i) for i in range(ensemble_size)]

    # Compute chains
    ensembles = []
    for val in varied_quantity_values:
        varied_quantity_setter(val)
        sampler = create_mcmc_sampler()
        chain_start = partial(sampler.run,
                              Settings.Sampling.u_0,
                              Settings.Sampling.N,
                              Settings.Sampling.burn_in,
                              Settings.Sampling.sample_interval)

        ensemble_manager = EnsembleManager(DATA_DIR,
                                           f"{Settings.filename()}_"
                                           f"E={ensemble_size}")

        ensemble = ensemble_manager.compute(chain_start,
                                            rngs,
                                            ensemble_size)

        # Add prior mean
        for j in range(ensemble_size):
            for i in range(Settings.Sampling.N):
                ensemble[j, :, i] += Settings.Prior.mean

        ensembles.append(ensemble)

    # Compute reference
    varied_quantity_setter(varied_quantity_reference)
    sampler = create_mcmc_sampler()
    ref_chain_start = partial(sampler.run,
                              Settings.Sampling.u_0,
                              Settings.Sampling.N,
                              Settings.Sampling.burn_in,
                              Settings.Sampling.sample_interval)

    ref_manager = EnsembleManager(DATA_DIR,
                                  f"{Settings.filename()}_ref")

    ref_chain = ref_manager.compute(ref_chain_start, rngs[:1], 1)[0, :, :]

    for i in range(Settings.Sampling.N):
        ref_chain[:, i] += Settings.Prior.mean

    # reset varied quantity
    varied_quantity_setter(original_value)

    return ensembles, ref_chain


def convergence(ensembles, reference, varied_quantity,
                observable_function, distance_function,
                plt_info, filename):
    """Compute convergence of obsverable_function over varied_quantity of
    ensembles towards reference, measured by distance_function.

    ensembles: list(np.array((ensemble_size, u_dim, chain_length)))
    reference: np.array((u_dim, chain_length))
    varied_quantity: list
        len(varied_quantity) == len(ensembles)
    observable_function: callable
        Takes as argument one element of an ensemble (np.array((u_dim, chain_length)))
        and returns a 1D np.array.
    distance_function: callable
        Takes as argument two return values from observable_function and returns a
        the distance between them (a float).
    plot_info: dict(str: str)
        Dict containing:
        title
        xlabel (name of varied quantity)
        ylabel (name of observable)
    filename: string
    """
    n_ensembles = len(ensembles)
    assert n_ensembles == len(varied_quantity), ""
    ensemble_size = ensembles[0].shape[0]
    for ensemble in ensembles:
        assert ensemble_size == ensemble.shape[0], "Require equal-sized ensembles"

    reference_observable = observable_function(reference)

    distances = np.zeros((n_ensembles, ensemble_size))

    for j, ensemble in enumerate(ensembles):
        ensemble_observables = np.empty((ensemble_size,
                                         *reference_observable.shape))
        for i in range(ensemble_size):
            ensemble_observables[i, :] = observable_function(ensemble[i, :])

        for i in range(ensemble_size):
            distances[j, i] = distance_function(ensemble_observables[i, :],
                                                reference_observable)

    means = np.array([np.mean(distances[j, :]) for j in range(n_ensembles)])
    l_quartile = np.array([np.quantile(distances[j, :], 0.25) for j in range(n_ensembles)])
    u_quartile = np.array([np.quantile(distances[j, :], 0.75) for j in range(n_ensembles)])

    plt.errorbar(x=varied_quantity,
                 y=means,
                 yerr=np.array([means - l_quartile, u_quartile - means]),
                 capsize=5)
    plt.plot(varied_quantity, [np.sqrt(varied_quantity[0]) * means[0] / np.sqrt(a)
                               for a in varied_quantity], '--', label="O(L^(-1/2))")
    plt.title(plt_info["title"])
    plt.xlabel(plt_info["xlabel"])
    plt.xscale("log")
    plt.ylabel(plt_info["ylabel"])
    store_figure(f"{filename}_convergence_{plt_info['ylabel']}_{plt_info['xlabel']}"
                 .replace('$', '').replace(' ', '_'))
    # $ can occur from latex in labels and gives trouble for bash-operations


class WassersteinDistanceComputer:
    def __init__(self, ensembles, n_bins):
        # determine range of u
        self.u_range = np.array([np.min([np.min(ensemble) for ensemble in ensembles]),
                                 np.max([np.max(ensemble) for ensemble in ensembles])])
        self.n_bins = n_bins

    def create_histogram(self, chain):
        assert np.all(chain >= self.u_range[0]), ("Value(s) of this chain are outside the "
                                                  "precomputed range")
        assert np.all(chain <= self.u_range[1]), ("Value(s) of this chain are outside the "
                                                  "precomputed range")

        ensemble_binned = np.histogram(chain,
                                       bins=self.n_bins,
                                       range=self.u_range,
                                       density=False)[0]
        # hand-made normalization
        ensemble_binned = ensemble_binned / np.sum(ensemble_binned)

        return ensemble_binned

    def compute_distance(self, hist1, hist2):
        return wasserstein_distance(hist1, hist2, self.u_range.reshape(1, 2))


def show_ensemble(ensemble):
    """Plz write docstring"""

    title = f"Chain length: {ensemble.shape[2]}, ensemble members: {ensemble.shape[0]}"
    for k in range(ensemble.shape[0] - 1):
        for i in range(3):
            plt.plot(ensemble[k, i, :])

    for i in range(3):
        plt.plot(ensemble[-1, i, :], label=Settings.Simulation.IC.names[i])

    for p in Settings.Measurement.points:
        l = p - Settings.Measurement.interval / 2
        r = p + Settings.Measurement.interval / 2
        plt.axhspan(l, r, facecolor='r', alpha=0.3)

    shock_locs = np.zeros_like(ensemble[0, 0, :])
    rarefactions = []
    for i in range(len(shock_locs)):
        try:
            shock_locs[i] = BurgersEquation.riemann_shock_pos(ensemble[0, 0, i] + 1,
                                                              ensemble[0, 1, i],
                                                              ensemble[0, 2, i],
                                                              1)
        except AssertionError:
            rarefactions += [i]
    plt.plot(shock_locs, color='r')
    if rarefactions:
        print(f"{len(rarefactions)} rarefactions during sampling")
        for i in rarefactions:
            plt.axvline(i, color='r', alpha=0.05)

    plt.legend()
    plt.title(title)
    plt.show()

    # densities
    intervals = [(-2, 2)] * 3
    priors = [norm(loc=mu, scale=Settings.Prior.std_dev)
              for mu in Settings.Prior.mean]
    fig, plts = plt.subplots(1, 3, figsize=(20, 10))

    plot_info = list(zip(intervals,
                         (0.025, -0.025, -0.02),
                         Settings.Simulation.IC.names,
                         priors,
                         plts))

    for k in range(ensemble.shape[0] - 1):
        for i, (interval, _, __, ___, ax) in enumerate(plot_info):
            x_range = np.linspace(*interval, num=500)
            ax.hist(ensemble[k, i, :], density=True, alpha=0.5)

    for i, (interval, true_val, name, prior, ax) in enumerate(plot_info):
        x_range = np.linspace(*interval, num=500)
        ax.plot(x_range, [prior.pdf(x) for x in x_range])
        ax.hist(ensemble[-1, i, :], density=True, color='b', alpha=0.5)
        ax.axvline(true_val, c='r')
        ax.set_title(f"Posterior for {name}")
        ax.set(xlabel=name, ylabel="Probability")

    plt.show()


# These setters are required to be able to give the create_data
# the varied quantity as argument and change the global value
def sample_N_change(new_N):
    global Settings
    Settings.Sampling.N = new_N


def sample_N_get():
    return Settings.Sampling.N


def grid_N_change(new_N):
    global Settings
    Settings.Simulation.N_gridpoints = new_N


def grid_N_get():
    return Settings.Simulation.N_gridpoints


def wasserstein_convergence_chainlength():
    ensemble_size = 10
    chain_lengths = [250, 500, 1000, 2000]
    ref_length = 5000
    ensembles, ref_chain = create_data(ensemble_size,
                                       sample_N_change,
                                       sample_N_get,
                                       chain_lengths,
                                       ref_length)

    wasserstein_plot_info = {"title": "$W_1$ for different chain lengths",
                             "xlabel": "Length of the chain",
                             "ylabel": "$W_1$"}
    for i, name in enumerate(Settings.Simulation.IC.names):
        # extract 1 dim of u from ensembles
        one_var_ensembles = [ensemble[:, i, :].reshape(ensemble_size,
                                                       1,
                                                       chain_length)
                             for ensemble, chain_length in zip(ensembles, chain_lengths)]
        n_bins = 20
        wasserstein = WassersteinDistanceComputer(one_var_ensembles, n_bins)
        convergence(ensembles=one_var_ensembles,
                    reference=ref_chain[i, :].reshape(1, ref_length),
                    varied_quantity=chain_lengths,
                    observable_function=wasserstein.create_histogram,
                    distance_function=wasserstein.compute_distance,
                    plt_info=wasserstein_plot_info,
                    filename=f"{Settings.filename()}_{name}")


def wasserstein_convergence_grid():
    ensemble_size = 3
    grid_sizes = [32, 64]
    ref_grid = 128
    ensembles, ref_chain = create_data(ensemble_size,
                                       grid_N_change,
                                       grid_N_get,
                                       grid_sizes,
                                       ref_grid)

    wasserstein_plot_info = {"title": "$W_1$ for different grid spacings",
                             "xlabel": "Number of gridpoints",
                             "ylabel": "$W_1$"}
    for i, name in enumerate(Settings.Simulation.IC.names):
        # extract 1 dim of u from ensembles
        one_var_ensembles = [ensemble[:, i, :].reshape(ensemble_size,
                                                       1,
                                                       Settings.Sampling.N)
                             for ensemble in ensembles]
        n_bins = 20
        wasserstein = WassersteinDistanceComputer(one_var_ensembles, n_bins)
        convergence(ensembles=one_var_ensembles,
                    reference=ref_chain[i, :].reshape(1, Settings.Sampling.N),
                    varied_quantity=grid_sizes,
                    observable_function=wasserstein.create_histogram,
                    distance_function=wasserstein.compute_distance,
                    plt_info=wasserstein_plot_info,
                    filename=f"{Settings.filename()}_{name}")


def convergence_scalar_function_chainlength(function, name):
    """Convergence over scalar function of the posterior"""
    ensemble_size = 10
    chain_lengths = [250, 500, 1000, 2000]
    ref_length = 5000
    ensembles, ref_chain = create_data(ensemble_size,
                                       sample_N_change,
                                       sample_N_get,
                                       chain_lengths,
                                       ref_length)

    mean_plot_info = {"title": f"Posterior {name} for different chain lengths",
                      "xlabel": "Length of the chain",
                      "ylabel": f"{name}"}
    for i, name in enumerate(Settings.Simulation.IC.names):
        # extract 1 dim of u from ensembles
        one_var_ensembles = [ensemble[:, i, :].reshape(ensemble_size,
                                                       1,
                                                       chain_length)
                             for ensemble, chain_length in zip(ensembles, chain_lengths)]

        convergence(ensembles=one_var_ensembles,
                    reference=ref_chain[i, :].reshape(1, ref_length),
                    varied_quantity=chain_lengths,
                    observable_function=lambda x: np.array([function(x)]),
                    distance_function=lambda x, y: np.abs(x-y),
                    plt_info=mean_plot_info,
                    filename=f"{Settings.filename()}_{name}")


if __name__ == '__main__':
    # wasserstein_convergence_chainlength()
    wasserstein_convergence_grid()
    # convergence_scalar_function_chainlength(np.mean, "mean")
    # convergence_scalar_function_chainlength(np.var, "variance")
