import numpy as np
import matplotlib.pyplot as plt

from ip_mcmc import MCMCSampler, AnalyticAccepter, CountedAccepter, StandardRWProposer


def bimodal(x):
    l = 10
    mu = 3
    sigma = 1

    if abs(x) > l:
        return 0

    # only used in acceptance, can forget normalisation
    return (np.exp(-.5 * ((x - mu) / sigma) ** 2)
            + np.exp(-.5 * ((x + mu) / sigma) ** 2))


def normal(x):
    return np.exp(-0.5*x**2)


def main():
    random_walk_proposer = StandardRWProposer(0.25, 1)
    acceptance = CountedAccepter(AnalyticAccepter(bimodal))
    mcmc = MCMCSampler(random_walk_proposer, acceptance, np.random.default_rng(1))

    x_0 = np.array([0])

    # show distribution
    X1 = mcmc.run(x_0, 1000)
    plt.hist(X1, bins=20)
    plt.show()

    # print acceptance ratio
    print(f"Acceptance ratio: {acceptance.accepts / acceptance.calls}")

    # calculate autocorrelation
    X2 = mcmc.run(x_0, n_samples=5000, sample_interval=1).flatten()
    ac = np.correlate(X2, X2, mode='full')
    ac = ac[ac.size//2:]
    ac /= ac[0]

    plt.plot(ac)
    plt.show()


if __name__ == '__main__':
    main()
