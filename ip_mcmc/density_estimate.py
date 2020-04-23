import numpy as np
import matplotlib.pyplot as plt

from mcmc import MCMCSampler, StandardRWAccepter, StandardRWProposer, GaussianDistribution


def bimodal(x):
    l = 10
    mu = 3
    sigma = 1

    if abs(x) > l:
        return 0

    # only used in acceptance, can forget normalisation
    return (np.exp(-.5 * ((x - mu) / sigma) ** 2)
            + np.exp(-.5 * ((x + mu) / sigma) ** 2))


class Measurements():
    def __init__(self, rho):
        self.rho = rho
        self.u = []

    def measure(self, points):
        for p in points:
            self.u.append(rho(p))

    def __call__(self, x):
        pass


def main():
    m = Measurements(bimodal)
    m.measure([-1, 0, 1])

    random_walk_proposer = StandardRWProposer(delta=0.25, 1)
    acceptance = StandardRWAccepter(potential, GaussianDistribution())
    sampler = MCMCSampler(random_walk_proposer, acceptance, np.random.default_rng(2))

    x_0 = np.array([0])
    X = mcmc.run(x_0, 1000)

    plt.hist(X, bins=20)
    plt.show()

    
if __name__ == '__main__':
    main()
