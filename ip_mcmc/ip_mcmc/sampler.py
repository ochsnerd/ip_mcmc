import numpy as np

from .accepter import CountedAccepter


class MCMCSampler:
    def __init__(self, proposal, acceptance, rng):
        self.proposer = proposal
        self.accepter = acceptance
        self.rng = rng

    def run(self, u_0, n_samples, burn_in=1000, sample_interval=200):
        u = u_0

        if isinstance(self.accepter, CountedAccepter):
            self.accepter.reset()

        for _ in range(max(0, burn_in - sample_interval)):
            u = self._step(u, self.rng)

        samples = np.empty((n_samples, u.size))

        for i in range(n_samples):
            for _ in range(sample_interval):
                u = self._step(u, self.rng)

            samples[i, :] = u

        if isinstance(self.accepter, CountedAccepter):
            print(f"Acceptance ratio: {self.accepter.ratio()}")

        return samples

    def _step(self, u, rng):
        v = self.proposer(u, rng)

        if self.accepter(u, v, rng):
            return v

        return u
