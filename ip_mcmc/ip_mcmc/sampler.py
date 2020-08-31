import numpy as np

from .accepter import CountedAccepter


class MCMCSampler:
    def __init__(self, proposal, acceptance):
        self.proposer = proposal
        self.accepter = acceptance

    def run(self, u_0, n_samples, rng, burn_in=1000, sample_interval=200):
        u = u_0

        if isinstance(self.accepter, CountedAccepter):
            self.accepter.reset()

        for _ in range(max(0, burn_in - sample_interval)):
            u = self._step(u, rng)

        samples = np.empty((n_samples, u.size))

        for i in range(n_samples):
            # print(f"Sampling {i + 1}/{n_samples}")
            for _ in range(sample_interval):
                u = self._step(u, rng)

            samples[i, :] = u

        if isinstance(self.accepter, CountedAccepter):
            print(f"Acceptance ratio: {self.accepter.ratio()}")

        return samples

    def _step(self, u, rng):
        v = self.proposer(u, rng)

        if self.accepter(u, v, rng):
            return v

        return u

    @classmethod
    def autocorr(cls, x):
        x_ = x - np.mean(x)
        result = np.correlate(x_, x_, mode='full')
        result = result[-len(x):]
        # the warning raised by numpy on divide-by-zero
        # somehow doesn't get caught by try-except
        if result[0] == 0:
            # When x is constant, x_ is 0, resulting
            # in 0 correlation, when it should be 1
            return np.ones_like(result)
        return result / result[0]
