import numpy as np

from .proposer import ProposerBase


class MockProposer(ProposerBase):
    def __call__(self, u, rng):
        return u


class MockRNG(np.random.Generator):
    def __init__(self, result):
        super(MockRNG, self).__init__(np.random.PCG64())
        self.result = result

    def normal(self, loc=None, scale=None):
        return self.result

    def multivariate_normal(self, mean=None, cov=None):
        assert mean is not None, "Actually need to provide mean to infer dimension"
        return self.result * np.ones_like(mean)

    def random(self):
        if 0 <= self.result <= 1:
            return self.result
        return 0.5
