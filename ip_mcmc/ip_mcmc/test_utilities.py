from .proposer import ProposerBase


class MockProposer(ProposerBase):
    def __call__(self, u, rng):
        return u


class MockRNG:
    def __init__(self, result):
        self.result = result

    def normal(self, loc=0, scale=1):
        return self.result

    def random(self):
        if 0 <= self.result <= 1:
            return self.result
        return 0.5
