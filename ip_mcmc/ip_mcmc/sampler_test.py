import numpy as np

from .sampler import MCMCSampler
from .test_utilities import MockProposer, MockRNG
from .accepter import AnalyticAccepter, CountedAccepter


def test_sampler():
    a = CountedAccepter(AnalyticAccepter(lambda x: x))
    s = MCMCSampler(MockProposer(),
                    a)

    r = s.run(np.array([1]),
              n_samples=10,
              rng=MockRNG(0.1),
              burn_in=100,
              sample_interval=20)

    assert a.calls == 100 + 9*20, ""
    assert a.accepts == a.calls, ""
    assert all(np.isclose(r_, 1) for r_ in r[0]), ""
