import numpy as np

from .accepter import AnalyticAccepter, StandardRWAccepter
from .distribution import GaussianDistribution
from .test_utilities import MockRNG


def test_AnalyticAccepter():
    rng = MockRNG(0.5)

    a = AnalyticAccepter(lambda x: x)

    assert a(1, 1, rng) is True, ""
    assert a(1, 0, rng) is False, ""
    assert a(1, 0.499, rng) is False, ""
    assert a(1, 0.501, rng) is True, ""


def test_StandardRWAccepter():
    def potential(u):
        return np.linalg.norm(u)

    rng = MockRNG(np.exp(-(np.sqrt(2) + 0.5)) + 0.01)
    d = GaussianDistribution(mean=0, covariance=2)

    a = StandardRWAccepter(potential, d)

    assert np.isclose(a._I(1), 1 + 0.25), ""
    assert np.isclose(a._I(5), 5 + 25 / 4), ""

    assert a(0, 0, rng), ""
    assert not a(0, np.sqrt(2), rng), ""
