import numpy as np

from abc import ABC, abstractmethod


class AccepterBase(ABC):
    @abstractmethod
    def __call__(self, u, v, rng):
        """Return True if v is accepted"""
        ...


class CountedAccepter(AccepterBase):
    """Count the steps/accepts of the underlying accepter"""
    def __init__(self, accepter):
        self.accepter = accepter
        self.calls = 0
        self.accepts = 0

    def __call__(self, u, v, rng):
        accepted = self.accepter(u, v, rng)

        self.calls += 1
        if accepted:
            self.accepts += 1

        return accepted

    def reset(self):
        self.calls = 0
        self.accepts = 0

    def ratio(self):
        if self.calls == 0:
            raise ValueError("No samples yet!")
        return self.accepts / self.calls


class ConstrainAccepter(AccepterBase):
    """Decline a new move if it violates some constraint.

    Decorates another accepter"""
    def __init__(self, accepter, constraint):
        """
        accepter: instance of AccepterBase
        constraint: callable
            Takes as argument a state and returns False if it violates
        """
        self.accepter = accepter
        self.is_valid = constraint

    def __call__(self, u, v, rng):
        if self.is_valid(v):
            return self.accepter(u, v, rng)
        return False


class ProbabilisticAccepter(AccepterBase):
    def __call__(self, u, v, rng):
        """Return True if v is accepted"""
        a = self.accept_probability(u, v)
        return a > rng.random()

    @abstractmethod
    def accept_probability(self, u, v):
        ...


class AnalyticAccepter(ProbabilisticAccepter):
    """Accepter for sampling from a known distibution rho

    Assuming a symmetric proposal function (transition kernel):
    q(u|v) = q(v|u)

    Accept v|u with probability
    a = min(1, rho(v)/rho(u))
    """

    def __init__(self, rho):
        self.rho = rho

    def accept_probability(self, u, v):
        return self.rho(v) / self.rho(u)


class StandardRWAccepter(ProbabilisticAccepter):
    """Accept a move from u to v with probability
    a(u,v) = min{1, exp(I(u) - I(v))}, where
    I(u) = Theta(U) + 0.5 * norm(C^(-1/2)u)^2

    Based on (1.2) MCMCMF
    """

    def __init__(self, potential, prior):
        self.theta = potential
        self.prior = prior

        # Instead of just printing to stdout,
        # numpy will raise an exception.
        # This is so that we can surpress the warning
        # when calculating the accept_probability below.
        # There it can happen when a large improvement is
        # proposed that the return value overflows a 64-bit float.
        # This is not a problem however, since the return value
        # is only relevant if it's in [0,1]
        np.seterr(all='raise')

    def accept_probability(self, u, v):
        Iu = self._I(u)
        Iv = self._I(v)

        try:
            return np.exp(Iu - Iv)
        except FloatingPointError:
            # The result is too big for np.float64,
            # we however only care if the value
            # is in [0,1].
            return 1

    def _I(self, w):
        regularizer = .5 * np.linalg.norm(self.prior.apply_sqrt_covariance(w)) ** 2
        return self.theta(w) + regularizer


class pCNAccepter(ProbabilisticAccepter):
    """Accept a move from u to v with probability
    a(u,v) = min{1, exp(Theta(u) - Theta(v))}

    Works with both CN and pCN proposers.

    Based on (4.11) in MCMCMF.

    """
    def __init__(self, potential):
        self.theta = potential

    def accept_probability(self, u, v):
        return np.exp(self.theta(u) - self.theta(v))
