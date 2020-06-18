import numpy as np

import sys
sys.path.append("/home/david/fs20/thesis/code/report/scripts/")
sys.path.append("/home/david/fs20/thesis/code/report/scripts/burgers")
from rusanov import RusanovFVM
from helpers import store_figure, load_or_compute, autocorrelation


class FVMObservationOperator:
    """
    Observation operator for MCMC based on the time-evolution
    of a FVM
    """
    def __init__(self, IC, prior_means, integrator, measurement_operator):
        """
        Write requirements here

        IC: given u as argument, return callable used as initial condition for
            the integrator.
        prior_means: mean values of the prior placed on u,
            i.e. integrate IC(prior_means + u).
        integrator: given a callable as initial condition, integrate it
            and return the end-state.
        measurement_operator: given the end-state returned by the integrator,
            measure the observales.
        """
        self.IC = IC
        self.u_0 = prior_means
        self.integrator = integrator
        self.meas_op = measurement_operator

    def __call__(self, u):
        return self.meas_op(self.integrator(self.IC(self.u_0 + u)))


class PerturbedRiemannIC:
    """
    f(x) = 1 + delta_1 if x < sigma
           delta_2     if x > sigma
    """
    def __init__(self, params):
        """
        params[0]: delta_1
        params[1]: delta_2
        params[2]: sigma
        """
        self.left = 1 + params[0]
        self.right = params[1]
        self.jump = params[2]

    def __call__(self, x):
        if x < self.jump:
            return self.left
        return self.right


class RusanovMCMC:
    """
    Rusanov FVM integrator for use in the MCMC algorithm

    All parameters except the IC are set during initialization
    """
    def __init__(self, flux, flux_prime, domain, N, T):
        self.FVM = RusanovFVM(flux, flux_prime, domain, N)
        self.T = T

    def __call__(self, IC):
        # u as in "final state u", not "MCMC state u"
        u, t = self.FVM.integrate(IC, self.T)
        assert t >= self.T, "Did not reach final time during integration"
        return u


class Measurer:
    """
    Measure around points
    """
    def __init__(self, measurement_points, measurement_interval, x_values):
        # x_values: evenly spaced points where the given values are located
        self.n_meas = len(measurement_points)

        self.n_x_vals = len(x_values)
        self.dx = x_values[1] - x_values[0]
        m_p = np.asarray(measurement_points, dtype=np.float)
        self.left_limits = np.searchsorted(x_values,
                                           m_p - measurement_interval / 2,
                                           side='left')
        self.right_limits = np.searchsorted(x_values,
                                            m_p + measurement_interval / 2,
                                            side='left')

    def __call__(self, values):
        assert len(values) == self.n_x_vals, "Provided values don't match x_vals"

        m = np.empty_like(self.left_limits, dtype=np.float)
        for i in range(self.n_meas):
            left = self.left_limits[i]
            right = self.right_limits[i]
            m[i] = 10 * np.trapz(values[left:right], dx=self.dx)

        return m


class BurgersEquation:
    @staticmethod
    def flux(w):
        return .5 * w * w

    @staticmethod
    def flux_prime(w):
        return w

    @staticmethod
    def riemann_shock_pos(w_l, w_r, s_0, T):
        """Return the location of the shock.

        Return the position of the shock after time T
        of a Riemann problem with initial conditions
        w(x, 0) = w_l  if x < s_0
                  w_r  if x > s_0
        """
        assert w_l > w_r, "doesn't work for rarefactions"
        return s_0 + 0.5 * (w_r**2 - w_l**2) / (w_r - w_l) * T
