import numpy as np

import matplotlib.pyplot as plt


class RusanovFVM:
    # open boundary conditions

    # not really happy with the solution of having
    # arrays for u as class attrbutes instead of
    # function arguments/returns. It makes stuff
    # pretty unclear and the performance gain
    # is dubious and hypothetical anyway.

    def __init__(self, flux, flux_prime, domain, N):
        self.f = flux
        self.fp = flux_prime
        dx = (domain[1] - domain[0]) / N
        # include ghost-cells
        self.N = N + 2
        # self.x are the cell centers
        self.x, self.dx = np.linspace(start=domain[0] - .5 * dx,
                                      stop=domain[1] + .5 * dx,
                                      num=self.N,
                                      retstep=True)

        self.u = np.zeros_like(self.x)
        self.u_star = np.zeros_like(self.u)
        self.dudt = np.zeros_like(self.u)

    def integrate(self, IC, T, only_endstate=True):
        self.u = np.array([IC(x_) for x_ in self.x])

        t = 0

        if not only_endstate:
            U = [np.copy(self.u)]  # store sequence of states
            Ts = [t]               # store sequence of timepoints

        while (t < T):
            dt = self._cfl()
            t += dt

            # writes new state into self.u
            self._step(dt)

            if not only_endstate:
                U += [np.copy(self.u)]
                Ts += [t]

        if not only_endstate:
            U_ = np.empty((len(self.u), len(Ts)))
            for i in range(len(Ts)):
                U_[:, i] = U[i]

            T_ = np.array(Ts)

            return U_, T_

        return np.copy(self.u), t

    def _step(self, dt):
        """Write state after dt evolution of self.u into self.u"""
        # SSPRK2
        self._rate_of_change(self.u, dt)
        self.u_star = self.u + dt * self.dudt
        self._apply_bc(self.u_star)

        self._rate_of_change(self.u_star, dt)
        self.u_star += dt * self.dudt

        self.u += self.u_star
        self.u /= 2
        self._apply_bc(self.u)

    def _rate_of_change(self, u_0, dt):
        """Write 1 / dx (F_j+1/2 - F_j-1/2) based on u_0 into dudt"""
        self._flux_difference(u_0)
        self.dudt /= -self.dx

    def _flux_difference(self, u_0):
        """write (F_j+1/2 - F_j-1/2) based on u_0 into dudt"""
        # flux over left boundary, to be updated in loop
        f_right = self._flux(u_0[0], u_0[1])

        for i in range(1, self.N - 1):
            f_left = f_right
            f_right = self._flux(u_0[i], u_0[i+1])

            self.dudt[i] = f_right - f_left

    def _flux(self, u_l, u_r):
        flux_average = 0.5 * (self.f(u_l) + self.f(u_r))
        speed = max(abs(self.fp(u_l)), abs(self.fp(u_r)))

        return flux_average - 0.5 * speed * (u_r - u_l)

    def _apply_bc(self, u):
        u[0] = u[1]
        u[-1] = u[-2]

    def _cfl(self):
        """dt = 0.5 * dx / max(|f'(u)|)"""
        max_speed = 0
        # this could probably be vectorized
        for i in range(1, self.N - 1):
            max_speed = max(max_speed, abs(self.fp(self.u[i])))

        return 0.5 * self.dx / max_speed


def test_RunanovFVM():
    def test_domain():
        # dx = 0.2
        # N = 12 (includes ghost cells)
        # x = [-1.1, -0.9, ..., 0.9, 1.1] (cell centers, includes ghosts)
        r = RusanovFVM(None, None, (-1, 1), 10)

        assert np.isclose(r.dx, 0.2), ""
        assert r.N == 12, ""
        assert np.isclose(r.x[1], -0.9), ""
        assert np.isclose(r.x[-2], 0.9), ""

    def test_cfl():
        # dx = 0.25
        r = RusanovFVM(None,
                       burgers_flux_prime,
                       (0,1),
                       4)

        r.u = np.array([1,1,1,1,1,1])
        assert np.isclose(r._cfl(), 0.125), ""

        r.u = np.array([1,1,-2,1,1,1])
        assert np.isclose(r._cfl(), 0.0625), ""

    def test_apply_bc():
        r = RusanovFVM(None, None, (0,1), 1)

        r.u = np.array([1,2,1])
        r._apply_bc(r.u)
        assert r.u[0] == 2 and r.u[2] == 2, ""

    def test_flux():
        r = RusanovFVM(burgers_flux,
                       burgers_flux_prime,
                       (0,1),
                       10)

        assert np.isclose(r._flux(1,1), .5), ""
        assert np.isclose(r._flux(0,1), -.25), ""
        assert np.isclose(r._flux(0,-1), .75), ""
        assert np.isclose(r._flux(4,5), 7.75), ""

    def test_rate_of_change():
        r = RusanovFVM(burgers_flux,
                       burgers_flux_prime,
                       (0,.3),
                       3)

        u = np.array([1, 1, -1, 2, 2])
        r._rate_of_change(u, .1)
        assert all(np.isclose(r.dudt[1:-1],
                              np.array([-10, 32.5, -37.5]))), ""

    test_domain()
    test_cfl()
    test_apply_bc()
    test_flux()
    test_rate_of_change()


def burgers_flux(x):
    return .5 * x**2


def burgers_flux_prime(x):
    return x


def show_burgers_riemann():
    r = RusanovFVM(burgers_flux,
                   burgers_flux_prime,
                   (-1,1),
                   500)

    U, T = r.integrate(lambda x: float(x < 0), 1, False)

    plt.plot(r.x, U[:, -1])
    plt.show()

if __name__ == '__main__':
    test_RunanovFVM()
    show_burgers_riemann()
