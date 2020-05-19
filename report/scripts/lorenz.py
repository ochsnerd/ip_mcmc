import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from ip_mcmc import (MCMCSampler,
                     pCNAccepter,
                     pCNProposer,
                     EvolutionPotential,
                     GaussianDistribution)


class Lorenz96:
    """
    Lorenz-96 model as described in "Earth Systems Modelling 2.0"
    """
    def __init__(self, K, J, F, h, c, b):
        """
        K : int
            # of X (slow variables)
        J : int
            # of Y (fast variables)
        F : float
            forcing of slow variables
        h : float
            coupling strength between fast and slow variables
        c : float
            relative damping between fast and slow variables
        b : float
            amplitude of nonlinear fast-fast interaction

        Size of resulting state vector : K * (1 + J)
        """
        self.K = K
        self.J = J
        self.F = F
        self.h = h
        self.c = c
        self.b = b

        self.fast_slow_fact = self.h * self.c
        self.slow_fast_fact = self.h / self.J

    def __call__(self, _, in_state):
        """
        d(state)/dt = __call__(t, state)
        t is unused
        state = [X_0, .. , X_{K-1}, Y_{0,0}, .., Y_{0,J-1}, Y_{1,0}, .., Y_{K-1,J-1}]
        Y-indexing not the same as paper!
        """
        K, J = self.K, self.J

        out_state = np.empty_like(in_state)

        # Split up state
        X_in, X_out = in_state[:K], out_state[:K]

        Y_in, Y_out = (np.reshape(in_state[K:], (K, J)),
                       np.reshape(out_state[K:], (K, J)))

        # slow variables
        X_out[:] = self._slow_variables(X_in, Y_in)

        # fast variables
        for k in range(K):
            beg = k * J
            end = (k + 1) * J
            Y_out[k, :] = self._fast_variables(Y_in[k, :], X_in[k])

        return out_state

    def _slow_variables(self, X_in, Y_in):
        def slow_nonlinearity(X):
            return np.roll(X, 1) * np.roll(X, 2) - np.roll(X, 1) * np.roll(X, -1)

        X_out = np.copy(X_in)

        X_out *= -1
        X_out -= slow_nonlinearity(X_in)
        X_out += self.F

        # fast-slow interaction
        for k in range(self.K):
            X_out[k] -= self.fast_slow_fact * np.average(Y_in[k, :])

        return X_out

    def _fast_variables(self, Yk_in, Xk_in):
        def fast_nonlinearity(Yk):
            return np.roll(Yk, -1) * np.roll(Yk, -2) - np.roll(Yk, 1) * np.roll(Yk, -1)

        Yk_out = np.copy(Yk_in)

        Yk_out *= -1
        Yk_out -= self.b * fast_nonlinearity(Yk_in)
        Yk_out += self.h / self.J * Xk_in
        Yk_out *= self.c

        return Yk_out

    def slow_energy(self, state):
        return np.linalg.norm(state[:self.K])**2

    def fast_energy(self, state, k):
        return np.linalg.norm(state[self.K + self.J * k:
                                    self.K + self.J * (k + 1)])**2

    def total_energy(self, state):
        return np.linalg.norm(state)**2


def test_Lorenz96():
    def forcing():
        l_test = Lorenz96(3,1,2,1,1,1)
        return all(np.isclose(l_test(1, np.array([0,0,0,0,0,0], dtype=float)),
                              [2,2,2,0,0,0]))

    def slow_nonlinearity():
        l_test = Lorenz96(4,1,0,0,0,0)
        return all(np.isclose(l_test(1, np.array([1,2,3,4,0,0,0,0], dtype=float)),
                              [-5,-3,3,-7,0,0,0,0]))

    def fast_nonlinearity():
        l_test = Lorenz96(1,4,0,0,1,2)

        return all(np.isclose(l_test(1, np.array([0,1,2,3,4], dtype=float)),
                              [0,3,-20,5,-2]))

    def slow_to_fast_interaction():
        l_test = Lorenz96(1,1,0,2,-1,0)

        return all(np.isclose(l_test(1, np.array([1,0], dtype=float)),
                              [-1, -2]))

    def fast_to_slow_interaction():
        l_test = Lorenz96(1,2,0,2,-1,0)

        return all(np.isclose(l_test(1, np.array([0,1,2], dtype=float)),
                              [3,1,2]))

    def step():
        l_test = Lorenz96(2,2,1,1,1,1)

        return all(np.isclose(l_test(1, np.array([2,3,4,5,6,7], dtype=float)),
                              [-2.5, -10.5, 2, -8, 2.5, -11.5]))

    def energies():
        l_test = Lorenz96(2,2,0,0,0,0)

        if not np.isclose(l_test.slow_energy([-2,2,2,2,2,2]), 8):
            return False

        if not np.isclose(l_test.fast_energy([2,2,-2,2,2,2], 0), 8):
            return False

        if not np.isclose(l_test.total_energy([-2,2,2,-2,2,2]), 24):
            return False

        return True

    assert forcing(), ""
    assert slow_nonlinearity(), ""
    assert fast_nonlinearity(), ""
    assert slow_to_fast_interaction(), ""
    assert fast_to_slow_interaction(), ""
    assert step(), ""
    assert energies(), ""

    print("Passed tests")


def store_figure(name):
    """
    Store a figure in the figures directory.
    Assumes there is an active pyplot-Plot and clears it after
    """
    plt.savefig("/home/david/fs20/thesis/code/report/figures/" + name + ".svg", format='svg')
    plt.clf()


def equilibrium_requirements(y_equilibrium, model):
    """
    Check whether eq (14) + (15) from "Earth systems modelling 2.0"
    are statisfied for y_equilibrium
    """
    y = y_equilibrium
    n_t = y.shape[1]
    K, J = model.K, model.J
    F, h, c = model.F, model.h, model.c

    X_errors = []
    for k in range(K):
        m_X = np.mean(y[k, :])
        m_X_sq = np.mean(y[k, :]**2)

        m_XY = 0
        for t in range(n_t):
            m_XY += y[k, t] * np.mean(y[K + k * J: K + (k + 1) * J, t])
        m_XY /= n_t

        X_errors += [F * m_X - h * c * m_XY - m_X_sq]

    Y_errors = []
    for k in range(K):
        m_Y_sq_a = 0
        m_XY = 0
        for t in range(n_t):
            m_Y_sq_a += np.mean(y[K + k * J: K + (k + 1) * J, t]**2)
            m_XY += y[k, t] * np.mean(y[K + k * J: K + (k + 1) * J, t])
        m_Y_sq_a /= n_t
        m_XY /= n_t

        Y_errors += [h / J * m_XY - m_Y_sq_a]

    return np.mean(X_errors), np.mean(Y_errors)

def solve_ODE(y_0, f, tspan):
    result = solve_ivp(fun=f,
                       t_span=tspan,
                       y0=y_0,
                       method='RK45')

    return result.t, result.y

def show_lorenz96(K, J, F, h, c, b, t_final):

    l = Lorenz96(K, J, F, h, c, b)

    IC = np.random.random(K * (J + 1))

    t, y = solve_ODE(IC, l, (0, t_final))

    n_t = len(t)
    print(f"steps: {n_t}")

    plt.plot(np.repeat(y[:K, 0], J), label="slow")
    plt.plot(y[K:, 0], label="fast")
    plt.title("Inital conditions timestep")
    plt.legend()
    store_figure("lorenz96_IC")

    plt.plot(np.repeat(y[:K, n_t // 2], J), label="slow")
    plt.plot(y[K:,n_t // 2], label="fast")
    plt.title("Middle timestep")
    plt.legend()
    store_figure("lorenz96_middle")

    plt.plot(np.repeat(y[:K, -1], J), label="slow")
    plt.plot(y[K:,-1], label="fast")
    plt.title("Last timestep")
    plt.legend()
    store_figure("lorenz96_last")

    plt.plot(t[:], [(l.slow_energy(y[:, i])) for i in range(len(t))], label="slow")
    plt.plot(t[:], [(l.total_energy(y[:, i])) for i in range(len(t))], label="total")
    plt.plot(t[:], [(l.fast_energy(y[:, i], 0)) for i in range(len(t))], label="fast, first cell")
    plt.title("Energy")
    plt.legend()
    store_figure("lorenz96_energies")

    mean_x_error,  mean_y_error = equilibrium_requirements(y[:, len(t) // 4:], l)

    print(f"{mean_x_error=}, {mean_y_error=}")


def equilibrium_requirements_analysis():
    rng = np.random.default_rng(1)
    K, J = 10, 5
    F, h, c, b = 10, 1, 10, 10

    def run_lorenz(t_final, rng):
        l = Lorenz96(K, J, F, h, c, b)
        IC = rng.random(K * (J + 1))
        t, y = solve_ODE(IC, l, (0, t_final))

        print(f"{len(t)=}")

        return equilibrium_requirements(y[:, len(t) // 4:], l)

    times = [10, 20, 30, 40, 50, 60]
    rms_x_errors = []
    rms_y_errors = []
    for T in times:
        n_runs = 10
        errors = []
        for _ in range(n_runs):
            print(f"{_ + 1} of {T=}")
            errors += [run_lorenz(T, rng)]

        print(errors)

        rms_x_errors += [np.sqrt(np.mean([e[0]**2 for e in errors]))]
        rms_y_errors += [np.sqrt(np.mean([e[1]**2 for e in errors]))]

    plt.plot(times, rms_x_errors, label="X")
    plt.plot(times, rms_y_errors, label="Y")
    plt.title("RMSE from equilibrium requirement")
    plt.legend()
    store_figure("equilibrium_error_n=5")


def main():
    test_Lorenz96()
    show_lorenz96(K=36, J=10, F=10, h=1, c=10, b=10, t_final=60)
    equilibrium_requirements_analysis()


if __name__ == '__main__':
    main()
