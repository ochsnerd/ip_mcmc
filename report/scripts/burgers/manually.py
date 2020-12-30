import numpy as np
import matplotlib.pyplot as plt

from rusanov import RusanovFVM
from utilities import (PerturbedRiemannIC,
                       BurgersEquation,
                       Measurer)

delta_1 = 0.25
delta_2 = 0
sigma_0 = 0
IC = PerturbedRiemannIC((delta_1, delta_2, sigma_0))

FVM = RusanovFVM(BurgersEquation.flux,
                 BurgersEquation.flux_prime,
                 (-1,1),
                 128)
u_end, _ = FVM.integrate(IC, 1, True)
x = FVM.x[1:-1]  # exlude ghost cells

plt.plot(x, [IC(x_) for x_ in x])
plt.title("Initial condition")
plt.show()

plt.plot(x, u_end)
plt.title("Endstate")
plt.show()

meas = Measurer([-0.5, -0.25, 0.25, 0.5, 0.75], 0.1, x)
print(x[meas.left_limits[0]], x[meas.right_limits[0]])
meas_result = meas(u_end)
print(meas_result)
