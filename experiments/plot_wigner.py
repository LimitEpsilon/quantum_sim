import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from qutip import *

# dimension of finite Hilbert space
N = 100

b  = destroy(N)
nb = num(N)
tlist = np.linspace(0, 10, 100)

H0 = b.dag() * b
H1 = b.dag() * b.dag() * b.dag() / 6
H2 = b * b * b / 6

coupling = 0.2

alpha = coupling * np.exp(-2j * tlist)

# Hamiltonian for evolution into squeezed state
H = [[H0, tlist], [H1, alpha], [H2, np.conjugate(alpha)]]

# start in the ground (vacuum) state
psi0 = basis(N, 0)

decay = 1.0

c_ops = [decay * b]

e_ops = []

output = mesolve(H, psi0, tlist, [], e_ops)
output_lossy = mesolve(H, psi0, tlist, c_ops, e_ops)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# plot the Wigner function of the non-lossy evolution
# plt.subplot(1, 2, 1)
psi = output.states[49]
xvec = np.arange(-100., 100.) * 5. / 100
W = wigner(psi, xvec, xvec)
c = axes[0].contourf(xvec, xvec, W, 100)
axes[0].set_xlim([-5, 5])
axes[0].set_ylim([-5, 5])
axes[0].set_title('Without loss')

# plot the Wigner function of the lossy evolution
# plt.subplot(1, 2, 2)
psi_lossy = output_lossy.states[49]
W_lossy = wigner(psi_lossy, xvec, xvec)
c_lossy = axes[1].contourf(xvec, xvec, W_lossy, 100)
axes[1].set_xlim([-5, 5])
axes[1].set_ylim([-5, 5])
axes[1].set_title('With loss')

plt.savefig('squeezed_decay_1.0.png', bbox_inches='tight')
