import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from qutip import *

# dimension of finite Hilbert space
N = 75

b  = destroy(N)
nb = num(N)
tlist = np.linspace(0, 10, 100)

H0 = b.dag() * b
H1 = b.dag() * b.dag()
H2 = b * b

coupling = 1.2

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

nb_e = np.zeros(shape(tlist))
nb_s = np.zeros(shape(tlist))
nb_e_lossy = np.zeros(shape(tlist))
nb_s_lossy = np.zeros(shape(tlist))

for idx, psi in enumerate(output.states):
    nb_e[idx] = expect(nb, psi)
    nb_s[idx] = expect(nb*nb, psi)

for idx, psi in enumerate(output_lossy.states):
    nb_e_lossy[idx] = expect(nb, psi)
    nb_s_lossy[idx] = expect(nb*nb, psi)

# substract the average squared to obtain variances
nb_s = nb_s - nb_e ** 2
nb_s_lossy = nb_s_lossy - nb_e_lossy ** 2

fig, axes = plt.subplots(2, 2, sharex=True, figsize=(8,5))

line1 = axes[0,0].plot(tlist, nb_e, 'r', linewidth=2)
axes[0,0].set_ylabel(r'$\langle a^\dagger a \rangle$', fontsize=18)

line2 = axes[0,1].plot(tlist, nb_e_lossy, 'b', linewidth=2)

line3 = axes[1,0].plot(tlist, nb_s, 'r', linewidth=2)
axes[1,0].set_xlabel('$t$', fontsize=18)
axes[1,0].set_ylabel(r'$Std[a^\dagger a]$, $Std[b^\dagger b]$', fontsize=18)

line4 = axes[1,1].plot(tlist, nb_s_lossy, 'b', linewidth=2)
axes[1,1].set_xlabel('$t$', fontsize=18)

plt.savefig('number_variance_strong.png', bbox_inches='tight')
