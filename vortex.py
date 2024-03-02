


import numpy as np
from scipy.integrate import RK45

import matplotlib.pyplot as plt
import imageio.v3 as iio

N = 400 # number of vortex
epsilon = 0.5

halfN = 1.0/(2.0*N)
eps2 = epsilon**2
twopi = 2*np.pi

domain_size = 2.0

intial_amplitude = 0.01
initial_wave = intial_amplitude*np.sin(twopi*np.linspace(0,domain_size,N))

x_0 = np.linspace(0,domain_size,N) + initial_wave
y_0 = -initial_wave

s_0 = np.concatenate((x_0,y_0))


def rhs(t,s,):
    x, y = s[:N], s[N:]
    dsdt = np.zeros(2*N)

    for i in range(N):
        denominator = 1.0/ \
            (np.cosh(twopi*(y[i] - y)) - np.cos(twopi*(x[i] - x)) + eps2)
        # no need to remove self-induction singularity if eps > 0
        # sinh(0) = sin(0) = 0
        dsdt[i]   = -halfN*np.sum( np.sinh(twopi*(y[i] - y)) * denominator )
        dsdt[N+i] =  halfN*np.sum(  np.sin(twopi*(x[i] - x)) * denominator )

    return dsdt

dt = 1.0/20
t_start = 0.0
t_end = 4.0

sol = RK45(
    rhs,t_start,s_0,t_end,
    max_step=dt,
    vectorized=True
    )

fig, ax = plt.subplots(
    1,1,layout='constrained',figsize=(16/2.54,6/2.54))

ax.axis('equal')
ax.set_ylim([-0.3,0.3])
ax.set_xlim([0,domain_size])

i = 0
t = t_start

files = []
while sol.t <= t_end - dt:
    ax.plot(
        sol.y[:N],sol.y[N:],
        '-ok',
        linewidth=0.75,
        markersize=2,
        )
    ax.set_title(f't = {sol.t:0.3f}')

    filename = f'output/solution_{i:04d}.png'
    fig.savefig(filename)
    files.append(filename)

    i += 1
    print(f'{i:>5d} : {sol.t:.5e}')
    sol.step()

    ax.lines[0].remove()


frames = np.stack([iio.imread(f) for f in files], axis = 0)

print(f'Generating gif...')
iio.imwrite('output/vortex.gif', frames)