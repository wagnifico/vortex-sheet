


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
field_half_height = 0.25

intial_amplitude = 0.01
initial_wave = intial_amplitude*np.sin(twopi*np.linspace(0,domain_size,N))

vortex_x_0 = np.linspace(0,domain_size,N) + initial_wave
vortex_y_0 = -initial_wave

vortex_s_0 = np.concatenate((vortex_x_0,vortex_y_0))


def calculate_velocity(x,y,vortex_x,vortex_y):
    denominator = 1.0/ \
        (np.cosh(twopi*(y - vortex_y)) - np.cos(twopi*(x - vortex_x)) + eps2)
    # no need to remove self-induction singularity if eps > 0
    # because sinh(0) = sin(0) = 0
    ux = -halfN*np.sum( np.sinh(twopi*(y - vortex_y)) * denominator )
    uy =  halfN*np.sum(  np.sin(twopi*(x - vortex_x)) * denominator )
    return ux, uy

def rhs(t,s,):
    dsdt = np.zeros(2*N)
    for i in range(N):
        dsdt[i], dsdt[N+i] = calculate_velocity(s[i],s[N+i],s[:N],s[N:])

    return dsdt

dt = 1.0/20
t_start = 0.0
t_end = 4.0

sol = RK45(
    rhs,t_start,vortex_s_0,t_end,
    max_step=dt,
    vectorized=True
    )

fig, ax = plt.subplots(
    1,1,layout='constrained',figsize=(16/2.54,6/2.54))

#ax.axis('equal')
ax.set_ylim([-field_half_height,field_half_height])
ax.set_xlim([0,domain_size])

i = 0
t = t_start

files = []
while sol.t <= t_end - dt:
    vortex_x, vortex_y = sol.y[:N], sol.y[N:]
    ax.plot(
        vortex_x,vortex_y,
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

    for l in ax.lines: l.remove()


print(f'Generating gif...')
frames = np.stack([iio.imread(f) for f in files], axis = 0)
iio.imwrite('output/vortex.gif', frames)