

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45

N = 100 # number of vortex
epsilon = 0.005 # regularization factor (as proposed by Krasny)

domain_size = 2.0

x_0 = np.linspace(-domain_size/2,domain_size/2,N)
y_0 = 0*x_0
s_0 = np.concatenate((x_0,y_0))

def rhs(t,s,):
    x, y = s[:N], s[N:]
    dsdt = np.zeros(2*N)

    for i in range(N):
        x_i, y_i = x[i], y[i]
        
        # not working, probably due to self-induction singularity
        # to fix!
        dsdt[i] = (-1/(2*N))*np.sum(
                np.sinh(2*np.pi*(y_i - y)) / (
                    np.cosh(2*np.pi*(y_i - y)) - np.cos(2*np.pi*(x_i - x))
                        + epsilon**2)
            )
        
        dsdt[N+i] = (1/(2*N))*np.sum(
                np.sin(2*np.pi*(x_i - x)) / (
                    np.cosh(2*np.pi*(y_i - y)) - np.cos(2*np.pi*(x_i - x))
                        + epsilon**2)
            )

    return dsdt

dt = 1/100
t_start = 0.0
t_end = 100*dt

sol = RK45(
    rhs,t_start,s_0,t_end,
    #max_step=dt,
    vectorized=True
    )


fig, ax = plt.subplots(1,1,layout='constrained')
# ax.set_xlim([-1,1])
ax.set_ylim([-0.01,0.01])

i = 0
t = t_start
#while sol.t <= t_end:
while i <= 1000:
    ax.plot(sol.y[:N],sol.y[N:],'-o')
    fig.savefig(f'output/solution_{i:04d}.png')
    ax.lines[0].remove()

    i += 1
    print(f'{i:5>d} : {sol.t:.5e}')
    sol.step()
