

import os

import numpy as np
from scipy.integrate import RK45

import matplotlib.pyplot as plt
import imageio.v3 as iio


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


if __name__ == '__main__':
    
    # parameters
    output_folder = './output/'
    os.makedirs(output_folder,exist_ok=True)
    N = 400 # number of vortex
    epsilon = 0.5 # vortex regularization parameter

    dt = 1.0/20 # time-marching timestep, s
    t_end = 4.0 # final time, s

    intial_amplitude = 0.01 # vortex sheet wave amplitude
    domain_size = 2.0
    plot_half_height = 0.25
    
    halfN = 1.0/(2.0*N)
    eps2 = epsilon**2
    twopi = 2*np.pi

    initial_wave = intial_amplitude*np.sin(twopi*np.linspace(0,domain_size,N))

    vortex_x_0 = np.linspace(0,domain_size,N) + initial_wave
    vortex_y_0 = -initial_wave
    vortex_s_0 = np.concatenate((vortex_x_0,vortex_y_0))

    sol = RK45(
        rhs,0.0,vortex_s_0,t_end+dt,
        max_step=dt,
        vectorized=True
        )

    fig, ax = plt.subplots(
        1,1,
        figsize=(16/2.54,6/2.54),
        layout='constrained',
        )
    ax.axis('equal')
    ax.set_ylim([-plot_half_height,plot_half_height])
    ax.set_xlim([0,domain_size])

    i = 0
    files = []
    while sol.t <= t_end:
        vortex_x, vortex_y = sol.y[:N], sol.y[N:]
        ax.plot(
            vortex_x,vortex_y,
            '-ok',
            linewidth=0.75,
            markerfacecolor='white',
            markersize=1,
            )
        ax.set_title(f't = {sol.t:0.2f}')

        filename = f'{output_folder}/solution_{i:04d}.png'
        fig.savefig(filename)
        files.append(filename)

        i += 1
        print(f'{i:>5d} : {sol.t:.5e}')
        sol.step()

        for l in ax.lines: l.remove()


    print(f'Generating gif...')
    frames = np.stack([iio.imread(f) for f in files], axis = 0)
    # make first and last frames to last more so it is easier to follow
    gif_fps = 15
    durations = np.ones(len(frames),dtype=int)*(1000//gif_fps) # in milliseconds
    durations[0] *= 5
    durations[-1] *= 10
    iio.imwrite(
        f'{output_folder}/vortex.gif',
        frames,format='GIF',
        loop=0, # loop forever
        duration=durations.tolist(),
        )