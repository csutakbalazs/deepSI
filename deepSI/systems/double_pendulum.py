from deepSI.systems.system import System, System_deriv, System_data

import numpy as np
import jax.numpy as jnp

def f_double_pendulum(state, t=0, m1=1, m2=1, l1=1, l2=1, g=9.8):
    t1, t2, w1, w2 = state

    a1 = (l2 / l1) * (m2 / (m1 + m2)) * np.cos(t1 - t2)
    a2 = (l1 / l2) * np.cos(t1 - t2)
    f1 = -(l2 / l1) * (m2 / (m1 + m2)) * (w2**2) * np.sin(t1 - t2) - \
      (g / l1) * np.sin(t1)
    f2 = (l1 / l2) * (w1**2) * np.sin(t1 - t2) - (g / l2) * np.sin(t2)
    g1 = (f1 - a1 * f2) / (1 - a1 * a2)
    g2 = (f2 - a2 * f1) / (1 - a1 * a2)
    return np.stack([w1, w2, g1, g2])


def normalize_double_pendulum(state):
    # wrap generalized coordinates to [-pi, pi]
    return jnp.concatenate([(state[:2] + np.pi) % (2 * np.pi) - np.pi, state[2:]])


class DoublePendulum(System_deriv):

    def __init__(self, x0, m1=1, m2=1, l1=1, l2=1, g=9.8, dt=0.01):
        super(DoublePendulum, self).__init__(dt=dt, nx=4, nu=2, ny=4)
        self.x0 = x0
        self.x = x0
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g
        self.dt = dt

    def reset(self):
        self.x = self.x0
        return self.h(self.x)  # return position

    def get_state(self):
        return self.x

    def deriv(self,x,u):
        return f_double_pendulum(x, m1=self.m1, m2=self.m2, l1=self.l1, l2=self.l2, g=self.g) + u

    def h(self,x):
        t1, t2, w1, w2 = normalize_double_pendulum(x)
        return t1, t2, w1, w2


def double_pendulum_to_video(system_data: System_data, system: DoublePendulum = None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from moviepy.editor import ImageSequenceClip
    from functools import partial
    import proglog
    from PIL import Image

    if system is not None:
        L1, L2 = system.l1, system.l2
        dt = system.dt
    else:
        L1, L2 = 1,1
        dt = 0.04 # conforms 25 fps


    def make_plot(i, cart_coords, l1, l2, max_trail=30, trail_segments=20, r=0.05):
        # Plot and save an image of the double pendulum configuration for time step i.
        plt.cla()

        x1, y1, x2, y2 = cart_coords
        ax.plot([0, x1[i], x2[i]], [0, y1[i], y2[i]], lw=2, c='k')  # rods
        c0 = Circle((0, 0), r / 2, fc='k', zorder=10)  # anchor point
        c1 = Circle((x1[i], y1[i]), r, fc='b', ec='b', zorder=10)  # mass 1
        c2 = Circle((x2[i], y2[i]), r, fc='r', ec='r', zorder=10)  # mass 2
        ax.add_patch(c0)
        ax.add_patch(c1)
        ax.add_patch(c2)

        # plot the pendulum trail (ns = number of segments)
        s = max_trail // trail_segments
        for j in range(trail_segments):
            imin = i - (trail_segments - j) * s
            if imin < 0: continue
            imax = imin + s + 1
            alpha = (j / trail_segments) ** 2  # fade the trail into alpha
            ax.plot(x2[imin:imax], y2[imin:imax], c='r', solid_capstyle='butt',
                    lw=2, alpha=alpha)

        # Center the image on the fixed anchor point. Make axes equal.
        ax.set_xlim(-l1 - l2 - r, l1 + l2 + r)
        ax.set_ylim(-l1 - l2 - r, l1 + l2 + r)
        ax.set_aspect('equal', adjustable='box')
        plt.axis('off')
        # plt.savefig('./frames/_img{:04d}.png'.format(i//di), dpi=72)

    def radial2cartesian(t1, t2, l1, l2):
        # Convert from radial to Cartesian coordinates.
        x1 = l1 * np.sin(t1)
        y1 = -l1 * np.cos(t1)
        x2 = x1 + l2 * np.sin(t2)
        y2 = y1 - l2 * np.cos(t2)
        return x1, y1, x2, y2

    def fig2image(fig):
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return image

    theta1, theta2 = system_data.x[:, 0], system_data.x[:, 1]
    cart_coords = radial2cartesian(theta1, theta2, L1, L2)

    fig = plt.figure(figsize=(8.3333, 6.25), dpi=72)
    ax = fig.add_subplot(111)

    import warnings
    warnings.filterwarnings("ignore")

    images = []
    di = 1
    N = 300
    for i in range(0, N, di):
        print("{}/{}".format(i // di, N // di), end='\n' if i // di % 20 == 0 else ' ')
        make_plot(i, cart_coords, L1, L2)
        images.append(fig2image(fig))

    import importlib
    importlib.reload(proglog)
    proglog.default_bar_logger = partial(proglog.default_bar_logger, None)
    return ImageSequenceClip(images, fps=25)



if __name__=='__main__':
    from matplotlib import pyplot as plt

    sys = DoublePendulum(x0=np.array([3*np.pi/7, 3*np.pi/4, 0, 0], dtype=np.float32))

    exp = System_data(u=np.zeros(shape=(3000, 4), dtype=np.float32))
    train_data = sys.apply_experiment(exp, save_state=True)



    train_data.plot(show=False)
    plt.show()

    videoclip = double_pendulum_to_video(train_data)
    videoclip.write_videofile('double_pendulum.mp4')

