from deepSI.fit_systems import System_fittable
from deepSI.systems import System_deriv
from deepSI.system_data import System_data

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt
from functools import partial  # reduces arguments to function by making some subset implicit
from jax.experimental import stax
from jax.experimental import optimizers
from numpy import save
from numpy import load


# Building blocks

def rk4_step(f, x, u, t, h):
    # one step of runge-kutta integration
    k1 = h * (f(x, t) + u)
    k2 = h * (f(x + k1 / 2, t + h / 2) + u)
    k3 = h * (f(x + k2 / 2, t + h / 2) + u)
    k4 = h * (f(x + k3, t + h) + u)
    return x + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# Building blocks of the LNN
def learned_hamiltonian(nn_params, nn_forward_fn, normalize_state):
    def hamiltonian(q, q_t):
        assert q.shape == (2,)
        state = normalize_state(jnp.concatenate([q, q_t]))
        return jnp.squeeze(nn_forward_fn(nn_params, state), axis=-1)

    return hamiltonian

def create_permutation_tensor(n):
    M = None
    '''Constructs the Levi-Civita permutation tensor'''
    M = np.ones(n)  # matrix of ones
    M = M - np.eye(n)  # clear diagonals
    M[::2] *= -1  # pattern of signs
    M[:, ::2] *= -1

    for i in range(n):  # make asymmetric
        for j in range(i + 1, n):
            M[i, j] *= -1
    return M


# this is called time derivative in the original article
def f_from_hamiltonian(hamiltonian, state, t=None):

    q, q_t = jnp.split(state, 2)
    #q_tt = (jnp.linalg.pinv(jax.hessian(lagrangian, 1)(q, q_t))
    #        @ (jax.grad(lagrangian, 0)(q, q_t)
    #           - jax.jacobian(jax.jacobian(lagrangian, 1), 0)(q, q_t) @ q_t))

    #q_tt = jax.grad(hamiltonian)(q, q_t) @ f_from_hamiltonian.M # gradients for solenoidal field
    q_tt = -jax.grad(hamiltonian, 1)(q, q_t) @ f_from_hamiltonian.M


    return jnp.concatenate([q_t, q_tt])


f_from_hamiltonian.M = create_permutation_tensor(2)

@jax.partial(jax.jit, static_argnames=('nn_forward_fn', 'normalize_state', 'time_step'))
def loss(batch, nn_params, nn_forward_fn, normalize_state, time_step=None):
    state, u, targets = batch
    # if time_step is not None:
    assert time_step is not None
    f_only = partial(f_from_hamiltonian, learned_hamiltonian(nn_params, nn_forward_fn, normalize_state))

    predictions = jax.vmap(partial(rk4_step, f_only, t=0.0, h=time_step))(x=state, u=u)
    # else:
    #     predictions = jax.vmap(partial(f_from_lagrangian, learned_lagrangian(nn_params, nn_forward_fn, normalize_state)))(state)
    return jnp.mean((predictions - targets) ** 2)


@jax.partial(jax.jit, static_argnames=('nn_forward_fn', 'normalize_state', 'opt_update', 'get_params'))
def update_derivative(iteration, batch, nn_forward_fn, normalize_state, opt_state, opt_update, get_params):
    params = get_params(opt_state)
    return opt_update(iteration,
                      jax.grad(loss, argnums=1)(batch, params, nn_forward_fn, normalize_state, time_step=0.01),
                      opt_state)


def do_nothing(param):
    return param


class HNN(System_fittable):

    def __init__(self, normalization_fn=None, **kwargs):
        super(HNN, self).__init__(**kwargs)

        # create a neural network
        self.init_random_params, self.nn_forward_fn = stax.serial(
            stax.Dense(128),
            stax.Softplus,
            stax.Dense(128),
            stax.Softplus,
            stax.Dense(1),
        )

        # Parameters of the neural networks - will be initialized later
        self.params = None

        # Normalization function TODO: probably not the best name, change
        if normalization_fn is None:
            self.normalize_state = do_nothing
        else:
            self.normalize_state = normalization_fn



    # TODO: timestep is hardcoded!
    def fit_and_validate(self, train_data: System_data, test_data=None, batch_size=100, test_every=10,
                         num_batches=1500):

        x_train = jax.device_put(self.normalize_state(train_data.x[0:-1, :]))
        u_train = jax.device_put(train_data.u[0:-1, :])
        y_train = jax.device_put(self.normalize_state(train_data.x[1:, :]))

        if test_data is not None:
            x_test = jax.device_put(self.normalize_state(test_data.x[0:-1, :]))
            u_test = jax.device_put(test_data.u[0:-1, :])
            y_test = jax.device_put(self.normalize_state(test_data.x[1:, :]))

        # Randomly initialize parameters
        rng = jax.random.PRNGKey(0)
        _, init_params = self.init_random_params(rng, (-1, self.nx))

        # Save train losses for analysis
        train_losses = []
        test_losses = []

        # adam with learn rate decay
        opt_init, opt_update, get_params = optimizers.adam(
            lambda t: jnp.select([t < batch_size * (num_batches // 3),
                                  t < batch_size * (2 * num_batches // 3),
                                  t > batch_size * (2 * num_batches // 3)],
                                 [1e-3, 3e-4, 1e-4]))
        opt_state = opt_init(init_params)

        for iteration in range(batch_size * num_batches + 1):
            if iteration % batch_size == 0:
                params = get_params(opt_state)

                train_loss = loss(batch=(x_train, u_train, y_train),
                                  nn_params=params, nn_forward_fn=self.nn_forward_fn,
                                  normalize_state=self.normalize_state, time_step=0.01)
                train_losses.append(train_loss)

                if test_data is not None:
                    test_loss = loss(batch=(x_test, u_test, y_test),
                                     nn_params=params, nn_forward_fn=self.nn_forward_fn,
                                     normalize_state=self.normalize_state, time_step=0.01)

                else:
                    test_loss = np.nan
                test_losses.append(test_loss)

                if iteration % (batch_size * test_every) == 0:
                    print(f"iteration={iteration}, train_loss={train_loss:.6f}, test_loss={test_loss:.6f}")

            opt_state = update_derivative(iteration=iteration, batch=(x_train, u_train, y_train),
                                          nn_forward_fn=self.nn_forward_fn, normalize_state=self.normalize_state,
                                          opt_state=opt_state, opt_update=opt_update, get_params=get_params)

        # save network parameters
        self.params = get_params(opt_state)

    def save_params(self, filename):
        jnp.save(filename, self.params)

    def load_params(self, filename):
        self.params = jnp.load(filename, allow_pickle=True)


class HNN_system(HNN, System_deriv):
    def __init__(self, x0, nx, nu, ny, dt=0.01, normalization_fn=None):
        # LNN.__init__(self, normalization_fn=normalization_fn, nx=nx, nu=nu, ny=ny, dt=dt)
        super(HNN_system, self).__init__(nx=nx, nu=nu, ny=ny, dt=dt, normalization_fn=normalization_fn)

        self.x0 = x0
        self.x = x0

    def reset(self):
        self.x = self.x0
        return self.h(self.x)

    def get_state(self):
        return self.x

    def deriv(self, x, u):
        return (f_from_hamiltonian(hamiltonian=learned_hamiltonian(nn_params=self.params,
                                                                nn_forward_fn=self.nn_forward_fn,
                                                                normalize_state=self.normalize_state),
                                  state=x,
                                  ) + u)

    def h(self, x):
        return self.normalize_state(x)
