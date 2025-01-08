from abc import abstractmethod
from typing import Optional, Union

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

import math
import diffrax as dfx
import functools as ft
import jax.random as jr

class Sampler:
    def __init__(self, schedule, model, data_shape):
        @eqx.filter_jit
        def drift_precursor(model, schedule, t, y, args):
            g2 = schedule.g2(t)
            s = - g2 * model(t, y, None) 
            return s
        
        @eqx.filter_jit
        def diffusion_precursor(sigma, t, y):
            return sigma(t) * jnp.eye(y.shape[0])
        
        @eqx.filter_jit
        def score_model_precursor(model, data_shape, t, y, args): 
            yr = jnp.reshape(y, data_shape)
            s = model(t, yr)
            return jnp.reshape(s, (math.prod(data_shape)))
        
        score_model = ft.partial(score_model_precursor, model, data_shape)
        drift = ft.partial(drift_precursor, score_model, schedule)
        diffusion = ft.partial(diffusion_precursor, schedule.sigma)
        self.schedule = schedule
        self.score_model = score_model
        self.data_shape = data_shape
        self.drift = drift
        self.diffusion = diffusion

    @eqx.filter_jit
    def sample(self, N, *, context = None, key = jr.PRNGKey(12345), steps = 300):
        if context is None:
            subkey = jax.random.split(key, N)
            y0 = jr.normal(subkey[0], (N, math.prod(self.data_shape))) * self.schedule.sigma_max # technically wrong
            dt0 = (self.schedule.tmin - self.schedule.tmax)/ steps

            def wrapper(schedule, dt0, y0, key):
                brownian = dfx.VirtualBrownianTree(schedule.tmin, schedule.tmax, tol=1e-2, shape=y0.shape, key=key)
                solver = dfx.Heun()
                f = dfx.ODETerm(self.drift)
                g = dfx.ControlTerm(self.diffusion, brownian)
                terms = dfx.MultiTerm(f, g)
                sol = dfx.diffeqsolve(terms, solver, schedule.tmax, schedule.tmin, dt0=dt0, y0=y0)
                return sol.ys

            desolver = ft.partial(wrapper, self.schedule, dt0)
            samples = jax.vmap(desolver)(y0, subkey)
            samples = jnp.reshape(samples, (N,  *self.data_shape))
            return samples
        else:
            print("Context is not None")
            return None