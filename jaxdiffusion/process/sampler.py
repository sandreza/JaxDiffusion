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
    def __init__(self, fwd_process, model, data_shape):
        @eqx.filter_jit
        def drift_precursor(model, fwd_process, t, y, args):
            g = fwd_process.diff(t, y, None)
            g2 = jnp.dot(g, jnp.transpose(g))
            s = - jnp.dot(g2, model(t, y, None) )
            return s
        
        @eqx.filter_jit
        def score_model_precursor(model, data_shape, t, y, args): 
            yr = jnp.reshape(y, data_shape)
            s = model(t, yr)
            return jnp.reshape(s, (math.prod(data_shape)))
        
        score_model = ft.partial(score_model_precursor, model, data_shape)
        drift = ft.partial(drift_precursor, score_model, fwd_process)
        self.fwd_process = fwd_process
        self.score_model = score_model
        self.data_shape = data_shape
        self.drift = drift

    @eqx.filter_jit
    def sample(self, N, *, context = None, key = jr.PRNGKey(12345), steps = 300):
        if context is None:
            subkey = jax.random.split(key, N)
            y0 = jr.normal(subkey[0], (N, math.prod(self.data_shape))) * self.fwd_process.sigma_max # technically wrong
            dt0 = (self.fwd_process.tmin - self.fwd_process.tmax)/ steps

            def wrapper(fwd_process, dt0, y0, key):
                brownian = dfx.VirtualBrownianTree(fwd_process.tmin, fwd_process.tmax, tol=1e-2, shape=y0.shape, key=key)
                solver = dfx.Heun()
                f = dfx.ODETerm(self.drift)
                g = dfx.ControlTerm(fwd_process.diff, brownian)
                terms = dfx.MultiTerm(f, g)
                sol = dfx.diffeqsolve(terms, solver, fwd_process.tmax, fwd_process.tmin, dt0=dt0, y0=y0)
                return sol.ys

            desolver = ft.partial(wrapper, self.fwd_process, dt0)
            samples = jax.vmap(desolver)(y0, subkey)
            samples = jnp.reshape(samples, (N,  *self.data_shape))
            return samples
        else:
            print("Context is not None")
            return None