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
import lineax as lx

class VelocitySampler:
    def __init__(self, schedule, model, data_shape):
        @eqx.filter_jit
        def drift_precursor(model, schedule, t, y, args):
            g2 = schedule.g2(t)
            sigma =  schedule.sigma(t)
            scaling = 1 + sigma
            # score(y) = -y / std**2 * alpha + beta * nn(y / scaling) / (std**2)
            # score(data + std * noise) = - data / std**2 * alpha - noise / std *alpha + ... 
            # loss = -data/std**2 * alpha  + (1-alpha) * Z / std + beta * nn / std**2 
            # weight = std**2 / beta
            # weight * loss = -data *alpha/beta + (1-alpha)/beta * std* Z + nn 
            # alpha -> 0 as std -> 0, alpha -> 1 as std -> inf
            # beta -> O(std) as std -> 0, beta -> O(1) as std -> inf
            # alpha = std / (1 + std), beta = std / (1 + std)
            # 1 - alpha = 1 / (1 + std)
            # w1 = alpha / beta 
            # w2 = (1 - alpha) / beta * std
            # weight * loss = -data * w1 + Z * w2 + nn
            # weight * loss = -data + Z  + nn
            # weight(t) = std**2 / (1 + std)
            prefactor = g2 / (sigma**2)
            alpha = sigma / (1 + sigma)
            beta = sigma / (1 + sigma)
            tau = jnp.log(sigma) / 4
            scaled_score = - y * alpha + beta * model(tau , y / scaling, None) 
            s = - prefactor * scaled_score
            return s
        
        @eqx.filter_jit
        def diffusion_precursor(g, t, y, args = None):
            return lx.DiagonalLinearOperator(g(t) * jnp.ones(y.shape[0]))
        
        @eqx.filter_jit
        def score_model_precursor(model, data_shape, t, y, args): 
            yr = jnp.reshape(y, data_shape)
            s = model(t, yr)
            return jnp.reshape(s, (math.prod(data_shape)))
        
        score_model = ft.partial(score_model_precursor, model, data_shape)
        drift = ft.partial(drift_precursor, score_model, schedule)
        diffusion = ft.partial(diffusion_precursor, schedule.g)
        self.schedule = schedule
        self.score_model = score_model
        self.data_shape = data_shape
        self.drift = drift
        self.diffusion = diffusion

    @eqx.filter_jit
    def precursor_desolver(self, dt0, y0, key):
        brownian = dfx.VirtualBrownianTree(self.schedule.tmin, self.schedule.tmax, tol=1e-2, shape=y0.shape, key=key)
        solver = dfx.Heun()
        f = dfx.ODETerm(self.drift)
        g = dfx.ControlTerm(self.diffusion, brownian)
        terms = dfx.MultiTerm(f, g)
        sol = dfx.diffeqsolve(terms, solver, self.schedule.tmax, self.schedule.tmin, dt0=dt0, y0=y0)
        return sol.ys
    
    @eqx.filter_jit
    def sample(self, N, *, key = jr.PRNGKey(12345), steps = 300):
        subkey = jax.random.split(key, N)
        y0 = jr.normal(subkey[0], (N, math.prod(self.data_shape))) * self.schedule.sigma(1.0) 
        dt0 = (self.schedule.tmin - self.schedule.tmax)/ steps
        desolver = ft.partial(self.precursor_desolver, dt0)
        samples = jax.vmap(desolver)(y0, subkey)
        samples = jnp.reshape(samples, (N,  *self.data_shape))
        return samples
    

class VelocityODESampler:
    def __init__(self, schedule, model, data_shape, * , solver = dfx.Heun()):
        @eqx.filter_jit
        def drift_precursor(model, schedule, t, y, args):
            g2 = schedule.g2(t)
            sigma =  schedule.sigma(t)
            scaling = 1 + sigma
            # score(y) = -y / std**2 * alpha + beta * nn(y / scaling) / (std**2)
            # score(data + std * noise) = - data / std**2 * alpha - noise / std *alpha + ... 
            # loss = -data/std**2 * alpha  + (1-alpha) * Z / std + beta * nn / std**2 
            # weight = std**2 / beta
            # weight * loss = -data *alpha/beta + (1-alpha)/beta * std* Z + nn 
            # alpha -> 0 as std -> 0, alpha -> 1 as std -> inf
            # beta -> O(std) as std -> 0, beta -> O(1) as std -> inf
            # alpha = std / (1 + std), beta = std / (1 + std)
            # 1 - alpha = 1 / (1 + std)
            # w1 = alpha / beta 
            # w2 = (1 - alpha) / beta * std
            # weight * loss = -data * w1 + Z * w2 + nn
            # weight * loss = -data + Z + nn
            # weight(t) = std**2 / (1 + std)
            prefactor = g2 / (2 * sigma**2)
            alpha = sigma / (1 + sigma)
            beta = sigma / (1 + sigma)
            tau = jnp.log(sigma) / 4
            scaled_score = - y * alpha + beta * model(tau, y / scaling, None) 
            s = - prefactor * scaled_score
            return s
        
        @eqx.filter_jit
        def score_model_precursor(model, data_shape, t, y, args): 
            yr = jnp.reshape(y, data_shape)
            s = model(t, yr)
            return jnp.reshape(s, (math.prod(data_shape)))
        
        score_model = ft.partial(score_model_precursor, model, data_shape)
        drift = ft.partial(drift_precursor, score_model, schedule)
        self.schedule = schedule
        self.score_model = score_model
        self.data_shape = data_shape
        self.drift = drift
        self.solver = solver

    @eqx.filter_jit
    def precursor_desolver(self, dt0, y0, key):
        f = dfx.ODETerm(self.drift)
        sol = dfx.diffeqsolve(f, self.solver, self.schedule.tmax, self.schedule.tmin, dt0=dt0, y0=y0)
        return sol.ys
    
    @eqx.filter_jit
    def sample(self, N, *, key = jr.PRNGKey(12345), steps = 300):
        subkey = jax.random.split(key, N)
        y0 = jr.normal(subkey[0], (N, math.prod(self.data_shape))) * self.schedule.sigma(1.0) 
        dt0 = (self.schedule.tmin - self.schedule.tmax)/steps
        desolver = ft.partial(self.precursor_desolver, dt0)
        samples = jax.vmap(desolver)(y0, subkey)
        samples = jnp.reshape(samples, (N,  *self.data_shape))
        return samples