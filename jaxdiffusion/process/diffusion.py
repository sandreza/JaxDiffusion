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

class DiffusionProcess(eqx.Module):
    @abstractmethod
    def drift(self, t: Float, y: Float[Array, "n"], args=None) -> Float[Array, "n"]:
        raise NotImplementedError

    @abstractmethod
    def diff(
        self, t: Float, y: Float[Array, "n"], args=None
    ) -> Union[Float, Float[Array, "n n"]]:
        raise NotImplementedError

    @eqx.filter_jit
    def simulate(
            self,
            y0: Float[Array, "n"],
            ts: Float[Array, "nt"],
            dt: Float,
            key: PRNGKeyArray,
            args: Optional[PyTree] = None,
    ):
        tstart, tend = ts[0], ts[-1]

        _, subkey = jrandom.split(key)
        brownian = dfx.VirtualBrownianTree(
            tstart, tend, tol=1e-2, shape=y0.shape, key=subkey
        )
        
        solver = dfx.Heun()
        f = dfx.ODETerm(self.drift)
        g = dfx.ControlTerm(self.diff, brownian)
        terms = dfx.MultiTerm(f, g)

        saveat = dfx.SaveAt(ts=ts)
        sol = dfx.diffeqsolve(
            terms, solver, tstart, tend, dt0=dt, y0=y0, saveat=saveat, args=args
        )

        return sol.ys

class ForwardDiffusion(DiffusionProcess):
    @abstractmethod
    def kernel_mean(self, t: Float, y0: Float[Array, "n"]) -> Float[Array, "n"]:
        raise NotImplementedError

    @abstractmethod
    def kernel_cholesky(self, t: Float, y0: Float[Array, "n"]) -> Float[Array, "n n"]:
        raise NotImplementedError


class VarExpBrownianMotion(ForwardDiffusion):
    sigma_min: Float
    sigma_max: Float
    tmin: Float
    tmax: Float

    def __init__(self, sigma_min: Float, sigma_max: Float):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.tmin = jnp.log(2) / (2 * jnp.log(sigma_max / sigma_min))
        self.tmax = 1.0

    @eqx.filter_jit
    def drift(self, t: Float, y: Float[Array, "n"], args=None) -> Float[Array, "n"]:
        return jax.lax.stop_gradient(jnp.zeros_like(y))

    @eqx.filter_jit
    def diff(
        self, t: Float, y: Float[Array, "n"], args=None
    ) -> Union[Float, Float[Array, "n n"]]:
        return jax.lax.stop_gradient(
            jnp.eye(y.shape[0])
            * self.sigma_min
            * (self.sigma_max / self.sigma_min) ** t
            * jnp.sqrt(2 * (jnp.log(self.sigma_max) - jnp.log(self.sigma_min)))
        )

    @eqx.filter_jit
    def kernel_mean(self, t: Float, y0: Float[Array, "n"]) -> Float[Array, "n"]:
        return jax.lax.stop_gradient(y0)

    @eqx.filter_jit
    def kernel_cholesky(self, t: Float, y0: Float[Array, "n"]) -> Float[Array, "n n"]:
        return jax.lax.stop_gradient(
            jnp.eye(y0.shape[0])
            * self.sigma_min
            * jnp.sqrt((self.sigma_max / self.sigma_min) ** (2*t) - 1)
        )

class ReverseProcess(DiffusionProcess):
    fwd_process: ForwardDiffusion
    score_model: eqx.Module

    def __init__(
        self,
        fwd_process: ForwardDiffusion,
        score_model: eqx.Module,
    ):
        self.fwd_process = fwd_process
        self.score_model = score_model

    @eqx.filter_jit
    def drift(self, t: Float, y: Float[Array, "n"], args=None) -> Float[Array, "n"]:
        # we only ever optimize the score model
        b = self.fwd_process.drift(t, y, args)
        g = self.fwd_process.diff(t, y, args)
        g2 = jnp.dot(g, jnp.transpose(g))
        s = self.score_model(t, y, args)

        return b - jnp.dot(g2, s)

    @eqx.filter_jit
    def diff(self, t: Float, y: Float[Array, "n"], args=None) -> Float[Array, "n n"]:
        return self.fwd_process.diff(t, y, args)

    @eqx.filter_jit
    def drift_aug(
        self, t: Float, y: Float[Array, "n+1"], args=None
    ) -> Float[Array, "n+1"]:
        f = self.drift(t, y[:-1], args)
        s = self.score_model(t, y[:-1], args)
        g = self.diff(t, y[:-1], args)
        f_aug = (
            jnp.square(jnp.dot(jnp.transpose(g), s)).sum(keepdims=True) / 2
        )  # augment drift via "kinetic energy cost"

        return jnp.concatenate([f, f_aug])

    @eqx.filter_jit
    def diff_aug(
        self, t: Float, y: Float[Array, "n+1"], args=None
    ) -> Float[Array, "n+1 n+1"]:
        g = self.diff(t, y[:-1], args)
        g_aug = jnp.zeros((y.shape[0], y.shape[0]))
        g_aug = g_aug.at[:-1, :-1].set(
            g
        )  # augmented diffusion is zero for "kinetic energy cost"

        return g_aug

    @eqx.filter_jit
    def probability_flow_drift(
        self, t: Float, y: Float[Array, "n"], args=None
    ) -> Float[Array, "n"]:
        # we only ever optimize the score model
        b = self.fwd_process.drift(t, y, args)
        g = self.fwd_process.diff(t, y, args)
        s = self.score_model(t, y, args)
        return b - jnp.dot(jnp.dot(g, jnp.transpose(g)), s) / 2

    @eqx.filter_jit
    def simulate_ode(
        self,
        y0: Float[Array, "n"],
        ts: Float[Array, "nt"],
        dt: Float,
        args: Optional[PyTree] = None,
    ):
        tstart, tend = ts[0], ts[-1]
        term = dfx.ODETerm(self.probability_flow_drift)

        solver = dfx.Tsit5()
        saveat = dfx.SaveAt(ts=ts)
        sol = dfx.diffeqsolve(
            term, solver, tstart, tend, dt0=dt, y0=y0, saveat=saveat, args=args
        )

        return sol.ys

    @eqx.filter_jit
    def simulate_aug(
        self,
        y0: Float[Array, "n+1"],
        ts: Float[Array, "nt"],
        dt: Float,
        key: PRNGKeyArray,
        args: Optional[PyTree] = None,
    ):
        tstart, tend = ts[0], ts[-1]

        _, subkey = jrandom.split(key)
        brownian = dfx.VirtualBrownianTree(
            tstart, tend, tol=1e-2, shape=y0.shape, key=subkey
        )
        f = dfx.ODETerm(self.drift_aug)
        g = dfx.ControlTerm(self.diff_aug, brownian)
        terms = dfx.MultiTerm(f, g)

        solver = dfx.ReversibleHeun()
        saveat = dfx.SaveAt(ts=ts)
        sol = dfx.diffeqsolve(
            terms, solver, tstart, tend, dt0=dt, y0=y0, saveat=saveat, args=args
        )

        return sol.ys
