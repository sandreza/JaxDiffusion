import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

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
