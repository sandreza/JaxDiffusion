import jax.numpy as jnp
import equinox as eqx

class VarianceExplodingBrownianMotion():
    def __init__(self, sigma_min, sigma_max):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.tmin = jnp.log(2) / (2 * jnp.log(sigma_max / sigma_min))
        self.tmax = 1.0

    @eqx.filter_jit
    def g(self, t):
        dsigmasqd = self.sigma_min * (self.sigma_max / self.sigma_min) ** t * jnp.sqrt(2 * (jnp.log(self.sigma_max / self.sigma_min)))
        return dsigmasqd
    
    @eqx.filter_jit
    def g2(self, t):
        dsigmasqd = (self.sigma_min**2) * (self.sigma_max / self.sigma_min) ** (2 * t) * (2 * (jnp.log(self.sigma_max / self.sigma_min)))
        return dsigmasqd

    @eqx.filter_jit
    def sigma(self, t):
        return self.sigma_min * jnp.sqrt((self.sigma_max / self.sigma_min) ** (2*t) - 1)
