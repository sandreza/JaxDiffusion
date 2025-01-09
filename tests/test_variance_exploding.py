import jax.numpy as jnp
import numpy as np
import equinox as eqx
from jaxdiffusion.process.schedule import VarianceExplodingBrownianMotion


schedule = VarianceExplodingBrownianMotion(0.1, 2.0)

np.testing.assert_allclose(schedule.sigma(0), 0, rtol=1e-7, atol=1e-7)
np.testing.assert_allclose(schedule.sigma(1), jnp.sqrt(2**2 - 0.1**2), rtol=1e-7, atol=1e-7)
np.testing.assert_allclose(schedule.g2(0.5), schedule.g(0.5) ** 2, rtol=1e-7, atol=1e-7)
np.testing.assert_allclose(schedule.sigma(schedule.tmin), schedule.sigma_min, rtol=1e-7, atol=1e-7)
schedule.tmin > 0