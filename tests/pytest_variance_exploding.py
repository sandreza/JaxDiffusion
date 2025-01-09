import pytest
import jax.numpy as jnp
import numpy as np
from jaxdiffusion.process.schedule import VarianceExplodingBrownianMotion

@pytest.fixture
def schedule():
    return VarianceExplodingBrownianMotion(0.1, 2.0)

def test_sigma_at_zero(schedule):
    np.testing.assert_allclose(schedule.sigma(0), 0, rtol=1e-7, atol=1e-7)

def test_sigma_at_one(schedule):
    np.testing.assert_allclose(schedule.sigma(1), jnp.sqrt(2**2 - 0.1**2), rtol=1e-7, atol=1e-7)

def test_g2(schedule):
    np.testing.assert_allclose(schedule.g2(0.5), schedule.g(0.5) ** 2, rtol=1e-7, atol=1e-7)

def test_sigma_min(schedule):
    np.testing.assert_allclose(schedule.sigma(schedule.tmin), schedule.sigma_min, rtol=1e-7, atol=1e-7)

def test_tmin_positive(schedule):
    assert schedule.tmin > 0