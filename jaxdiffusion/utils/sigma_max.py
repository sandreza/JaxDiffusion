import jax 
import jax.numpy as jnp

def compute_sigma_max(data):
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    random_index_1 = jax.random.randint(key, 10, 0, data.shape[0]-1)
    key, subkey = jax.random.split(key)
    random_index_2 = jax.random.randint(key, 10, 0, data.shape[0]-1)
    sigma_max = jnp.linalg.norm(data[random_index_1, 0, :, ] - data[random_index_2, 0, :, :], axis=(1, 2))
    return sigma_max