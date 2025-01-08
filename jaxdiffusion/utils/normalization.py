import jax.numpy as jnp

def mean_std_normalize(images):
    data_mean = jnp.mean(images)
    data_std = jnp.std(images)
    data = (images - data_mean) / data_std 
    return data