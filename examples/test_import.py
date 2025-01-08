import array
import functools as ft
import gzip
import os
import struct
import urllib.request

import diffrax as dfx  # https://github.com/patrick-kidger/diffrax
import einops  # https://github.com/arogozhnikov/einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax

from jaxdiffusion.models.unet import UNet
from jaxdiffusion.models.save_and_load import save, load, load_hyperparameters

seed = 12345

unet_hyperparameters = {
    "data_shape": [1, 28, 28],      # Single-channel grayscale MNIST images
    "is_biggan": True,              # Whether to use BigGAN architecture
    "dim_mults": [1, 2, 4],         # Multiplicative factors for UNet feature map dimensions
    "hidden_size": 16,              # Base hidden channel size
    "heads": 8,                     # Number of attention heads
    "dim_head": 7,                  # Size of each attention head
    "num_res_blocks": 4,            # Number of residual blocks per stage
    "attn_resolutions": [28, 14]    # Resolutions for applying attention
}

key = jr.PRNGKey(seed)
model = UNet(key = key, **unet_hyperparameters)
save("test_save.mo", unet_hyperparameters, model)
model_loaded = load("test_save.mo", UNet)

data = jnp.ones((1, 28, 28))
tmp1 = model_loaded(1.0, data)
tmp2 = model(1.0, data)
jnp.linalg.norm(tmp1 - tmp2)

hyperparameters_loaded = load_hyperparameters("test_save.mo")

hyperparameters_loaded == unet_hyperparameters