"""
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
from jaxdiffusion.losses.score_matching_loss import make_step, single_loss_fn
from jaxdiffusion.process.diffusion import VarExpBrownianMotion, ReverseProcess
from jaxdiffusion.models.save_and_load import save, load
"""

from jaxdiffusion import *

def mnist():
    filename = "train-images-idx3-ubyte.gz"
    url_dir = "https://storage.googleapis.com/cvdf-datasets/mnist"
    # target_dir = os.getcwd() + "/data/mnist"
    target_dir = "/orcd/home/001/sandre/Repositories/JaxUvTest/data/mnist"
    url = f"{url_dir}/{filename}"
    target = f"{target_dir}/{filename}"
    if not os.path.exists(target):
        os.makedirs(target_dir, exist_ok=True)
        urllib.request.urlretrieve(url, target)
        print(f"Downloaded {url} to {target}")
    with gzip.open(target, "rb") as fh:
        _, batch, rows, cols = struct.unpack(">IIII", fh.read(16))
        shape = (batch, 1, rows, cols)
        return jnp.array(array.array("B", fh.read()), dtype=jnp.uint8).reshape(shape)

def dataloader(data, batch_size, key):
    dataset_size = data.shape[0]
    indices = jnp.arange(dataset_size)
    while True:
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, indices)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield data[batch_perm]
            start = end
            end = start + batch_size

data = mnist() 
data_mean = jnp.mean(data)
data_std = jnp.std(data)
data_max = jnp.max(data)
data_min = jnp.min(data)
data_shape = data.shape[1:]
data = (data - data_mean) / data_std 


# add util for calculating the distance
key = jr.PRNGKey(0)
key, subkey = jax.random.split(key)
random_index_1 = jax.random.randint(key, 10, 0, data.shape[0]-1)
key, subkey = jax.random.split(key)
random_index_2 = jax.random.randint(key, 10, 0, data.shape[0]-1)

tmp = jnp.linalg.norm(data[random_index_1, 0, :, :] - data[random_index_2, 0, :, :], axis=(1, 2))
sigma_max = max(tmp) 
sigma_min = 1e-2
key, subkey = jax.random.split(key)
fwd_process = VarExpBrownianMotion(sigma_min, sigma_max) 

seed = 12345

unet_hyperparameters = {
    "data_shape": (1, 28, 28),      # Single-channel grayscale MNIST images
    "is_biggan": True,              # Whether to use BigGAN architecture
    "dim_mults": [1, 2, 4],         # Multiplicative factors for UNet feature map dimensions
    "hidden_size": 16,              # Base hidden channel size
    "heads": 8,                     # Number of attention heads
    "dim_head": 7,                  # Size of each attention head
    "num_res_blocks": 4,            # Number of residual blocks per stage
    "attn_resolutions": [28, 14]    # Resolutions for applying attention
}

key = jr.PRNGKey(seed)

if os.path.exists("test_save.mo"):
    print("Loading file test_save.mo")
    model = load("test_save.mo", UNet)
    print("Done Loading model")
else:
    print("File test_save.mo does not exist. Creating UNet")
    model = UNet(key = key, **unet_hyperparameters)
    print("Done Creating UNet")

model = load("test_save.mo", UNet)

# Optimisation hyperparameters
num_steps=1000
lr=3e-4
batch_size=32
print_every=100
# Sampling hyperparameters
dt0= 0.05
sample_size=10
# Seed
seed=5678


opt = optax.adam(lr)
# Optax will update the floating-point JAX arrays in the model.
opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

model_key, train_key, loader_key, sample_key = jr.split(key, 4) 

total_value = 0
total_size = 0
for step, data in zip(
    range(num_steps), dataloader(data, batch_size, key=loader_key)
):
    value, model, train_key, opt_state = make_step(
        model, fwd_process, data, train_key, opt_state, opt.update
    )
    total_value += value.item()
    total_size += 1
    if (step % print_every) == 0 or step == num_steps - 1:
        print(f"Step={step} Loss={total_value / total_size}")
        total_value = 0
        total_size = 0
        save("test_save.mo", unet_hyperparameters, model)


@eqx.filter_jit
def score_model_precursor(model, t, y, args): 
    yr = jnp.reshape(y, (1, 28, 28))
    s = model(t, yr)
    return jnp.reshape(s, (28**2))

score_model = ft.partial(score_model_precursor, model)

rev_process = ReverseProcess(fwd_process, score_model)

@eqx.filter_jit
def drift_precursor(model, fwd_process, t, y, args):
    g = fwd_process.diff(t, y, None)
    g2 = jnp.dot(g, jnp.transpose(g))
    s = - jnp.dot(g2, model(t, y, None) )
    return s

drift = ft.partial(drift_precursor, score_model, fwd_process)

key, subkey = jax.random.split(key)
y0 = jr.normal(subkey, data[0:9, :, :, :].shape) * sigma_max
y0r = jnp.reshape(y0, (9 , 28**2))

brownian = dfx.VirtualBrownianTree(
    fwd_process.tmin, fwd_process.tmax, tol=1e-2, shape=y0r[0, :].shape, key=subkey
)

dt0 = (fwd_process.tmin - fwd_process.tmax)/ 300
solver = dfx.Heun()
f = dfx.ODETerm(drift)
g = dfx.ControlTerm(fwd_process.diff, brownian)
terms = dfx.MultiTerm(f, g)

sol = dfx.diffeqsolve(
    terms, solver, fwd_process.tmax, fwd_process.tmin, dt0=dt0, y0=y0r[0, :]
)

def wrapper(terms, solver, fwd_process, dt0, y0r):
    sol = dfx.diffeqsolve(terms, solver, fwd_process.tmax, fwd_process.tmin, dt0=dt0, y0=y0r)
    return sol.ys

desolver = ft.partial(wrapper, terms, solver, fwd_process, dt0)

tmp = jax.vmap(desolver)(y0r)

sample = jnp.reshape(tmp[0:9, :], (3, 3, 28, 28))
sample = jnp.transpose(sample, (0, 2, 1, 3))
sample = jnp.reshape(sample, (3 * 28, 3 * 28))

sample = data_mean + data_std * sample
sample = jnp.clip(sample, data_min, data_max)

plt.figure()
plt.imshow(sample, cmap="Greys")
plt.axis("off")
plt.tight_layout()
plt.show()
filename = "mnist_diffusion_unet_quax.png"
plt.savefig(filename)