import array
import functools as ft
import gzip
import os
import struct
import urllib.request

import diffrax as dfx
import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax

from .models.unet import UNet
from .losses.score_matching_loss import make_step, single_loss_fn
from .process.diffusion import VarExpBrownianMotion, ReverseProcess
from .models.save_and_load import save, load