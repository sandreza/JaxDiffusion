import math
from collections.abc import Callable
from typing import Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

# Utility functions
def downsampling_padding(dims, factor):
    return tuple(([0, factor - 1], ) * dims)

def upsampling_padding(data_shape, factor):
    # (y + factor-1) // factor is the assumed size of the previously downscaled image
    # the -y is to get the right padding size
    return tuple([0, factor * ((y + factor-1) // factor) - y] for y in data_shape[1:])  

def phf(y, factor, kernel_size):
    return factor * ((y + factor-1) // factor) - y

def upsampling_padding_two(data_shape, factor, kernel_size):
    return tuple([phf(y, factor, kernel_size) // 2 + (phf(y, factor, kernel_size) - 1) % 2, phf(y, factor, kernel_size) // 2] for y in data_shape[1:])  

def smoothing_padding(dims, factor):
    left_pad  = (factor - 1)//2 
    right_pad = (factor - 1)//2 + (factor - 1)%2
    return tuple(([left_pad, right_pad], ) * dims)

class GaussianFourierProjection(eqx.Module):
    emb: jax.Array

    def __init__(self, dim):
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        self.emb = jnp.exp(jnp.arange(half_dim) * -emb)

    def __call__(self, x):
        emb = x * self.emb
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        return emb

class ConvBlock(eqx.Module):
    norm: eqx.nn.GroupNorm
    conv: eqx.nn.Conv
    def __init__(self, *,  dims, kernel_size=3, padding_mode = 'REPLICATE', min_group_norm_channel = 32, in_channels, out_channels, key = jr.PRNGKey(0)):
        key1, key2, key3 = jr.split(key, 3)
        padding = smoothing_padding(dims, kernel_size)
        self.norm = eqx.nn.GroupNorm(min(in_channels//4, min_group_norm_channel), in_channels)
        self.conv = eqx.nn.Conv(dims, in_channels, out_channels, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, key=key2)
    def __call__(self, x): 
        h = self.norm(x)
        h = jax.nn.silu(h)
        h = self.conv(h)
        return h

class ResnetBlock(eqx.Module):
    conv_block1: ConvBlock
    conv_block2: ConvBlock
    conv: eqx.nn.Conv
    linear: eqx.nn.Linear
    def __init__(self, *, dims = 2, kernel_size=3, padding_mode = 'REPLICATE', min_group_norm_channel = 32, in_channels, out_channels, embed_dim=256, key = jr.PRNGKey(0)):
        key1, key2, key3, key4 = jr.split(key, 4)
        self.conv_block1 = ConvBlock(dims = dims, kernel_size=kernel_size, padding_mode = padding_mode, min_group_norm_channel = min_group_norm_channel, in_channels = in_channels, out_channels = out_channels, key = key1)
        self.conv_block2 = ConvBlock(dims = dims, kernel_size=kernel_size, padding_mode = padding_mode, min_group_norm_channel = min_group_norm_channel, in_channels = out_channels, out_channels = out_channels, key = key2)
        self.conv = eqx.nn.Conv(dims, in_channels, out_channels, kernel_size = 1, key = key3)
        self.linear = eqx.nn.Linear(embed_dim, out_channels, key=key4)
    def __call__(self, t, x):
        h = self.conv_block1(x)
        t = self.linear(jax.nn.silu(t))
        h = h + jnp.expand_dims(t, axis=tuple(range(1, h.ndim)))
        h = self.conv_block2(h)
        h = h + self.conv(x)
        return h
    
class ResnetBlockDown(eqx.Module):
    norm1: eqx.nn.GroupNorm
    conv1: ConvBlock
    down: eqx.nn.Conv
    conv_block2: ConvBlock
    conv: eqx.nn.Conv
    linear: eqx.nn.Linear
    def __init__(self, *, dims = 2, kernel_size=3, padding_mode = 'REPLICATE', min_group_norm_channel = 32, in_channels, out_channels, embed_dim=256, key = jr.PRNGKey(0)):
        key0, key1, key2, key3, key4 = jr.split(key, 5)
        padding = smoothing_padding(dims, kernel_size)
        self.down = eqx.nn.Conv(dims, in_channels, in_channels, kernel_size=3, stride=2, padding=1, key=key0, padding_mode=padding_mode)
        self.norm1 = eqx.nn.GroupNorm(min(in_channels//4, min_group_norm_channel), in_channels)
        self.conv1 = eqx.nn.Conv(dims, in_channels, out_channels, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, key=key1)
        self.conv_block2 = ConvBlock(dims = dims, kernel_size=kernel_size, padding_mode = padding_mode, min_group_norm_channel = min_group_norm_channel, in_channels = out_channels, out_channels = out_channels, key = key2)
        self.conv = eqx.nn.Conv(dims, in_channels, out_channels, kernel_size = 1, key = key3)
        self.linear = eqx.nn.Linear(embed_dim, out_channels, key=key4)
    def __call__(self, t, x):
        h = jax.nn.silu(self.norm1(x))
        h = self.down(h)
        x = self.down(x)
        x = self.conv(x)
        h = self.conv1(h)
        t = self.linear(jax.nn.silu(t))
        h = h + jnp.expand_dims(t, axis=tuple(range(1, h.ndim)))
        h = self.conv_block2(h)
        h = h + x
        return h

class ResnetBlockUp(eqx.Module):
    norm1: eqx.nn.GroupNorm
    conv1: ConvBlock
    up: eqx.nn.ConvTranspose
    conv_block2: ConvBlock
    conv: eqx.nn.Conv
    linear: eqx.nn.Linear
    def __init__(self, *, dims = 2, kernel_size=3, padding_mode = 'REPLICATE', padding_mode_up = 'ZEROS', min_group_norm_channel = 32, in_channels, out_channels, embed_dim=256, key = jr.PRNGKey(0)):
        key0, key1, key2, key3, key4 = jr.split(key, 5)
        padding = smoothing_padding(dims, kernel_size)
        if padding_mode_up == 'CIRCULAR':
            padding_up = 'SAME'
        else:
            padding_up = 1
        self.up = eqx.nn.ConvTranspose(dims, in_channels, in_channels, kernel_size=4, stride=2, padding=padding_up, padding_mode = padding_mode_up, key=key0)
        self.norm1 = eqx.nn.GroupNorm(min(in_channels//4, min_group_norm_channel), in_channels)
        self.conv1 = eqx.nn.Conv(dims, in_channels, out_channels, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, key=key1)
        self.conv_block2 = ConvBlock(dims = dims, kernel_size=kernel_size, padding_mode = padding_mode, min_group_norm_channel = min_group_norm_channel, in_channels = out_channels, out_channels = out_channels, key = key2)
        self.conv = eqx.nn.Conv(dims, in_channels, out_channels, kernel_size = 1, key = key3)
        self.linear = eqx.nn.Linear(embed_dim, out_channels, key=key4)
    def __call__(self, t, x):
        h = jax.nn.silu(self.norm1(x))
        h = self.up(h)
        x = self.up(x)
        x = self.conv(x)
        h = self.conv1(h)
        t = self.linear(jax.nn.silu(t))
        h = h + jnp.expand_dims(t, axis=tuple(range(1, h.ndim)))
        h = self.conv_block2(h)
        h = h + x
        return h
    
class UNet(eqx.Module):
    gfp: GaussianFourierProjection
    mlp: eqx.nn.MLP
    conv: eqx.nn.Conv
    downblock: list[ResnetBlockDown]
    upblock: list[ResnetBlockUp]
    upresblock: list[ResnetBlock]
    downfeatureblock: list[ResnetBlock]
    upfeatureblock: list[ResnetBlock]
    downfeatureblock: list[ResnetBlock]
    upblockbefore: list[list[ResnetBlock]]
    upblockafter: list[list[ResnetBlock]]
    downblockbefore: list[list[ResnetBlock]]
    downblockafter: list[list[ResnetBlock]]
    midblock: list[ResnetBlock]
    upfeatureblock_reduce: list[ResnetBlock]
    finalblock: list[ResnetBlock]
    resblock: ResnetBlock
    block: ConvBlock
    def __init__(self, *, context_channels=0, beforeblock_length = 0, padding_mode = 'REPLICATE',  afterblock_length = 0, midblock_length = 2, final_block_length = 0, kernel_size = 3, data_shape, features=[32, 64, 128], downscaling_factor=2, key=jr.PRNGKey(0)):
        in_channels = data_shape[0]
        out_channels = in_channels - context_channels
        dims = len(data_shape) - 1
        padding_mode_up = 'ZEROS'
        if padding_mode == 'CIRCULAR': 
            padding_mode_up = 'CIRCULAR'
        emb_dim = features[0]
        min_group_norm = features[0]
        keys = jr.split(key, 12)
        if type(kernel_size) == int:
            kernel_size = [kernel_size for _ in features]
        assert len(kernel_size) == len(features), "Kernel size must be the same length as feature factors"
        if type(downscaling_factor) == int:
            downscaling_factors = [downscaling_factor]
            for i in range(len(features)-2):
                downscaling_factors.append(downscaling_factor)
        assert (len(downscaling_factors)+1) == len(features), "Downscaling factors must be one less than feature factors"
        data_shapes = [data_shape]
        for i in range(len(features)-2):
            data_shapes.append(tuple([features[i]] + [(x + downscaling_factors[i] - 1 ) // downscaling_factors[i] for x in data_shapes[i][1:]]))
        data_shapes = data_shapes[::-1]

        keys = jr.split(key, 3*len(features) + 5)
        self.gfp = GaussianFourierProjection(emb_dim)
        self.mlp = eqx.nn.MLP(emb_dim, emb_dim, 4 * emb_dim, 1, activation=jax.nn.silu, key=keys[0])
        padding = smoothing_padding(dims, kernel_size[0])
        self.conv = eqx.nn.Conv(dims, in_channels, features[0], kernel_size=kernel_size[0], padding=padding, padding_mode = padding_mode, key=keys[1])
        self.downblock = []
        self.upblock = []
        self.upresblock = []
        self.downfeatureblock = []
        self.upfeatureblock = []
        self.downblockbefore = []
        self.upblockbefore = []
        self.downblockafter = []
        self.upblockafter = []
        self.upfeatureblock_reduce = []
        for i in range(len(features)-1):
            subkeys = jr.split(keys[2 + i], 8 + 2 * (beforeblock_length + afterblock_length))
            self.downfeatureblock.append(ResnetBlock(dims=dims, kernel_size = kernel_size[i+1], padding_mode = padding_mode, in_channels=features[i], out_channels=features[i+1], embed_dim=emb_dim, key=subkeys[1]))
            self.upfeatureblock.append(ResnetBlock(dims=dims, kernel_size = kernel_size[i+1], padding_mode = padding_mode, in_channels= 2*features[i+1], out_channels=features[i+1], embed_dim=emb_dim, key=subkeys[2]))
            self.upfeatureblock_reduce.append(ResnetBlock(dims=dims, kernel_size = kernel_size[i+1], padding_mode = padding_mode, in_channels=features[i+1], out_channels=features[i], embed_dim=emb_dim, key=subkeys[3]))
            downresblock_before = []
            upresblock_before = []
            for j in range(beforeblock_length):
                key1, key2 = jr.split(subkeys[3 + j])
                downresblock_before.append(ResnetBlock(dims=dims, kernel_size = kernel_size[i+1],padding_mode = padding_mode, in_channels=features[i+1], out_channels=features[i+1], embed_dim=emb_dim, key=key1))
                upresblock_before.append(ResnetBlock(dims=dims, kernel_size = kernel_size[i+1], padding_mode = padding_mode,in_channels=2*features[i+1], out_channels=features[i+1], embed_dim=emb_dim, key=key2))
            self.downblockbefore.append(downresblock_before)
            self.upblockbefore.append(upresblock_before)
            downresblock_after = []
            upresblock_after = []
            for j in range(afterblock_length):
                key1, key2 = jr.split(subkeys[3 + 2*beforeblock_length + j])
                downresblock_after.append(ResnetBlock(dims=dims, kernel_size = kernel_size[i+1], padding_mode = padding_mode, in_channels=features[i+1], out_channels=features[i+1], embed_dim=emb_dim, key=key1))
                upresblock_after.append(ResnetBlock(dims=dims, kernel_size = kernel_size[i+1], padding_mode = padding_mode, in_channels=2*features[i+1], out_channels=features[i+1], embed_dim=emb_dim, key=key2))
            self.downblockafter.append(downresblock_after)
            self.upblockafter.append(upresblock_after)
            self.downblock.append(ResnetBlockDown(dims=dims, kernel_size = kernel_size[i+1], padding_mode = padding_mode, in_channels=features[i+1], out_channels=features[i+1], embed_dim=emb_dim, key=subkeys[3 + 2 * (beforeblock_length + afterblock_length) + 2]))
            self.upblock.append(ResnetBlockUp(dims=dims, kernel_size = kernel_size[i+1], padding_mode = padding_mode, padding_mode_up = padding_mode_up, in_channels=features[i+1], out_channels=features[i+1], embed_dim=emb_dim, key=subkeys[3 + 2 * (beforeblock_length + afterblock_length) + 3]))
            self.upresblock.append(ResnetBlock(dims=dims, kernel_size = kernel_size[i+1], padding_mode = padding_mode, in_channels= 2*features[i+1], out_channels=features[i+1], embed_dim=emb_dim, key=subkeys[3 + 2 * (beforeblock_length + afterblock_length) + 4]))
        self.midblock = []
        midblock_keys = jr.split(keys[2*len(features)+3], midblock_length)
        for i in range(midblock_length):
            self.midblock.append(ResnetBlock(dims=dims, kernel_size = kernel_size[0], padding_mode = padding_mode, in_channels=features[-1], out_channels=features[-1], embed_dim=emb_dim, key=midblock_keys[i]))
        self.finalblock = []
        finalblock_keys = jr.split(keys[2*len(features)+4], final_block_length)
        for i in range(final_block_length):
            self.finalblock.append(ResnetBlock(dims=dims, kernel_size = kernel_size[0], padding_mode = padding_mode, in_channels=features[0], out_channels=features[0], embed_dim=emb_dim, key=finalblock_keys[i]))
        self.resblock = ResnetBlock(dims=dims, kernel_size = kernel_size[0], padding_mode = padding_mode, in_channels=2*features[0], out_channels=features[0], embed_dim=emb_dim, key=keys[2*len(features)+1])
        self.block = ConvBlock(dims = dims, kernel_size = kernel_size[0], padding_mode = padding_mode, in_channels = features[0], out_channels = out_channels, key=keys[2*len(features)+2])
    @eqx.filter_jit
    def __call__(self, t, y):
        # Lift 
        t = self.gfp(t)  # 1 -> features[0]
        t = self.mlp(t)  # features[0] -> features[0]
        h = self.conv(y) # in_channels -> features[0]
        #Downsample
        hs = [h]
        for i in range(len(self.downblock)):
            h = self.downfeatureblock[i](t, h) # features[i] -> features[i+1]
            hs.append(h)
            for resblock in self.downblockbefore[i]:
                h = resblock(t, h)             # features[i+1] -> features[i+1]
                hs.append(h)
            h = self.downblock[i](t, h)        # features[i+1] -> features[i+1]
            hs.append(h)
            for resblock in self.downblockafter[i]:
                h = resblock(t, h)             # features[i+1] -> features[i+1]
                hs.append(h)
        # Middle Block
        for i in range(len(self.midblock)):
            h = self.midblock[i](t, h)         # features[-1] -> features[-1]
        # Upsample
        for i in range(len(self.upblock)):
            for resblock in self.upblockafter[-(i+1)]:
                h = jnp.concatenate((h, hs.pop()), axis=0) 
                h = resblock(t, h)                       # 2*features[-(i+1)] -> features[-(i+1)]
            h = jnp.concatenate((h, hs.pop()), axis=0)
            h = self.upresblock[-(i+1)](t, h)            # 2*features[-(i+1)] -> features[-(i+1)]
            h = self.upblock[-(i+1)](t, h)               # features[-(i+1)] -> features[-(i+1)]
            for resblock in self.upblockbefore[-(i+1)]:
                h = jnp.concatenate((h, hs.pop()), axis=0) 
                h = resblock(t, h)                       # 2*features[-(i+1)] -> features[-(i+1)]
            h = jnp.concatenate((h, hs.pop()), axis=0)
            h = self.upfeatureblock[-(i+1)](t, h)        # 2*features[-(i+1)] -> features[-(i+1)]
            h = self.upfeatureblock_reduce[-(i+1)](t, h) # features[-(i+1)] -> features[-i]
        # Projection
        h = jnp.concatenate((h, hs.pop()), axis=0)
        h = self.resblock(t, h) # 2 * features[0] -> features[0]
        for resblock in self.finalblock:
            h = resblock(t, h)  # features[0] -> features[0]
        # Final Layer 
        h = self.block(h)       # features[0] -> out_channels
        return h