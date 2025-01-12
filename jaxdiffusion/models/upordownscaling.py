import equinox as eqx
import jax.numpy as jnp

class ConvDownSampling(eqx.Module):
    downsample: eqx.nn.Conv

    def __init__(self, *, dims, in_channels, out_channels, factor = 2, padding_mode = 'REPLICATE', box_filter = True, key):
        model = eqx.nn.Conv(dims, in_channels, out_channels, factor, padding = self.downsampling_padding(dims, factor), key=key, padding_mode = padding_mode, stride = factor, use_bias = False, groups = in_channels)
        if box_filter:
            model = eqx.tree_at(lambda m: m.weight, model, replace_fn=lambda _: jnp.full_like(model.weight, 1/(factor**dims)))
        self.downsample = model
    def downsampling_padding(self, dims, factor):
        return tuple(([0, factor - 1], ) * dims)
    @eqx.filter_jit
    def __call__(self, x):
        return self.downsample(x)
    
# incorrect but ketting there
class ConvUpSampling(eqx.Module):
    upsample: eqx.nn.ConvTranspose
    smooth: eqx.nn.Conv

    def __init__(self, *, dims, in_channels, out_channels, kernel_size = 2, data_shape = None, padding_mode = 'ZEROS', weighted_neighbor = True, padding_mode_smoother = 'REPLICATE', key):
        if data_shape == None:
            data_shape = tuple((0 for x in range(dims + 1)))     
        assert len(data_shape) == dims+1, f"data_shape must be of size {dims + 1}"

        model_up = eqx.nn.ConvTranspose(dims, in_channels, out_channels, kernel_size, padding = self.upsampling_padding(data_shape, kernel_size), key=key, padding_mode = padding_mode, stride = kernel_size, use_bias = False, groups = in_channels)
        model_smooth = eqx.nn.Conv(dims, out_channels, out_channels, kernel_size, padding = self.smoothing_padding(dims, kernel_size), key=key, padding_mode = padding_mode_smoother, stride = 1, use_bias = False, groups = out_channels)
        if weighted_neighbor:
            model_up = eqx.tree_at(lambda m: m.weight, model_up, replace_fn=lambda _: jnp.full_like(model_up.weight, 1.0))
            model_smooth = eqx.tree_at(lambda m: m.weight, model_smooth, replace_fn=lambda _: jnp.full_like(model_smooth.weight,  1/(kernel_size**dims)))
        self.upsample = model_up
        self.smooth = model_smooth
    def upsampling_padding(self, data_shape, kernel_size):
        # (y // kernel_size + y%kernel_size) is the assumed size of the previously downscaled image
        return tuple([0, kernel_size * (y // kernel_size + y%kernel_size) - y] for y in data_shape[1:])  
    def smoothing_padding(self, dims, kernel_size):
        return tuple(([0, kernel_size - 1], ) * dims)
    @eqx.filter_jit
    def __call__(self, x):
        return self.smooth(self.upsample(x))