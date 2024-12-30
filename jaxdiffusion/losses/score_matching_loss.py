import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import functools as ft

@eqx.filter_jit
def single_loss_fn(model, std, data, t, key):
    noise = jr.normal(key, data.shape)
    y = data + std * noise
    pred = model(t, y)
    return jnp.mean((pred * std + noise  ) ** 2)

@eqx.filter_jit
def batch_loss_fn(model, fwd_process, data, key):
    batch_size = data.shape[0]
    tkey, losskey = jr.split(key)
    losskey = jr.split(losskey, batch_size)

    t0 = fwd_process.tmin
    t1 = jnp.array([1.0])
    t = jr.uniform(tkey, (batch_size,), minval=t0, maxval=t1 )
    sigma_p = jax.vmap(fwd_process.kernel_cholesky)(t, data)

    loss_fn = ft.partial(single_loss_fn, model)
    loss_fn = jax.vmap(loss_fn)
    return jnp.mean(loss_fn(sigma_p, data, t, losskey))

@eqx.filter_jit
def make_step(model, fwd_process, data, key, opt_state, opt_update):
    loss_fn = eqx.filter_value_and_grad(batch_loss_fn)
    loss, grads = loss_fn(model, fwd_process, data, key)
    updates, opt_state = opt_update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    key = jr.split(key, 1)[0]
    return loss, model, key, opt_state