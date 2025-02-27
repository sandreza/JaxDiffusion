import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import functools as ft

@eqx.filter_jit
def residual_single_loss_function(model, std, data, t, key):
    noise = jr.normal(key, data.shape)
    y = data + std * noise
    nn = model(t, y) 
    # usual loss = |sigma(t) * score(t, data + sigma(t) Z) + Z|^2
    # usual score(t, y) = nn(t, y) / sigma(t)
    # this score(t, y) = - y / sigma^2 + nn(t, y) / sigma^2
    # this loss = |sigma(t) * score(t, data + sigma(t) Z) + Z|^2 * sigma^{3/2}(t)
    # which transforms to loss = |nn(t, data + sigma(t) Z) - data|^2 / sqrt(sigma(t))
    # score = (nn - y) / std**2
    return jnp.mean((nn - data) ** 2) # accounts for cancellations 

@eqx.filter_jit
def residual_batch_loss_function(model, schedule, data, key):
    batch_size = data.shape[0]
    tkey, losskey = jr.split(key)
    losskey = jr.split(losskey, batch_size)

    t0 = schedule.tmin
    t1 = jnp.array([1.0])
    t = jr.uniform(tkey, (batch_size,), minval=t0, maxval=t1)
    sigmas = jax.vmap(schedule.sigma)(t)

    loss_function = ft.partial(residual_single_loss_function, model)
    loss_function = jax.vmap(loss_function)
    return jnp.mean(loss_function(sigmas, data, t, losskey))

@eqx.filter_jit
def residual_make_step(model, schedule, data, key, opt_state, opt_update):
    loss_function = eqx.filter_value_and_grad(residual_batch_loss_function)
    loss, grads = loss_function(model, schedule, data, key)
    updates, opt_state = opt_update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    key = jr.split(key, 1)[0]
    return loss, model, key, opt_state

@eqx.filter_jit
def residual_conditional_single_loss_function(model, context_size, std, data, t, key):
    data_shape = list(data.shape)
    data_shape[0] -= context_size
    data_shape = tuple(data_shape)
    
    noise = jr.normal(key, data_shape)
    y = jnp.copy(data)
    y = y.at[:-context_size, ...].add(std * noise)
    # score = (nn - y) / std**2
    return jnp.mean((model(t, y) -  data[:-context_size, ...]) ** 2) # accounts for cancellations

@eqx.filter_jit
def residual_conditional_batch_loss_function(model, context_size, schedule, data, key):
    batch_size = data.shape[0]
    tkey, losskey = jr.split(key)
    losskey = jr.split(losskey, batch_size)

    t0 = schedule.tmin
    t1 = jnp.array([1.0])
    t = jr.uniform(tkey, (batch_size,), minval=t0, maxval=t1)
    sigmas = jax.vmap(schedule.sigma)(t)

    loss_function = ft.partial(residual_conditional_single_loss_function, model, context_size)
    loss_function = jax.vmap(loss_function)
    return jnp.mean(loss_function(sigmas, data, t, losskey))

@eqx.filter_jit
def residual_conditional_make_step(model, context_size, schedule, data, key, opt_state, opt_update):
    loss_function = eqx.filter_value_and_grad(residual_conditional_batch_loss_function)
    loss, grads = loss_function(model, context_size, schedule, data, key)
    updates, opt_state = opt_update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    key = jr.split(key, 1)[0]
    return loss, model, key, opt_state