import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import functools as ft

@eqx.filter_jit
def velocity_single_loss_function(model, std, data, t, key):
    noise = jr.normal(key, data.shape)
    y = data + std * noise
    scaling = 1 + std
    tau = jnp.log(std) / 4
    nn = model(tau, y / scaling) 
    # score(y) = -y / std**2 * alpha + beta * nn(y / scaling) / (std**2)
    # score(data + std * noise) = - data / std**2 * alpha - noise / std *alpha + ... 
    # loss = -data/std**2 * alpha  + (1-alpha) * Z / std + beta * nn / std**2 
    # weight = std**2 / beta
    # weight * loss = -data *alpha/beta + (1-alpha)/beta * std* Z + nn 
    # alpha -> 0 as std -> 0, alpha -> 1 as std -> inf
    # beta -> O(std) as std -> 0, beta -> O(1) as std -> inf
    # alpha = std / (1 + std), beta = std / (1 + std)
    # 1 - alpha = 1 / (1 + std)
    # w1 = alpha / beta 
    # w2 = (1 - alpha) / beta * std
    # weight * loss = -data * w1 + Z * w2 + nn
    # weight * loss = -data + Z + nn
    # weight(t) = std**2 / (1 + std)
    w1 = 1 
    w2 = 1
    return jnp.mean((nn - (data * w1)  + noise * w2) ** 2) # accounts for cancellations 

@eqx.filter_jit
def velocity_batch_loss_function(model, schedule, data, key):
    batch_size = data.shape[0]
    tkey, losskey = jr.split(key)
    losskey = jr.split(losskey, batch_size)

    t0 = schedule.tmin
    t1 = schedule.tmax
    t = jr.uniform(tkey, (batch_size,), minval=0, maxval=1)
    t = t0 + (t1 - t0) * t
    sigmas = jax.vmap(schedule.sigma)(t)

    loss_function = ft.partial(velocity_single_loss_function, model)
    loss_function = jax.vmap(loss_function)
    return jnp.mean(loss_function(sigmas, data, t, losskey))

@eqx.filter_jit
def velocity_make_step(model, schedule, data, key, opt_state, opt_update):
    loss_function = eqx.filter_value_and_grad(velocity_batch_loss_function)
    loss, grads = loss_function(model, schedule, data, key)
    updates, opt_state = opt_update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    key = jr.split(key, 1)[0]
    return loss, model, key, opt_state

@eqx.filter_jit
def velocity_conditional_single_loss_function(model, context_size, std, data, t, key):
    data_shape = list(data.shape)
    data_shape[0] -= context_size
    data_shape = tuple(data_shape)
    
    noise = jr.normal(key, data_shape)
    y = jnp.copy(data)
    y = y.at[:-context_size, ...].add(std * noise)
    scaling = 1 + std
    y = y.at[:-context_size, ...].divide(scaling)
    # score(y) = -y / std**2 * alpha + beta * nn(y / scaling) / (std**2)
    # score(data + std * noise) = - data / std**2 * alpha - noise / std *alpha + ... 
    # loss = -data/std**2 * alpha  + (1-alpha) * Z / std + beta * nn / std**2 
    # weight = std**2 / beta
    # weight * loss = -data *alpha/beta + (1-alpha)/beta * std* Z + nn 
    # alpha -> 0 as std -> 0, alpha -> 1 as std -> inf
    # beta -> O(std) as std -> 0, beta -> O(1) as std -> inf
    # alpha = std / (1 + std), beta = std / (1 + std)
    # 1 - alpha = 1 / (1 + std)
    # w1 = alpha / beta 
    # w2 = (1 - alpha) / beta * std
    # weight * loss = -data * w1 + Z * w2 + nn
    # weight * loss = -data  + Z + nn
    # weight(t) = std**2 / (1 + std)
    w1 = 1 
    w2 = 1
    tau = jnp.log(std) / 4
    nn = model(tau, y) 
    return jnp.mean((nn -  (data[:-context_size, ...] * w1) + noise * w2) ** 2) # accounts for cancellations

@eqx.filter_jit
def velocity_conditional_batch_loss_function(model, context_size, schedule, data, key):
    batch_size = data.shape[0]
    tkey, losskey = jr.split(key)
    losskey = jr.split(losskey, batch_size)

    t0 = schedule.tmin
    t1 = schedule.tmax
    t = jr.uniform(tkey, (batch_size,), minval=0, maxval=1)
    t = t0 + (t1 - t0) * t
    sigmas = jax.vmap(schedule.sigma)(t)

    loss_function = ft.partial(velocity_conditional_single_loss_function, model, context_size)
    loss_function = jax.vmap(loss_function)
    return jnp.mean(loss_function(sigmas, data, t, losskey))

@eqx.filter_jit
def velocity_conditional_make_step(model, context_size, schedule, data, key, opt_state, opt_update):
    loss_function = eqx.filter_value_and_grad(velocity_conditional_batch_loss_function)
    loss, grads = loss_function(model, context_size, schedule, data, key)
    updates, opt_state = opt_update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    key = jr.split(key, 1)[0]
    return loss, model, key, opt_state