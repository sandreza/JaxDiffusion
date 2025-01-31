from jaxdiffusion import *
from jaxdiffusion.process.sampler import Sampler
from jaxdiffusion.process.sampler import ODESampler
from jaxdiffusion.models.unet import UNet

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

images = mnist() 
data_mean = jnp.mean(images)
data_std = jnp.std(images) * 2
data_max = jnp.max(images)
data_min = jnp.min(images)
data_shape = images.shape[1:]
data = (images - data_mean) / data_std 

# add util for calculating the distance
key = jr.PRNGKey(0)
key, subkey = jax.random.split(key)
random_index_1 = jax.random.randint(key, 10, 0, data.shape[0]-1)
key, subkey = jax.random.split(key)
random_index_2 = jax.random.randint(key, 10, 0, data.shape[0]-1)
tmp = jnp.linalg.norm(data[random_index_1, 0, :, :] - data[random_index_2, 0, :, :], axis=(1, 2))
sigma_max = max(tmp)
sigma_min = 1e-2
schedule = VarianceExplodingBrownianMotion(sigma_min, sigma_max) 

seed = 12345
key, subkey = jax.random.split(key)

unet_hyperparameters = {
    "context_channels": 0,
    "data_shape": (1, 28, 28),
    "features": [32, 64],
    "downscaling_factor": 2,
    "kernel_size": 3, 
    "beforeblock_length": 1,
    "afterblock_length": 1,
    "midblock_length": 2,
    "final_block_length": 0,
}
#beforeblock_length = 0, afterblock_length = 0, midblock_length = 2,
key = jr.PRNGKey(seed)
model_filename = "mnist.mo"
model_name = UNet
if os.path.exists(model_filename):
    print("Loading file " + model_filename)
    model = load(model_filename, model_name)
    model_hyperparameters = load_hyperparameters(model_filename)
    print("Done Loading model with hyperparameters")
    print(model_hyperparameters)
else:
    print("File does not exist. Creating UNet")
    model = model_name(key = key, **unet_hyperparameters)
    print("Done Creating UNet")

# Training
lr=1e-3
# Seed
seed=5678
opt = optax.chain(
    optax.adam(lr),
)
# Optax will update the floating-point JAX arrays in the model.
opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

model_key, train_key, loader_key, sample_key = jr.split(key, 4) 
# Train Test Split
test_size = 0.2
dataset_size = data.shape[0]
indices = jnp.arange(dataset_size)
perm = jax.random.permutation(subkey, indices)
test_size = int(dataset_size * test_size)
train_size = dataset_size - test_size
train_indices = perm[:train_size]
test_indices = perm[train_size:]
train_data = data[train_indices, :, :, :]
test_data = data[test_indices, :, :, :]

_, subkey, subkey2 = jax.random.split(subkey, 3)
train_size = train_data.shape[0]
test_size = test_data.shape[0]
batch_size = 32 * 4
train_skip_size = train_size // batch_size
test_skip_size = test_size // batch_size

# Training
model_key, train_key, test_key, loader_key, sample_key = jr.split(key, 5) 
total_value = 0
total_size = 0
test_value = 0
total_test_size = 0
losses = []
test_losses = []
epochs = 100

for epoch in range(epochs):
    _, subkey, subkey2, subkey3 = jax.random.split(subkey, 4)
    perm_train = jax.random.permutation(subkey, train_size)
    perm_test  = jax.random.permutation(subkey2, test_size)
    for chunk in range(train_skip_size-1):
        _, train_key = jax.random.split(train_key)
        tr_data = train_data[perm_train[chunk*batch_size:(chunk+1)*batch_size], :, :, :]
        value, model, train_key, opt_state = make_step(
            model, schedule, tr_data, train_key, opt_state, opt.update
        )
        total_value += value.item()
        total_size += 1
    for chunk in range(test_skip_size-1):
        _, test_key = jax.random.split(test_key)
        tst_data = test_data[perm_test[chunk*batch_size:(chunk+1)*batch_size], :, :, :]
        test_value += batch_loss_function(model, schedule, tst_data, test_key)
        total_test_size += 1
    print(f"------Epoch={epoch}------")
    print(f"Loss={total_value / total_size}")
    print(f"Test Loss={test_value / total_test_size}")
    total_value = 0
    total_size = 0
    test_value = 0
    total_test_size = 0
    save(model_filename, unet_hyperparameters, model)

save(model_filename, unet_hyperparameters, model)
# Sampling
_, _, subkey = jax.random.split(subkey, 3)
data_shape = data[0, :, :, :].shape
sampler = Sampler(schedule, model, data_shape)
sqrt_N = 10
samples = sampler.sample(sqrt_N**2, key = subkey, steps = 300)

# plotting
sample = jnp.reshape(samples, (sqrt_N, sqrt_N, 28, 28))
sample = jnp.transpose(sample, (0, 2, 1, 3))
sample = jnp.reshape(sample, (sqrt_N * 28, sqrt_N * 28))
sample = data_mean + data_std * sample
sample = jnp.clip(sample, data_min, data_max)

plt.figure()
plt.imshow(sample, cmap="Greys")
plt.axis("off")
plt.tight_layout()
plt.show()
filename = model_filename[0:-3] + "_sde.png"
plt.savefig(filename)

_, _, subkey = jax.random.split(subkey, 3)
data_shape = data[0, :, :, :].shape
sampler = ODESampler(schedule, model, data_shape)
sqrt_N = 10
samples = sampler.sample(sqrt_N**2, key = subkey, steps = 300)

# plotting
sample = jnp.reshape(samples, (sqrt_N, sqrt_N, 28, 28))
sample = jnp.transpose(sample, (0, 2, 1, 3))
sample = jnp.reshape(sample, (sqrt_N * 28, sqrt_N * 28))
sample = data_mean + data_std * sample
sample = jnp.clip(sample, data_min, data_max)

plt.figure()
plt.imshow(sample, cmap="Greys")
plt.axis("off")
plt.tight_layout()
plt.show()
filename = model_filename[0:-3] + "_ode.png"
plt.savefig(filename)