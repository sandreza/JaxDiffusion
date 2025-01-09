from jaxdiffusion import *
from jaxdiffusion.process.sampler import Sampler

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
data_std = jnp.std(images)
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
    "data_shape": (1, 28, 28),      # Single-channel grayscale MNIST images
    "is_biggan": True,              # Whether to use BigGAN architecture
    "dim_mults": [1, 2, 4],         # Multiplicative factors for UNet feature map dimensions
    "hidden_size": 32,              # Base hidden channel size
    "heads": 28,                     # Number of attention heads
    "dim_head": 28,                 # Size of each attention head
    "num_res_blocks": 4,            # Number of residual blocks per stage
    "attn_resolutions": [28, 14]    # Resolutions for applying attention
}

key = jr.PRNGKey(seed)

model_filename = "mnist_diffusion_unet.mo"
if os.path.exists(model_filename):
    print("Loading file " + model_filename)
    model = load(model_filename, UNet)
    model_hyperparameters = load_hyperparameters(model_filename)
    print("Done Loading model with hyperparameters")
    print(model_hyperparameters)
else:
    print("File test_save.mo does not exist. Creating UNet")
    model = UNet(key = key, **unet_hyperparameters)
    print("Done Creating UNet")

# Optimisation hyperparameters
num_steps= 100000
lr=3e-4
batch_size = 32*4
print_every=100
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
        model, schedule, data, train_key, opt_state, opt.update
    )
    total_value += value.item()
    total_size += 1
    if (step % print_every) == 0 or step == num_steps - 1:
        print(f"Step={step} Loss={total_value / total_size}")
        total_value = 0
        total_size = 0
        save(model_filename, unet_hyperparameters, model)

save(model_filename, unet_hyperparameters, model)
# Sampling
data_shape = data[0, :, :, :].shape
sampler = Sampler(schedule, model, data_shape)
sqrt_N = 10
samples = sampler.sample(sqrt_N**2)

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
filename = "mnist_diffusion_unet.png"
plt.savefig(filename)