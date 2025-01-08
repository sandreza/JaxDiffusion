from jaxdiffusion import *
from jaxdiffusion.process.sampler import Sampler

def mnist():
    image_filename = "train-images-idx3-ubyte.gz"
    label_filename = "train-labels-idx1-ubyte.gz"
    url_dir = "https://storage.googleapis.com/cvdf-datasets/mnist"
    target_dir = "/orcd/home/001/sandre/Repositories/JaxUvTest/data/mnist"

    # Download image data
    image_url = f"{url_dir}/{image_filename}"
    image_target = f"{target_dir}/{image_filename}"
    if not os.path.exists(image_target):
        os.makedirs(target_dir, exist_ok=True)
        urllib.request.urlretrieve(image_url, image_target)
        print(f"Downloaded {image_url} to {image_target}")

    # Download label data
    label_url = f"{url_dir}/{label_filename}"
    label_target = f"{target_dir}/{label_filename}"
    if not os.path.exists(label_target):
        urllib.request.urlretrieve(label_url, label_target)
        print(f"Downloaded {label_url} to {label_target}")

    # Process image data
    with gzip.open(image_target, "rb") as fh:
        _, batch, rows, cols = struct.unpack(">IIII", fh.read(16))
        image_shape = (batch, 1, rows, cols)
        images = jnp.array(array.array("B", fh.read()), dtype=jnp.uint8).reshape(image_shape)

    # Process label data
    with gzip.open(label_target, "rb") as fh:
        _, num_labels = struct.unpack(">II", fh.read(8))
        labels = jnp.array(array.array("B", fh.read()), dtype=jnp.uint8)

    return images, labels

# Assuming images and labels are already loaded
images, labels = mnist()

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
sigma_min = 1e-3
key, subkey = jax.random.split(key)
schedule = VarianceExplodingBrownianMotion(sigma_min, sigma_max) 

# Calculate the average value for each label
label_averages = jnp.array([jnp.mean(data[labels == label, :, :, :], axis=0)[0, :, :] for label in range(10)], dtype = data.dtype)
# Create the new array with the desired shape
conditional_data = jnp.zeros((data.shape[0], 2, data.shape[2], data.shape[3]), dtype=data.dtype)
# Populate the new array
conditional_data = conditional_data.at[:, 0, :, :].set(data[:, 0, :, :])
for label in range(10):
    conditional_data = conditional_data.at[labels == label, 1, :, :].set(label_averages[label])

seed = 12345
data_shape = conditional_data.shape[1:]
context_size = 1
unet_hyperparameters = {
    "data_shape": data_shape,       # grayscale MNIST images with a "context" channel
    "is_biggan": True,              # Whether to use BigGAN architecture
    "dim_mults": [1, 2, 4],         # Multiplicative factors for UNet feature map dimensions
    "hidden_size": 32,              # Base hidden channel size
    "heads": 28,                     # Number of attention heads
    "dim_head": 28,                  # Size of each attention head
    "num_res_blocks": 4,            # Number of residual blocks per stage
    "attn_resolutions": [28, 14, 7],   # Resolutions for applying attention
    "context_size": context_size,   # context dimension
}

key = jr.PRNGKey(seed)
key, unet_key, subkey = jax.random.split(key, 3)

if os.path.exists("conditional_test_save.mo"):
    print("Loading file conditional_test_save.mo")
    model = load("conditional_test_save.mo", UNet)
    print("Done Loading model")
else:
    print("File conditional_test_save.mo does not exist. Creating UNet")
    model = UNet(key = unet_key, **unet_hyperparameters)
    print("Done Creating UNet")

# Train Test Split
test_size = 0.2
dataset_size = data.shape[0]
indices = jnp.arange(dataset_size)
perm = jax.random.permutation(subkey, indices)
test_size = int(dataset_size * test_size)
train_size = dataset_size - test_size
train_indices = perm[:train_size]
test_indices = perm[train_size:]
train_data = conditional_data[train_indices, :, :, :]
test_data = conditional_data[test_indices, :, :, :]

# Create DataLoaders
subkey = jax.random.split(subkey)[1]
batchsize = 32 * 4
train_dataloader = dataloader(train_data, batchsize, subkey)
test_dataloader = dataloader(test_data, batchsize, subkey)

# Optimisation hyperparameters
num_steps= 100000
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

model_key, train_key, test_key, loader_key, sample_key = jr.split(key, 5) 

total_value = 0
total_size = 0
train_value = 0
for step, data, test_data in zip(range(num_steps), dataloader(train_data, batch_size, key=loader_key), dataloader(test_data, batch_size, key=loader_key)):
    value, model, train_key, opt_state = conditional_make_step(
        model, context_size, schedule, data, train_key, opt_state, opt.update
    )
    total_value += value.item()
    train_value += conditional_batch_loss_fn(model, context_size, schedule, test_data, test_key)
    total_size += 1
    if (step % print_every) == 0 or step == num_steps - 1:
        print(f"Step={step} Loss={total_value / total_size}")
        print(f"Test Loss={train_value / total_size}")
        total_value = 0
        train_value = 0
        total_size = 0
        save("conditional_test_save.mo", unet_hyperparameters, model)

def precursor_context_model(model, context, t, y):
    y = jnp.concatenate((y, context), axis=0)
    return model(t, y)

first_indices = {}
for i in range(10):
    first_indices[i] = jnp.where(labels == i)[0][0]

for ii in range(10):
    context_ind = first_indices[ii]
    context = conditional_data[context_ind, 1:, :, :]
    tmp = jnp.zeros((1, 28, 28))
    context_model = ft.partial(precursor_context_model, model, context)

    # Sampling
    print("Sampling " + str(labels[context_ind]))
    data_shape = conditional_data[0, 0:1, :, :].shape
    sampler = Sampler(schedule, context_model, data_shape)
    sqrt_N = 10
    samples = sampler.sample(sqrt_N**2, steps = 30)
    print("Done Sampling, Now Plotting")
    # plotting
    sample = jnp.reshape(samples, (sqrt_N, sqrt_N, 28, 28))
    sample = data_mean + data_std * sample
    sample = jnp.clip(sample, data_min, data_max)

    conditional_information = context * data_std + data_mean 
    conditional_information = jnp.clip(conditional_information, data_min, data_max)

    fig, axes = plt.subplots(sqrt_N, sqrt_N, figsize=(sqrt_N, sqrt_N))
    # Plot the original images (0 index of axis 1)
    for i in range(sqrt_N):
        for j in range(sqrt_N):
            if i == j == 0: 
                axes[j, i].imshow(context[0, :, :], cmap="Greys")
                axes[j, i].set_title(f"Context")
                axes[j, i].axis("off")
            else:
                axes[j, i].imshow(sample[i, j, :, :], cmap="Greys")
                axes[j, i].set_title(f"{i}, {j}")
                axes[j, i].axis("off")
    plt.tight_layout()
    plt.show()
    filename = "mnist_diffusion_unet_quax_with_context_" + str(labels[context_ind]) + ".png"
    plt.savefig(filename)