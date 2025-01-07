from jaxdiffusion import *
import numpy as np
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
sigma_min = 1e-2
key, subkey = jax.random.split(key)
fwd_process = VarExpBrownianMotion(sigma_min, sigma_max) 

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
unet_hyperparameters = {
    "data_shape": data_shape,       # Single-channel grayscale MNIST images
    "is_biggan": True,              # Whether to use BigGAN architecture
    "dim_mults": [1, 2, 4],         # Multiplicative factors for UNet feature map dimensions
    "hidden_size": 16,              # Base hidden channel size
    "heads": 8,                     # Number of attention heads
    "dim_head": 7,                  # Size of each attention head
    "num_res_blocks": 4,            # Number of residual blocks per stage
    "attn_resolutions": [28, 14],   # Resolutions for applying attention
    "context_size": 1               # context dimension
}

key = jr.PRNGKey(seed)
key, unet_key = jax.random.split(key)

if os.path.exists("conditional_test_save.mo"):
    print("Loading file conditional_test_save.mo")
    model = load("conditional_test_save.mo", UNet)
    print("Done Loading model")
else:
    print("File conditional_test_save.mo does not exist. Creating UNet")
    model = UNet(key = unet_key, **unet_hyperparameters)
    print("Done Creating UNet")

model(1.0, conditional_data[0, :, :, :])