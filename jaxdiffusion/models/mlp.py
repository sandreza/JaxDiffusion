from typing import Optional

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray
from jax import debug

class TimeEmbedding(eqx.Module):
    embed_dim: Int

    def __init__(self, embed_dim: Int):
        if embed_dim % 2 != 0:
            raise ValueError("Embedding dimension must be even!")
        self.embed_dim = embed_dim

    @eqx.filter_jit
    def __call__(self, t: Float):
        half_dim = self.embed_dim // 2
        embeddings = jnp.log(10000) / (half_dim - 1)
        embeddings = jnp.exp(jnp.arange(half_dim) * -embeddings)
        embeddings = t * embeddings
        embeddings = jnp.concatenate(
            [jnp.sin(embeddings), jnp.cos(embeddings)], axis=-1
        )

        return embeddings


class DenseScoreNetwork(eqx.Module):
    dense1: eqx.nn.Linear
    dense2: eqx.nn.Linear
    dense3: eqx.nn.Linear
    embedding: TimeEmbedding
    embed_dense1: eqx.nn.Linear
    embed_dense2: eqx.nn.Linear

    def __init__(
        self,
        n_dim: Int,
        hidden_dim: Int,
        key: PRNGKeyArray,
        output_dim: Optional[Int] = None,
    ):
        if hidden_dim % 2 != 0:
            raise ValueError("Embedding dimension must be even!")
        super().__init__()
        output_dim = n_dim if not output_dim else output_dim

        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.dense1 = eqx.nn.Linear(n_dim, hidden_dim, key=k1)
        self.dense2 = eqx.nn.Linear(hidden_dim, hidden_dim, key=k2)
        self.dense3 = eqx.nn.Linear(hidden_dim, output_dim, key=k3)

        self.embedding = TimeEmbedding(hidden_dim)
        self.embed_dense1 = eqx.nn.Linear(hidden_dim, hidden_dim, key=k4)
        self.embed_dense2 = eqx.nn.Linear(hidden_dim, hidden_dim, key=k5)

    @eqx.filter_jit
    def __call__(
        self, t: Float, x: Float[Array, "n_dim"], args=None
    ) -> Float[Array, "output_dim"]:
        t_embedded = self.embedding(t)

        x = jnn.swish(self.dense1(x)) + jnn.swish(self.embed_dense1(t_embedded))
        x = jnn.swish(self.dense2(x)) + jnn.swish(self.embed_dense2(t_embedded))
        x = self.dense3(x)

        return x
