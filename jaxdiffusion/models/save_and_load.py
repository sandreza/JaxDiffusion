import json
import equinox as eqx
import jax.random as jr

def save(filename, hyperparams, model):
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)

def load(filename, model_name):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model = model_name(key=jr.PRNGKey(0), **hyperparams)
        return eqx.tree_deserialise_leaves(f, model)


