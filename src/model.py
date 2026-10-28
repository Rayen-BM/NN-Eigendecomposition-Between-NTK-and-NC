import jax
import jax.numpy as jnp
from flax import nnx

class NTKLinear(nnx.Linear):
    """Linear layer with NTK parameterization scaling on the forward pass."""
    def __call__(self, x: jax.Array) -> jax.Array:
        # Scale weights by 1 / sqrt(fan_in) during the forward pass
        scale = 1.0 / jnp.sqrt(self.in_features)
        return (x @ (self.kernel.value * scale)) + self.bias.value


class NTKModel(nnx.Module):
    """Network using Neural Tangent Kernel (NTK) parameterization."""
    def __init__(self, in_features: int, width: int, out_features: int, rngs: nnx.Rngs, num_hidden_layers: int = 1):
        initializer = nnx.initializers.normal(stddev=1.0)
        
        layers = []
        layers.append(
            NTKLinear(in_features, width, kernel_init=initializer, bias_init=initializer, rngs=rngs)
        )
        
        for _ in range(num_hidden_layers - 1):
            layers.append(
                NTKLinear(width, width, kernel_init=initializer, bias_init=initializer, rngs=rngs)
            )
            
        self.hidden_layers = nnx.List(layers)
        self.out_layer = NTKLinear(width, out_features, kernel_init=initializer, bias_init=initializer, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.hidden_layers:
            x = nnx.relu(layer(x))
        return self.out_layer(x)


class StandardNetwork(nnx.Module):
    """Standard neural network with He/Glorot-style initialization."""
    def __init__(self, in_features: int, width: int, out_features: int, rngs: nnx.Rngs, num_hidden_layers: int = 1):
        init_in = nnx.initializers.normal(stddev=1.0 / jnp.sqrt(in_features))
        init_hid = nnx.initializers.normal(stddev=1.0 / jnp.sqrt(width))
        
        layers = []
        layers.append(
            nnx.Linear(in_features, width, kernel_init=init_in, bias_init=init_in, rngs=rngs)
        )
        
        for _ in range(num_hidden_layers - 1):
            layers.append(
                nnx.Linear(width, width, kernel_init=init_hid, bias_init=init_hid, rngs=rngs)
            )
            
        self.hidden_layers = nnx.List(layers)
        self.out_layer = nnx.Linear(width, out_features, kernel_init=init_hid, bias_init=init_hid, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.hidden_layers:
            x = nnx.relu(layer(x))
        return self.out_layer(x)


class MuPFirstLayer(nnx.Linear):
    """First layer for Maximal Update Parameterization (µP)."""
    def __call__(self, x: jax.Array) -> jax.Array:
        # µP scales the first layer output by sqrt(fan_out)
        scale = jnp.sqrt(self.out_features)
        out = (x @ (self.kernel.value * scale))
        return out + self.bias.value


class MaximalUpdateNetwork(nnx.Module):
    """Network using Maximal Update Parameterization (µP)."""
    def __init__(self, in_features: int, width: int, out_features: int, rngs: nnx.Rngs, num_hidden_layers: int = 1):
        init_edges = nnx.initializers.normal(stddev=1.0 / width)
        init_hid = nnx.initializers.normal(stddev=1.0 / jnp.sqrt(width))
        bias_init_zero = nnx.initializers.zeros
        
        layers = []
        layers.append(
            MuPFirstLayer(in_features, width, kernel_init=init_edges, bias_init=init_edges, rngs=rngs)
        )
        
        for _ in range(num_hidden_layers - 1):
            layers.append(
                nnx.Linear(width, width, kernel_init=init_hid, bias_init=bias_init_zero, rngs=rngs)
            )
            
        self.hidden_layers = nnx.List(layers)
        # Reuses NTKLinear for 1/sqrt(fan_in) scaling on the output forward pass
        self.out_layer = NTKLinear(width, out_features, kernel_init=init_edges, bias_init=bias_init_zero, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.hidden_layers:
            x = nnx.relu(layer(x))
        return self.out_layer(x)