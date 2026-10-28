import jax
import jax.numpy as jnp
from flax import nnx
from typing import Optional

def _get_jacobians(model: nnx.Module, x: jax.Array, **kwargs) -> jax.Array:
    """Computes the flattened per-sample Jacobians for all model parameters."""
    graphdef, state = nnx.split(model)

    def scalar_forward(state, x_sample):
        x_batched = jnp.expand_dims(x_sample, axis=0)
        # apply(state) returns a tuple of (output, state). [0] gets the output.
        logits = graphdef.apply(state)(x_batched, **kwargs)[0]
        return jnp.sum(logits)

    # Gradient with respect to the state (parameters)
    grad_fn = jax.grad(scalar_forward, argnums=0)
    
    # Vectorize across the batch dimension of x
    per_sample_grad_state = jax.vmap(grad_fn, in_axes=(None, 0))(state, x)
    
    # Extract only the learnable parameters, not fixed buffers
    param_grads = per_sample_grad_state.filter(nnx.Param)
    flat_grads, _ = jax.tree_util.tree_flatten(param_grads)
    
    # Flatten and concatenate all parameter gradients into a (batch_size, num_params) matrix
    batch_size = x.shape[0]
    flat_grads_matrix = jnp.concatenate(
        [g.reshape(batch_size, -1) for g in flat_grads], 
        axis=1
    )
    return flat_grads_matrix


def get_ntk(model: nnx.Module, x1: jax.Array, x2: Optional[jax.Array] = None, **kwargs) -> jax.Array:
    """
    Computes the empirical Neural Tangent Kernel (NTK).
    If x2 is None, computes the auto-covariance matrix K(x1, x1).
    If x2 is provided, computes the cross-covariance matrix K(x1, x2).
    """
    J1 = _get_jacobians(model, x1, **kwargs)
    
    if x2 is None:
        return jnp.dot(J1, J1.T)
        
    J2 = _get_jacobians(model, x2, **kwargs)
    return jnp.dot(J1, J2.T)


def _get_features(model: nnx.Module, x: jax.Array, **kwargs) -> jax.Array:
    """Extracts flattened forward-pass predictions/features."""
    graphdef, state = nnx.split(model)

    def forward_fn(state, x_sample):
        return graphdef.apply(state)(x_sample, **kwargs)[0]

    predictions = forward_fn(state, x)
    batch_size = x.shape[0]
    return predictions.reshape(batch_size, -1)


def get_nngp(model: nnx.Module, x1: jax.Array, x2: Optional[jax.Array] = None, **kwargs) -> jax.Array:
    """
    Computes the empirical Neural Network Gaussian Process (NNGP) Kernel.
    If x2 is None, computes the auto-covariance matrix K(x1, x1).
    If x2 is provided, computes the cross-covariance matrix K(x1, x2).
    """
    F1 = _get_features(model, x1, **kwargs)
    
    if x2 is None:
        return jnp.dot(F1, F1.T)
        
    F2 = _get_features(model, x2, **kwargs)
    return jnp.dot(F1, F2.T)