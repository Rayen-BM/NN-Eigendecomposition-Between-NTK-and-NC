import jax
import jax.numpy as jnp

def get_continuous_margin(inputs: jax.Array, shape: str) -> jax.Array:
    """
    Computes the exact continuous distance/margin to the theoretical boundary.
    Negative values indicate interior points; positive values indicate exterior points.
    
    All margins are calculated purely in Euclidean space to guarantee 
    apples-to-apples dataset comparisons across different network topologies.
    """
    X1, X2 = inputs[:, 0], inputs[:, 1]
    
    if shape == "circle":
        return X1**2 + X2**2 - 2.5
        
    elif shape == "heart":
        x_grid, y_grid = X1 * 0.77, X2 * 0.77
        return (x_grid**2 + y_grid**2 - 1)**3 - (x_grid**2) * (y_grid**3)
        
    elif shape == "star":
        r_grid = jnp.sqrt(X1**2 + X2**2)
        theta_grid = jnp.arctan2(X2, X1)
        boundary_radius = 1.54 + 0.4 * jnp.cos(5 * theta_grid)
        return r_grid - boundary_radius
        
    elif shape == "astroid":
        astroid_formula = jnp.abs(X1)**(2/3) + jnp.abs(X2)**(2/3)
        return astroid_formula - 2.603**(2/3)

    elif shape == "xor_circles":
        # Standard Euclidean domain (Two distinct anchor points)
        circle1 = (X1 - 2.0)**2 + (X2 - 2.0)**2 - 5.0
        circle2 = (X1 + 2.0)**2 + (X2 + 2.0)**2 - 5.0
        return jnp.minimum(circle1, circle2)
    
    else:
        raise ValueError(
            f"Unknown shape '{shape}'. Expected one of: "
            "circle, heart, star, astroid, xor_circles."
        )