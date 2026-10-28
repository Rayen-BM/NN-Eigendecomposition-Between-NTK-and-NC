import jax
import jax.numpy as jnp

from geometry import get_continuous_margin

def generate_data(num_samples: int = 1000, seed: int = 42, noise = 0 ,shape: str = "circle"):
    """
    Generates data using JAX's functional PRNG system.
    """
    key = jax.random.PRNGKey(seed)
    points_key, noise_key = jax.random.split(key)
    
    points = jax.random.uniform(
        points_key, 
        shape=(num_samples, 2), 
        minval=-2.0, 
        maxval=2.0
    )
    
    points = points + noise
    
    # 1 for exterior, 0 for interior
    margins = get_continuous_margin(points, shape)
    labels = (margins > 0.0).astype(jnp.int32)
    
    return points, labels

class DataLoader:
    def __init__(self, points, labels, batch_size, shuffle=True, drop_last=False, seed=42):
        self.points = points
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.key = jax.random.PRNGKey(seed)
        self.num_samples = len(points)

    def __iter__(self):
        if self.shuffle:
            self.key, subkey = jax.random.split(self.key)
            indices = jax.random.permutation(subkey, self.num_samples)
        else:
            indices = jnp.arange(self.num_samples)

        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            
            if len(batch_indices) < self.batch_size and self.drop_last:
                continue
                
            yield {
                "input": self.points[batch_indices],
                "label": self.labels[batch_indices]
            }

def get_toy(batch_size=32, num_samples=100, seed=42, shape="circle"):
    points, labels = generate_data(num_samples, seed, shape)
    return (points, labels), DataLoader(points, labels, batch_size, seed=seed)
