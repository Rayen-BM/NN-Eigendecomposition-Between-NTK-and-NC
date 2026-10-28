import matplotlib
# Use Agg backend for headless server plotting to prevent memory leaks
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax.numpy as jnp
import wandb
from typing import Optional, List

def plot_kernel(kernel_matrix: jax.Array, title: str, margins: Optional[jax.Array] = None, 
                extent: Optional[List[float]] = None, xlabel: Optional[str] = None, 
                ylabel: Optional[str] = None, boundary_coords: Optional[jax.Array] = None) -> plt.Figure:
    """Plots a normalized Kernel matrix (NTK or NNGP)."""
    frob_norm = jnp.linalg.norm(kernel_matrix)
    kernel_matrix = kernel_matrix / jnp.maximum(frob_norm, 1e-12)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(kernel_matrix, cmap="inferno", origin="upper", extent=extent) 
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalized Kernel value", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    # --- Boundary Visualization ---
    if margins is not None and extent is None:
        # Since margins are sorted, the boundary is exactly where it crosses 0.0
        boundary_idx = int(jnp.argmin(jnp.abs(margins)))
        ax.axhline(boundary_idx, color='cyan', linestyle='--', linewidth=1.5, alpha=0.8, label="Boundary")
        ax.axvline(boundary_idx, color='cyan', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.legend(loc="upper right", fontsize=10)
        
    elif extent is not None and boundary_coords is not None:
        # For extent maps (like diag_NTK), draw lines at exact continuous coordinates
        for idx, bc in enumerate(boundary_coords):
            label = "Boundary" if idx == 0 else ""
            ax.axhline(bc, color='cyan', linestyle='--', linewidth=1.5, alpha=0.8, label=label)
            ax.axvline(bc, color='cyan', linestyle='--', linewidth=1.5, alpha=0.8)
        if len(boundary_coords) > 0:
            ax.legend(loc="upper right", fontsize=10)

    # --- Axis Labeling & Ticks ---
    if margins is not None:
        num_ticks = 7
        tick_positions = jnp.linspace(0, kernel_matrix.shape[0] - 1, num_ticks).astype(int)
        tick_labels = [f"{float(margins[idx]):.2f}" for idx in tick_positions]
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
        
        ax.set_xlabel(xlabel if xlabel else "Margin (Distance to Boundary)", fontsize=12)
        ax.set_ylabel(ylabel if ylabel else "Margin (Distance to Boundary)", fontsize=12)
    elif extent is not None:
        ax.set_xlabel(xlabel if xlabel else "Coordinate", fontsize=12)
        ax.set_ylabel(ylabel if ylabel else "Coordinate", fontsize=12)
    else:
        ax.set_xlabel(xlabel if xlabel else "Sample Index ($j$)", fontsize=12)
        ax.set_ylabel(ylabel if ylabel else "Sample Index ($i$)", fontsize=12)
        
    fig.tight_layout()
    return fig

def log_label_identity_matrix(eval_labels: jax.Array):
    """Logs a binary matrix showing label similarities across the evaluation set."""
    num_samples = eval_labels.shape[0]
    identity_matrix = (eval_labels[:, None] == eval_labels[None, :]).astype(jnp.float32)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(identity_matrix, cmap='binary_r', origin='upper', interpolation='nearest')
    ax.set_title(f'Evaluation Data Label Identity Matrix (N={num_samples})', fontsize=16, fontweight='bold')
    ax.set_xlabel('Sample Index (j)', fontsize=14)
    ax.set_ylabel('Sample Index (i)', fontsize=14)
    
    ticks = jnp.arange(0, num_samples, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    
    cbar = fig.colorbar(im, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Different Label (0)', 'Similar Label (1)'], fontsize=12)
    
    plt.tight_layout()
    wandb.log({"eval/label_identity_matrix": wandb.Image(fig)})
    plt.close(fig)