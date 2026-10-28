import jax
import jax.numpy as jnp
import wandb
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any

from kernels import get_ntk, get_nngp
from plots import plot_kernel
from geometry import get_continuous_margin

class KernelEvaluator:
    """Evaluates Kernel metrics, Neural Collapse, and visualizes eigenfunctions."""
    def __init__(self, eval_data: Tuple[jax.Array, jax.Array], val_data: Tuple[jax.Array, jax.Array], 
                 margins: jax.Array, val_margins: jax.Array, shape: str, torus: bool = False,
                 grid_size: int = 200, batch_size: int = 200, num_diag: int = 100, num_eigen_plots: int = 4):
        self.eval_data = eval_data
        self.val_data = val_data
        self.margins = margins
        self.val_margins = val_margins
        self.shape = shape
        self.torus = torus
        
        self.k0 = None 
        
        self.grid_size = grid_size
        self.batch_size = batch_size
        self.num_diag = num_diag
        self.num_eigen_plots = num_eigen_plots

    def _project_to_torus(self, x: jax.Array) -> jax.Array:
        """Projects 2D coordinates to 4D Torus coordinates."""
        x_scaled = x * (jnp.pi / 2.0)
        return jnp.column_stack([
            jnp.cos(x_scaled[:, 0]), jnp.sin(x_scaled[:, 0]),
            jnp.cos(x_scaled[:, 1]), jnp.sin(x_scaled[:, 1])
        ])

    def evaluate(self, model, epoch: int, plot_val: bool = True):
        """Orchestrates the evaluation pipeline and logs metrics to wandb."""
        X_eval, Y_eval = self.eval_data
        log_dict = {}

        # 1. Basic Kernel Metrics
        ntk, basic_logs = self._compute_basic_metrics(model, X_eval, Y_eval, plot_val)
        log_dict.update(basic_logs)

        # 2. Eigenvalue Spectrum
        evals, evecs, spectrum_logs = self._compute_eigen_spectrum(ntk)
        log_dict.update(spectrum_logs)

        # 3. Grid Cross-Kernel (Continuous Eigenfunctions)
        grid_logs = self._evaluate_continuous_eigenfunctions(model, X_eval, evals, evecs)
        log_dict.update(grid_logs)

        # 4. Diagonal Slice (Nyström Extension)
        diag_logs = self._evaluate_diagonal_slice(model, X_eval, evals, evecs)
        log_dict.update(diag_logs)

        # 5. Neural Collapse
        collapse_logs = self._compute_collapse_metrics(ntk, Y_eval, evals)
        log_dict.update(collapse_logs)

        wandb.log(log_dict)

    def _compute_basic_metrics(self, model, X_eval: jax.Array, Y_eval: jax.Array, plot_val: bool) -> Tuple[jax.Array, Dict[str, Any]]:
        """Computes NTK/NNGP matrices, alignment scores, and generates kernel plots."""
        log_dict = {}
        
        ntk = get_ntk(model, X_eval)
        y_target = jnp.outer(Y_eval, Y_eval)
        
        inner_prod = jnp.sum(ntk * y_target)
        ntk_norm = jnp.linalg.norm(ntk)
        target_norm = jnp.linalg.norm(y_target)
        kernel_alignment_score = inner_prod / (ntk_norm * target_norm + 1e-7)  
        
        if self.k0 is None:
            self.k0 = ntk 

        ntk_fig = plot_kernel(ntk, "Frobenius Normalized NTK Matrix", margins=self.margins)
        nngp_matrix = get_nngp(model, X_eval)
        nngp_fig = plot_kernel(nngp_matrix, "Frobenius Normalized NNGP Matrix", margins=self.margins)

        log_dict.update({
            "kernel/alignment_score": kernel_alignment_score.item(),
            "kernel/NTK": wandb.Image(ntk_fig),
            "kernel/NNGP": wandb.Image(nngp_fig)
        })
        plt.close(ntk_fig)
        plt.close(nngp_fig)

        if plot_val:
            val_X, _ = self.val_data
            val_ntk = get_ntk(model, val_X)
            val_nngp = get_nngp(model, val_X)
            fig_ntk = plot_kernel(val_ntk, "Validation Data NTK", margins=self.val_margins)
            fig_nngp = plot_kernel(val_nngp, "Validation Data NNGP", margins=self.val_margins)
            
            log_dict.update({
                "val_kernel/NTK": wandb.Image(fig_ntk),
                "val_kernel/NNGP": wandb.Image(fig_nngp)
            })    
            plt.close(fig_ntk)
            plt.close(fig_nngp)
            
        return ntk, log_dict

    def _compute_eigen_spectrum(self, ntk: jax.Array) -> Tuple[jax.Array, jax.Array, Dict[str, Any]]:
        """Performs eigendecomposition and plots the eigenvalue spectrum."""
        evals, evecs = jnp.linalg.eigh(ntk)
        idx = jnp.argsort(evals)[::-1]
        evals = jnp.maximum(evals[idx], 1e-10) 
        evecs = evecs[:, idx]

        prob = evals / (jnp.sum(evals) + 1e-7)
        effective_rank = jnp.exp(-jnp.sum(prob * jnp.log(prob + 1e-7)))
        kernel_diff = jnp.linalg.norm(ntk - self.k0) / jnp.linalg.norm(self.k0)

        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(jnp.log10(evals + 1e-12), 'o-', ms=4, alpha=0.7, label='Eigenvalues') 
        ax1.set_title("NTK Eigenvalue Spectrum (Log10)", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Eigenvalue Index", fontsize=10)
        ax1.set_ylabel("Log10 Lambda", fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend()
        fig1.tight_layout()

        log_dict = {
            "kernel/top_eigenvalue": evals[0].item(),
            "kernel/effective_rank": effective_rank.item(),
            "kernel/diff_from_init": kernel_diff.item(),
            "kernel/eigenvalue_spectrum": wandb.Image(fig1)
        }
        plt.close(fig1)
        
        return evals, evecs, log_dict

    def _evaluate_continuous_eigenfunctions(self, model, X_eval: jax.Array, 
                                            evals: jax.Array, evecs: jax.Array) -> Dict[str, Any]:
        """Evaluates eigenfunctions over a dense 2D grid using Nyström extensions."""
        log_dict = {}
        grid_x = jnp.linspace(-2.0, 2.0, self.grid_size)
        grid_y = jnp.linspace(-2.0, 2.0, self.grid_size)
        X1, X2 = jnp.meshgrid(grid_x, grid_y)
        X_grid = jnp.column_stack([X1.ravel(), X2.ravel()])
        
        X_grid_kernel = self._project_to_torus(X_grid) if self.torus else X_grid
        
        k_grid_eval_list = []
        for i in range(0, X_grid_kernel.shape[0], self.batch_size):
            X_grid_batch = X_grid_kernel[i:i+self.batch_size] 
            
            # --- OPTIMIZED CROSS KERNEL ---
            # Directly calculate K(X_grid_batch, X_eval) without vstacking!
            cross_part = get_ntk(model, X_grid_batch, X_eval)
            k_grid_eval_list.append(cross_part)
            
        k_grid_eval = jnp.vstack(k_grid_eval_list)
        
        grid_margins = get_continuous_margin(X_grid, self.shape)
        margin_z = grid_margins.reshape(X1.shape)
        
        for i in range(min(self.num_eigen_plots, evecs.shape[1])):     
            grid_eigenfunction = jnp.dot(k_grid_eval, evecs[:, i]) / evals[i]
            grid_z = grid_eigenfunction.reshape(X1.shape)
            
            fig, ax = plt.subplots(figsize=(6, 5))
            tpc = ax.contourf(X1, X2, grid_z, levels=50, cmap='viridis')
            ax.contour(X1, X2, margin_z, levels=[0.0], colors='red', linewidths=2.5, linestyles='--')
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_title(f"Continuous Eigenfunction {i}")
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            fig.colorbar(tpc, ax=ax)
            
            log_dict[f"kernel/eigenfunction_{i}"] = wandb.Image(fig)
            plt.close(fig)
            
        return log_dict

    def _evaluate_diagonal_slice(self, model, X_eval: jax.Array, 
                                 evals: jax.Array, evecs: jax.Array) -> Dict[str, Any]:
        """Evaluates eigenfunctions along the x1=x2 diagonal slice."""
        log_dict = {}
        diag_coords = jnp.linspace(-2.0, 2.0, self.num_diag)
        diag_points = jnp.column_stack([diag_coords, diag_coords])
        
        diag_points_kernel = self._project_to_torus(diag_points) if self.torus else diag_points
            
        diag_ntk = get_ntk(model, diag_points_kernel)
        
        diag_margins = get_continuous_margin(diag_points, self.shape)
        signs = jnp.sign(diag_margins)
        crossings = jnp.where(signs[:-1] != signs[1:])[0]
        boundary_coords = (diag_coords[crossings] + diag_coords[crossings + 1]) / 2.0
        
        fig_diag_ntk = plot_kernel(
            diag_ntk, "Diagonal NTK Matrix", 
            extent=[-2.0, 2.0, 2.0, -2.0],
            xlabel="Coordinate $x_1 = x_2$", 
            ylabel="Coordinate $x_1 = x_2$",
            boundary_coords=boundary_coords
        )
        log_dict["kernel/diag_NTK"] = wandb.Image(fig_diag_ntk)
        plt.close(fig_diag_ntk)

        cross_part_diag = get_ntk(model, diag_points_kernel, X_eval)

        fig_diag_eig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig_diag_eig.suptitle("Top 4 Eigenfunctions (1D Diagonal Slice)", fontsize=14, fontweight='bold')
        axes = axes.flatten()
        
        for i in range(min(self.num_eigen_plots, evecs.shape[1])):
            slice_eigenfunction = jnp.dot(cross_part_diag, evecs[:, i]) / evals[i]
            axes[i].plot(diag_coords, slice_eigenfunction, color='tab:red', lw=2)
            
            for idx, bc in enumerate(boundary_coords):
                label = "Boundary" if idx == 0 else ""
                axes[i].axvline(bc, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label=label)
            
            axes[i].set_title(f"Diagonal Eigenfunction {i}")
            axes[i].set_xlabel("$x_1 = x_2$")
            axes[i].grid(True, linestyle='--', alpha=0.6)
            if len(boundary_coords) > 0:
                axes[i].legend(loc="upper right", fontsize=8)
            
        fig_diag_eig.tight_layout()
        log_dict["kernel/diag_NTK_eigenfunctions"] = wandb.Image(fig_diag_eig)
        plt.close(fig_diag_eig)
        
        return log_dict

    def _compute_collapse_metrics(self, ntk: jax.Array, Y_eval: jax.Array, evals: jax.Array) -> Dict[str, Any]:
        """Calculates Neural Tangent Collapse metrics."""
        diag = jnp.diag(ntk)
        inv_norm = 1.0 / jnp.sqrt(jnp.maximum(diag, 1e-12))
        norm_ntk = ntk * jnp.outer(inv_norm, inv_norm)

        Y_bipolar = 2.0 * Y_eval - 1.0
        y_target_bipolar = jnp.outer(Y_bipolar, Y_bipolar)
        
        inner_prod_norm = jnp.sum(norm_ntk * y_target_bipolar)
        ntk_frob_norm = jnp.linalg.norm(norm_ntk)
        target_frob_norm = jnp.linalg.norm(y_target_bipolar)
        normalized_alignment = inner_prod_norm / (ntk_frob_norm * target_frob_norm + 1e-7)

        intra_mask = (y_target_bipolar == 1)
        inter_mask = (y_target_bipolar == -1)
        intra_variance = jnp.var(jnp.where(intra_mask, norm_ntk, 0.0), where=intra_mask)
        inter_variance = jnp.var(jnp.where(inter_mask, norm_ntk, 0.0), where=inter_mask)

        trace = jnp.sum(evals)
        top_eigenvalue_ratio = evals[0] / (trace + 1e-7)

        return {
            "collapse/normalized_alignment": normalized_alignment.item(),
            "collapse/intra_class_variance": intra_variance.item(),
            "collapse/inter_class_variance": inter_variance.item(),
            "collapse/spectral_dominance": top_eigenvalue_ratio.item()
        }