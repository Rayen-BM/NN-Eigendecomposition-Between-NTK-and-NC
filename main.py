import argparse
import jax
import jax.numpy as jnp
from flax import nnx
import wandb
import data

from model import NTKModel, StandardNetwork, MaximalUpdateNetwork
from geometry import get_continuous_margin
from plots import log_label_identity_matrix
from eval import KernelEvaluator
from trainer import Trainer

def prepare_data(bsz, seed, shape, torus=False):
    # 1. Generate pure 2D datasets directly
    train_points, train_labels = data.generate_data(num_samples=500, seed=seed, shape=shape)
    val_points, val_labels = data.generate_data(num_samples=100, seed=seed+1, shape=shape)

    # 2. Setup eval data (train data + corners)
    corners = jnp.array([[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]])
    corner_labels = jnp.array([1, 1, 1, 1])
    
    eval_inputs = jnp.concatenate([train_points, corners], axis=0)
    eval_labels = jnp.concatenate([train_labels, corner_labels], axis=0)

    # 3. Calculate Euclidean margins and sort in 2D
    margins = get_continuous_margin(eval_inputs, shape)
    sort_indices = jnp.argsort(margins)
    margins = margins[sort_indices]
    
    eval_inputs = eval_inputs[sort_indices]
    eval_labels = eval_labels[sort_indices]
    
    val_margins = get_continuous_margin(val_points, shape)
    sort_val = jnp.argsort(val_margins)
    val_margins = val_margins[sort_val]
    
    val_points_sorted = val_points[sort_val]
    val_labels_sorted = val_labels[sort_val]
    
    # 4. --- FEATURE ENGINEERING: Project coordinates to Torus ---
    if torus:
        def project_to_torus(x_2d):
            x_scaled = x_2d * (jnp.pi / 2.0)
            return jnp.column_stack([
                jnp.cos(x_scaled[:, 0]), jnp.sin(x_scaled[:, 0]),
                jnp.cos(x_scaled[:, 1]), jnp.sin(x_scaled[:, 1])
            ])
            
        eval_inputs = project_to_torus(eval_inputs)
        val_points_sorted = project_to_torus(val_points_sorted)
        train_points = project_to_torus(train_points)

    # 5. Instantiate DataLoaders cleanly with final processed points
    train_loader = data.DataLoader(train_points, train_labels, bsz, shuffle=True, seed=seed)
    val_loader = data.DataLoader(val_points_sorted, val_labels_sorted, bsz, shuffle=False, seed=seed+1)

    eval_tuple = (eval_inputs, eval_labels)
    val_tuple = (val_points_sorted, val_labels_sorted)
    
    return train_loader, val_loader, eval_tuple, val_tuple, margins, val_margins

def main():
    # WandB config is populated by argparse in __main__
    cfg = wandb.config
    
    train_loader, val_loader, eval_tuple, val_tuple, margins, val_margins = prepare_data(
        cfg.batch_size, cfg.seed, cfg.shape, torus=cfg.torus
    )
    
    log_label_identity_matrix(eval_tuple[1])

    rngs = nnx.Rngs(cfg.seed)
    in_dim = 4 if cfg.torus else 2
    
    if cfg.init == "NTK":
        model = NTKModel(in_dim, cfg.width, 2, rngs=rngs, num_hidden_layers=cfg.num_hidden_layers)
    elif cfg.init == "max":
        model = MaximalUpdateNetwork(in_dim, cfg.width, 2, rngs=rngs, num_hidden_layers=cfg.num_hidden_layers)
    else: 
        model = StandardNetwork(in_dim, cfg.width, 2, rngs=rngs, num_hidden_layers=cfg.num_hidden_layers)
        
    evaluator = KernelEvaluator(
        eval_data=eval_tuple,
        val_data=val_tuple,
        margins=margins,
        val_margins=val_margins,
        shape=cfg.shape,
        torus=cfg.torus
    )

    trainer = Trainer(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader,
        evaluator=evaluator,
        lr=cfg.learning_rate, 
        epochs=cfg.epochs
    )
    
    try:
        trainer.train()
    finally:
        print("Training complete. Syncing logs...")
        jax.effects_barrier() 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shape Eigenfunctions Experiment Runner")
    
    parser.add_argument("--shape", type=str, default="circle", choices=["xor_circles", "circle", "heart", "star", "astroid"])
    parser.add_argument("--init", type=str, default="NTK", choices=["NTK", "standard", "max"])
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_hidden_layers", type=int, default=1)
    parser.add_argument("--torus", action="store_true", help="Project coordinates to torus")
    
    args = parser.parse_args()
    
    # Dynamically generate the run name based on hyperparameters
    run_name = f"s:{args.shape}_w:{args.width}_i:{args.init}_l:{args.num_hidden_layers}_t:{args.torus}"
    
    wandb.init(
        project="ENTER NAME OF PROJECT HERE",  # Replace with your WandB project name
        name=run_name,
        config=vars(args)
    )
    
    main()