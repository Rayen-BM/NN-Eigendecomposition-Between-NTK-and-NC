import jax
import jax.numpy as jnp
from flax import nnx
import optax
import wandb
from tqdm.auto import tqdm

class Trainer(nnx.Module):
    """Handles the training loop, validation, and metric logging."""
    def __init__(self, model: nnx.Module, train_loader, val_loader, evaluator, lr: float = 1.0, epochs: int = 1000):
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.evaluator = evaluator

        self.metrics = nnx.MultiMetric(
            accuracy=nnx.metrics.Accuracy(),
            loss=nnx.metrics.Average()
        )
        
        self.optimizer = nnx.Optimizer(model, optax.sgd(learning_rate=self.lr), wrt=nnx.Param)
        wandb.config.update({"adjusted_learning_rate": self.lr}, allow_val_change=True)
    
    @nnx.jit
    def train_step(self, batch: dict):
        def loss_fn(model_obj):
            logits = model_obj(batch['input'])
            loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=batch['label']
            ))
            return loss, logits

        (loss, logits), grads = nnx.value_and_grad(loss_fn, has_aux=True)(self.model)
        self.optimizer.update(self.model, grads.filter(nnx.Param)) 
        self.metrics.update(logits=logits, labels=batch['label'], values=loss)
        return loss

    @nnx.jit
    def eval_step(self, batch: dict):
        logits = self.model(batch['input'])
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch['label']
        ))
        self.metrics.update(logits=logits, labels=batch['label'], values=loss)
        return loss

    def train(self):
        """Executes the full training and evaluation loop."""
        # Initial Evaluation (Epoch 0)
        self.evaluator.evaluate(self.model, epoch=0)
        
        epoch_iterator = tqdm(range(self.epochs), desc="Training Epochs")
        
        for epoch in epoch_iterator:
            # --- Training Phase ---
            self.metrics.reset()
            self.model.train()    
            
            for batch in self.train_loader:
                self.train_step(batch)
            
            train_metrics = self.metrics.compute()
            
            # --- Validation Phase ---
            self.metrics.reset() 
            self.model.eval()
            
            for val_batch in self.val_loader:
                self.eval_step(val_batch)
            
            val_metrics = self.metrics.compute()
            
            # Extract scalars for logging and printing
            t_loss = train_metrics['loss'].item()
            t_acc = train_metrics['accuracy'].item()
            v_loss = val_metrics['loss'].item()
            v_acc = val_metrics['accuracy'].item()
            
            # --- Periodic Deep Evaluation ---
            if epoch % 50 == 0 and epoch > 0:
                self.evaluator.evaluate(self.model, epoch=epoch) 
            
            # --- Logging ---
            wandb.log({
                "train/loss": t_loss,
                "train/accuracy": t_acc,
                "val/loss": v_loss,
                "val/accuracy": v_acc, 
                "epoch": epoch
            })
            
            # Update progress bar description
            epoch_iterator.set_postfix(
                train_loss=f"{t_loss:.4f}", 
                val_loss=f"{v_loss:.4f}", 
                val_acc=f"{v_acc:.4f}"
            )