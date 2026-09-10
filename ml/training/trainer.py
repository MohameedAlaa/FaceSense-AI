"""
FaceSense AI - Training Pipeline Trainer
Implements training loop, validation loop, LR scheduler, checkpointing,
best model tracking on Macro F1, and final one-time test evaluation.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from ml.data.dataloader import (
    build_dataloaders,
    compute_class_weights,
    load_yaml_config,
)
from ml.models.builder import (
    build_model_from_config,
    get_model_summary,
)
from ml.training.metrics import (
    compute_evaluation_metrics,
    format_confusion_matrix_ascii,
    save_metrics,
)
from ml.training.utils import (
    EarlyStopping,
    MetricTracker,
    get_device,
    set_seed,
)


def _setup_cpu_performance(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Configures PyTorch CPU performance settings from config.
    Returns a dict of effective settings for logging.
    """
    cpu_cfg = config.get("cpu", {})

    # Intra-op thread count: tune to avoid cache thrashing on many-core CPUs
    num_threads = int(cpu_cfg.get("num_threads", torch.get_num_threads()))
    torch.set_num_threads(num_threads)

    return {
        "num_threads": num_threads,
        "use_compile": bool(cpu_cfg.get("use_compile", False)),
        "use_bf16_autocast": bool(cpu_cfg.get("use_bf16_autocast", False)),
    }


class EmotionTrainer:
    """
    Modular Trainer for FER2013 facial expression recognition.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.class_names = class_names or config["classes"]["names"]
        self.num_classes = len(self.class_names)

        # ── CPU performance configuration ────────────────────────────
        self.cpu_settings = _setup_cpu_performance(config)

        # Device setup
        if device is None:
            self.device, self.device_info = get_device()
        else:
            self.device = device
            self.device_info = {"device": str(device)}

        # ── BF16 autocast context (CPU only, requires AVX-512 BF16) ──
        self._use_bf16 = (
            self.cpu_settings["use_bf16_autocast"]
            and self.device.type == "cpu"
        )
        self._autocast_ctx = (
            lambda: torch.autocast(device_type="cpu", dtype=torch.bfloat16)
            if self._use_bf16
            else torch.autocast(device_type="cpu", enabled=False)
        )

        # ── Model (optionally compiled) ───────────────────────────────
        base_model = model.to(self.device)
        if (
            self.cpu_settings["use_compile"]
            and hasattr(torch, "compile")
            and self.device.type == "cpu"
        ):
            try:
                self.model = torch.compile(base_model)
                self._compiled = True
            except Exception as e:
                print(f"[Warning] torch.compile failed ({e}); using eager mode.")
                self.model = base_model
                self._compiled = False
        else:
            self.model = base_model
            self._compiled = False

        # Training params
        tr_cfg = self.config.get("training", {})
        self.epochs = int(tr_cfg.get("epochs", 50))
        self.lr = float(tr_cfg.get("learning_rate", 0.001))
        self.weight_decay = float(tr_cfg.get("weight_decay", 0.0001))
        self.label_smoothing = float(tr_cfg.get("label_smoothing", 0.05))
        self.patience = int(tr_cfg.get("early_stopping_patience", 10))

        # Optimizer: AdamW
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Loss function
        # WeightedRandomSampler is used by default in DataLoader.
        # We ensure class-weighted loss is NOT combined with WeightedRandomSampler.
        use_sampler = bool(tr_cfg.get("use_weighted_sampler", True))
        use_weighted_loss = bool(tr_cfg.get("use_class_weighted_loss", False))

        if use_weighted_loss and not use_sampler:
            # Only use class weights if sampler is disabled
            train_labels = self.train_loader.dataset.get_labels()
            class_weights = compute_class_weights(train_labels, num_classes=self.num_classes).to(self.device)
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=self.label_smoothing,
            )
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        # Scheduler: ReduceLROnPlateau or WarmupCosineAnnealingLR
        sched_cfg = tr_cfg.get("scheduler", {})
        sched_name = str(sched_cfg.get("name", "ReduceLROnPlateau")).lower()

        if "cosine" in sched_name or "warmup" in sched_name:
            warmup_epochs = int(sched_cfg.get("warmup_epochs", 1))
            min_lr = float(sched_cfg.get("min_lr", 1e-6))
            t_max = max(1, self.epochs - warmup_epochs)
            
            if warmup_epochs > 0:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=float(sched_cfg.get("warmup_start_factor", 0.1)),
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=t_max,
                    eta_min=min_lr,
                )
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs],
                )
            else:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.epochs,
                    eta_min=min_lr,
                )
            self._scheduler_is_plateau = False
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode=sched_cfg.get("mode", "max"),
                factor=float(sched_cfg.get("factor", 0.5)),
                patience=int(sched_cfg.get("patience", 3)),
                min_lr=float(sched_cfg.get("min_lr", 1e-6)),
            )
            self._scheduler_is_plateau = True

        # Early Stopping
        self.early_stopping = EarlyStopping(
            patience=self.patience,
            mode="max",
        )

        # Checkpoints & Outputs
        paths_cfg = self.config.get("paths", {})
        self.checkpoints_dir = Path(paths_cfg.get("checkpoints_dir", "ml/models/checkpoints"))
        self.best_checkpoint_path = Path(
            paths_cfg.get("best_checkpoint_path", self.checkpoints_dir / "best_model.pt")
        )
        self.metrics_dir = Path(paths_cfg.get("outputs_metrics_dir", "outputs/metrics"))
        self.plots_dir = Path(paths_cfg.get("outputs_plots_dir", "outputs/plots"))

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # History Tracker
        self.history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_macro_f1": [],
            "lr": [],
        }

    def train_epoch(self) -> Tuple[float, float]:
        """Runs one full training epoch over train_loader."""
        self.model.train()
        tracker = MetricTracker()

        with self._autocast_ctx():
            for batch_idx, (images, targets) in enumerate(self.train_loader):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                preds = outputs.argmax(dim=1)
                tracker.update(loss.item(), preds, targets)

        return tracker.avg_loss, tracker.accuracy

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluates the model on a DataLoader (validation or test set).
        Returns:
            (avg_loss, metrics_dictionary)
        """
        self.model.eval()
        tracker = MetricTracker()
        all_preds = []
        all_targets = []

        with self._autocast_ctx():
            for images, targets in data_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                preds = outputs.argmax(dim=1)
                tracker.update(loss.item(), preds, targets)

                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(targets.cpu().tolist())

        metrics = compute_evaluation_metrics(all_targets, all_preds, self.class_names)
        metrics["loss"] = tracker.avg_loss
        return tracker.avg_loss, metrics

    def save_checkpoint(
        self,
        epoch: int,
        val_metrics: Dict[str, Any],
        filepath: Union[str, Path],
        is_best: bool = False,
    ):
        """Saves model weights, optimizer, scheduler, config, and epoch metadata."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_metrics.get("loss"),
            "val_accuracy": val_metrics.get("accuracy"),
            "val_macro_f1": val_metrics.get("macro_f1"),
            "is_best": is_best,
            "class_names": self.class_names,
            "config": self.config,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Loads model weights and training state from checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint and hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and hasattr(self, "scheduler"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint

    def train(self, max_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Executes full training loop with validation, early stopping, and test evaluation.
        """
        epochs_to_run = max_epochs if max_epochs is not None else self.epochs
        print("=" * 70)
        print(f"FaceSense AI Training Pipeline")
        print(f"Device: {self.device_info}")
        print(f"Architecture: {self.model.__class__.__name__}")
        print(f"Compiled: {self._compiled}  |  BF16 Autocast: {self._use_bf16}")
        print(f"CPU Threads: {self.cpu_settings['num_threads']}")
        print(f"Total Epochs: {epochs_to_run} | Batch Size: {self.config['dataloader']['batch_size']}")
        print(f"Primary Selection Metric: Validation Macro F1")
        print("=" * 70)

        start_time = time.time()
        best_val_macro_f1 = -1.0

        for epoch in range(1, epochs_to_run + 1):
            epoch_start = time.time()
            train_loss, train_acc = self.train_epoch()
            val_loss, val_metrics = self.evaluate(self.val_loader)
            val_acc = val_metrics["accuracy"]
            val_macro_f1 = val_metrics["macro_f1"]

            # Step scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            if getattr(self, "_scheduler_is_plateau", True):
                self.scheduler.step(val_macro_f1)
            else:
                self.scheduler.step()

            # Record history
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_macro_f1"].append(val_macro_f1)
            self.history["lr"].append(current_lr)

            # Check for best model
            is_best = self.early_stopping.step(val_macro_f1, epoch)
            epoch_time = time.time() - epoch_start

            print(
                f"Epoch [{epoch:02d}/{epochs_to_run:02d}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                f"Val Macro F1: {val_macro_f1:.2f}% | LR: {current_lr:.1e} | "
                f"Time: {epoch_time:.1f}s" + (" [*BEST*]" if is_best else "")
            )

            # Save latest checkpoint
            latest_ckpt_path = self.checkpoints_dir / "latest_checkpoint.pt"
            self.save_checkpoint(epoch, val_metrics, latest_ckpt_path, is_best=False)

            # Save best checkpoint
            if is_best:
                best_val_macro_f1 = val_macro_f1
                self.save_checkpoint(epoch, val_metrics, self.best_checkpoint_path, is_best=True)

            if self.early_stopping.early_stop:
                print(f"\n[Early Stopping] No improvement in validation Macro F1 for {self.patience} epochs.")
                print(f"Best Epoch: {self.early_stopping.best_epoch} with Macro F1: {self.early_stopping.best_score:.2f}%")
                break

        total_training_time = time.time() - start_time
        print(f"\nTraining completed in {total_training_time / 60:.2f} minutes.")

        # Save training history
        save_metrics(self.history, output_dir=self.metrics_dir, prefix="training_history")

        # -------------------------------------------------------------
        # Evaluate Best Checkpoint on TEST Set Exactly Once
        # -------------------------------------------------------------
        print("\n" + "=" * 70)
        print("Evaluating Best Model Checkpoint on TEST Set (One-Time Final Evaluation)")
        print("=" * 70)

        if self.best_checkpoint_path.exists():
            self.load_checkpoint(self.best_checkpoint_path)
            print(f"Loaded best checkpoint from: {self.best_checkpoint_path}")
        else:
            print("Warning: Best checkpoint not found. Using current model weights.")

        test_loss, test_metrics = self.evaluate(self.test_loader)
        
        print("\nTest Set Results:")
        print(f"  Test Loss:        {test_loss:.4f}")
        print(f"  Overall Accuracy: {test_metrics['accuracy']:.2f}%")
        print(f"  Macro F1:         {test_metrics['macro_f1']:.2f}%")
        print(f"  Weighted F1:      {test_metrics['weighted_f1']:.2f}%\n")

        print("Confusion Matrix:")
        print(format_confusion_matrix_ascii(test_metrics["confusion_matrix"], self.class_names))
        print("\nClassification Report:\n")
        print(test_metrics["classification_report"])

        # Save test evaluation metrics
        test_metric_file = save_metrics(test_metrics, output_dir=self.metrics_dir, prefix="test_metrics")
        print(f"Test metrics saved to: {test_metric_file}")
        print("=" * 70)

        return {
            "history": self.history,
            "best_val_macro_f1": best_val_macro_f1,
            "test_metrics": test_metrics,
        }


def run_training_pipeline(
    config_path: Union[str, Path] = "configs/config.yaml",
    max_epochs: Optional[int] = None,
) -> Tuple[EmotionTrainer, Dict[str, Any]]:
    """
    Convenience function to initialize and run the full training pipeline from a config.
    """
    config = load_yaml_config(config_path)
    set_seed(int(config.get("project", {}).get("seed", 42)))

    # Build DataLoaders
    train_loader, val_loader, test_loader, metadata = build_dataloaders(config)

    # Build Model
    model = build_model_from_config(config)

    # Initialize Trainer
    trainer = EmotionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        class_names=metadata["class_names"],
    )

    results = trainer.train(max_epochs=max_epochs)
    return trainer, results
