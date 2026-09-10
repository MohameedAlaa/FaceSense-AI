"""
FaceSense AI - Lightweight Real-Data Performance Benchmark
Measures DataLoader, forward, backward, and optimizer timings using actual
FER2013 training data. Reports estimated epoch duration and training safety.

Usage:
    .\.venv\Scripts\python.exe scripts/benchmark_v2.py
"""

import sys
import time
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.optim import AdamW

from ml.data.dataloader import build_dataloaders, load_yaml_config
from ml.models.baseline_cnn import BaselineEmotionCNN
from ml.models.residual_cnn import ResidualEmotionCNN
from ml.training.utils import set_seed

# -- Constants ----------------------------------------------------------------
CONFIG_PATH = "configs/config.yaml"
WARMUP_BATCHES = 2       # discard first batches (Python JIT / OS cache warmup)
MEASURE_BATCHES = 5      # batches used for timing
LABEL_SMOOTHING = 0.05

# -- Previous baseline numbers (from prior profiling session) -----------------
# These were measured with num_workers=0, batch_size=64, CPU 8 threads
PRIOR_BASELINE = {
    "avg_fetch_ms":    120.0,   # estimated from prior runs
    "avg_forward_ms":  38.0,
    "avg_backward_ms": 85.0,
    "avg_optim_ms":    10.0,
    "avg_step_ms":     133.0,
    "epoch_min":        4.5,
}

SEP = "-" * 68


def time_ms(fn):
    """Returns elapsed milliseconds for a callable, with CPU synchronisation."""
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1000.0


def benchmark_model(model: nn.Module, train_loader, name: str) -> dict:
    """
    Runs a lightweight benchmark for one model using real batches from train_loader.
    Returns a dict of averaged timing results.
    """
    device = torch.device("cpu")
    model = model.to(device).train()
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    loader_iter = iter(train_loader)

    print(f"\n{'='*68}")
    print(f"  Benchmarking: {name}")
    print(f"{'='*68}")

    # -- DataLoader creation time ---------------------------------------------
    t0 = time.perf_counter()
    loader_iter = iter(train_loader)
    loader_init_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  DataLoader iterator creation : {loader_init_ms:7.1f} ms")

    # -- Warmup (not measured) ------------------------------------------------
    print(f"  Warming up ({WARMUP_BATCHES} batches, discarded) ...", end="", flush=True)
    for _ in range(WARMUP_BATCHES):
        try:
            images, targets = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            images, targets = next(loader_iter)
        with torch.no_grad():
            _ = model(images.to(device))
    print(" done")

    # -- Measure fetch time separately ----------------------------------------
    print(f"  Measuring {MEASURE_BATCHES} batches ...", end="", flush=True)
    fetch_times, fwd_times, bwd_times, optim_times = [], [], [], []

    for i in range(MEASURE_BATCHES):
        # Fetch
        t_fetch = time.perf_counter()
        try:
            images, targets = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            images, targets = next(loader_iter)
        fetch_ms = (time.perf_counter() - t_fetch) * 1000.0
        fetch_times.append(fetch_ms)

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Forward
        optimizer.zero_grad()
        t_fwd = time.perf_counter()
        outputs = model(images)
        loss = criterion(outputs, targets)
        fwd_ms = (time.perf_counter() - t_fwd) * 1000.0
        fwd_times.append(fwd_ms)

        # Backward
        t_bwd = time.perf_counter()
        loss.backward()
        bwd_ms = (time.perf_counter() - t_bwd) * 1000.0
        bwd_times.append(bwd_ms)

        # Optimizer step
        t_opt = time.perf_counter()
        optimizer.step()
        opt_ms = (time.perf_counter() - t_opt) * 1000.0
        optim_times.append(opt_ms)

    print(" done\n")

    avg_fetch  = sum(fetch_times)  / len(fetch_times)
    avg_fwd    = sum(fwd_times)    / len(fwd_times)
    avg_bwd    = sum(bwd_times)    / len(bwd_times)
    avg_opt    = sum(optim_times)  / len(optim_times)
    avg_step   = avg_fetch + avg_fwd + avg_bwd + avg_opt

    # Estimate epoch time using actual loader length
    num_batches = len(train_loader)
    epoch_sec   = (avg_step / 1000.0) * num_batches
    epoch_min   = epoch_sec / 60.0

    results = {
        "loader_init_ms": loader_init_ms,
        "avg_fetch_ms":   avg_fetch,
        "avg_forward_ms": avg_fwd,
        "avg_backward_ms": avg_bwd,
        "avg_optim_ms":   avg_opt,
        "avg_step_ms":    avg_step,
        "num_batches":    num_batches,
        "epoch_sec":      epoch_sec,
        "epoch_min":      epoch_min,
    }

    print(f"  {'Metric':<35} {'Value':>12}")
    print(f"  {SEP[:65]}")
    print(f"  {'DataLoader init':<35} {loader_init_ms:>10.1f} ms")
    print(f"  {'Avg batch fetch (5 batches)':<35} {avg_fetch:>10.1f} ms")
    print(f"  {'Avg forward pass':<35} {avg_fwd:>10.1f} ms")
    print(f"  {'Avg backward pass':<35} {avg_bwd:>10.1f} ms")
    print(f"  {'Avg optimizer step':<35} {avg_opt:>10.1f} ms")
    print(f"  {'Avg total step':<35} {avg_step:>10.1f} ms")
    print(f"  {'Training batches / epoch':<35} {num_batches:>10d}")
    print(f"  {'Estimated epoch time':<35} {epoch_sec:>8.0f} s  ({epoch_min:.1f} min)")

    return results


def print_comparison(baseline: dict, residual: dict):
    """Side-by-side comparison table."""
    print(f"\n{'='*68}")
    print("  Comparison: BaselineEmotionCNN  vs  ResidualEmotionCNN")
    print(f"{'='*68}")
    metrics = [
        ("Avg fetch time",     "avg_fetch_ms",   "ms"),
        ("Avg forward pass",   "avg_forward_ms", "ms"),
        ("Avg backward pass",  "avg_backward_ms","ms"),
        ("Avg optimizer step", "avg_optim_ms",   "ms"),
        ("Avg total step",     "avg_step_ms",    "ms"),
        ("Estimated epoch",    "epoch_min",      "min"),
    ]
    print(f"  {'Metric':<30} {'Baseline':>12} {'V2 Residual':>12} {'Overhead':>10}")
    print(f"  {SEP[:65]}")
    for label, key, unit in metrics:
        b_val = baseline.get(key, 0.0)
        r_val = residual.get(key, 0.0)
        overhead = ((r_val / b_val) - 1.0) * 100.0 if b_val > 0 else 0.0
        sign = "+" if overhead >= 0 else ""
        print(f"  {label:<30} {b_val:>10.1f}{unit[0]:1s} {r_val:>10.1f}{unit[0]:1s} {sign}{overhead:>8.1f}%")


def verdict(residual: dict):
    """Prints a human-readable training safety assessment."""
    epoch_min = residual["epoch_min"]
    fetch_ms  = residual["avg_fetch_ms"]
    fwd_ms    = residual["avg_forward_ms"]
    bwd_ms    = residual["avg_backward_ms"]

    print(f"\n{'='*68}")
    print("  Verdict")
    print(f"{'='*68}")

    issues = []
    if fetch_ms > 500:
        issues.append(f"  [WARN]  Fetch time {fetch_ms:.0f} ms — DataLoader bottleneck suspected.")
    if epoch_min > 15:
        issues.append(f"  [WARN]  Epoch ~{epoch_min:.1f} min — 25-epoch run would take ~{25*epoch_min/60:.1f} hours.")
    if epoch_min > 30:
        issues.append(f"  ✗  Epoch >30 min — training is NOT recommended without optimisation.")

    if not issues:
        print(f"  [OK]  Fetch: {fetch_ms:.0f} ms  |  Fwd: {fwd_ms:.0f} ms  |  Bwd: {bwd_ms:.0f} ms")
        print(f"  [OK]  Estimated epoch: {epoch_min:.1f} min")
        print(f"  [OK]  25-epoch run: ~{25*epoch_min/60:.1f} hours  ({25*epoch_min:.0f} min)")
        if epoch_min <= 10:
            print(f"\n  [PASS]  SAFE TO START 25-EPOCH V2 TRAINING.")
        else:
            print(f"\n  [WARN]️  Epoch is longer than ideal but within acceptable range.")
            print(f"      Consider reducing epochs or monitoring first few epochs before committing.")
    else:
        for issue in issues:
            print(issue)
        print(f"\n  [FAIL]  NOT recommended to start full 25-epoch training yet.")
        print(f"      Resolve the above issues first.")
    print(f"{'='*68}\n")


def main():
    print(f"\n{'='*68}")
    print("  FaceSense AI — Real-Data Performance Benchmark")
    print(f"  Config : {CONFIG_PATH}")
    print(f"  Batches measured : {MEASURE_BATCHES}  |  Warmup : {WARMUP_BATCHES}")
    print(f"{'='*68}")

    set_seed(42)
    torch.set_num_threads(8)

    config = load_yaml_config(CONFIG_PATH)

    # Force num_workers=0 for this benchmark (safe on Windows)
    config["dataloader"]["num_workers"] = 0
    config["dataloader"]["persistent_workers"] = False
    config["dataloader"]["prefetch_factor"] = None

    # Build dataloaders once — shared between both models
    print("\n  Building DataLoaders from real FER2013 data ...")
    t0 = time.perf_counter()
    train_loader, val_loader, test_loader, meta = build_dataloaders(config)
    dl_build_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  DataLoaders ready in {dl_build_ms:.0f} ms")
    print(f"  Train batches: {len(train_loader)}  |  "
          f"Train samples: {meta['train_count']}  |  "
          f"Batch size: {config['dataloader']['batch_size']}")

    # -- Baseline model --------------------------------------------------------
    model_cfg = config.get("model", {})
    baseline_model = BaselineEmotionCNN(
        in_channels=1,
        num_classes=7,
        channel_list=model_cfg.get("channel_list", [32, 64, 128, 256]),
        fc_dim=model_cfg.get("fc_dim", 128),
        conv_dropout=0.25,
        fc_dropout=0.5,
    )
    baseline_params = sum(p.numel() for p in baseline_model.parameters())

    # -- Residual V2 model -----------------------------------------------------
    residual_model = ResidualEmotionCNN(
        in_channels=1,
        num_classes=7,
        channel_list=model_cfg.get("channel_list", [32, 64, 128, 256]),
        fc_dim=model_cfg.get("fc_dim", 128),
        conv_dropout=float(model_cfg.get("conv_dropout", 0.1)),
        fc_dropout=float(model_cfg.get("fc_dropout", 0.4)),
    )
    residual_params = sum(p.numel() for p in residual_model.parameters())

    print(f"\n  BaselineEmotionCNN params : {baseline_params:,}")
    print(f"  ResidualEmotionCNN params : {residual_params:,}")
    print(f"  Param overhead            : +{((residual_params/baseline_params)-1)*100:.1f}%")

    # -- Run benchmarks --------------------------------------------------------
    baseline_results  = benchmark_model(baseline_model,  train_loader, "BaselineEmotionCNN")
    residual_results  = benchmark_model(residual_model,  train_loader, "ResidualEmotionCNN (V2)")

    # -- Comparison ------------------------------------------------------------
    print_comparison(baseline_results, residual_results)

    # -- Verdict ---------------------------------------------------------------
    verdict(residual_results)


if __name__ == "__main__":
    main()
