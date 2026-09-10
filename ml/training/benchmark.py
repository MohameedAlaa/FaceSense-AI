"""
FaceSense AI - CPU Performance Benchmark
Measures DataLoader creation time, batch fetch time, forward pass time,
and backward pass time. Optionally benchmarks torch.compile vs eager.

Usage:
    python ml/training/benchmark.py
    python ml/training/benchmark.py --num-batches 20
"""

import argparse
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from ml.data.dataloader import build_dataloaders, load_yaml_config
from ml.models.baseline_cnn import build_model_from_config


def time_batches(loader, n: int) -> float:
    """Returns average batch fetch time in seconds over n batches."""
    it = iter(loader)
    next(it)  # warm-up (includes worker init on first call)
    times = []
    for _ in range(n):
        try:
            t = time.perf_counter()
            next(it)
            times.append(time.perf_counter() - t)
        except StopIteration:
            break
    return sum(times) / len(times) if times else 0.0


def time_forward_backward(model, images, targets, n: int = 5) -> tuple:
    """
    Returns (avg_fwd_s, avg_bwd_s) over n passes.
    First pass is always a warm-up (may trigger compilation on compiled models).
    """
    criterion = torch.nn.CrossEntropyLoss()
    model.train()

    # Warm-up pass (not timed)
    try:
        out = model(images)
        loss = criterion(out, targets)
        loss.backward()
    except Exception as e:
        return None, None, str(e)

    fwd_times, bwd_times = [], []
    for _ in range(n):
        t = time.perf_counter()
        out = model(images)
        fwd_times.append(time.perf_counter() - t)

        loss = criterion(out, targets)
        t = time.perf_counter()
        loss.backward()
        bwd_times.append(time.perf_counter() - t)

    avg_fwd = sum(fwd_times) / len(fwd_times)
    avg_bwd = sum(bwd_times) / len(bwd_times)
    return avg_fwd, avg_bwd, None


def benchmark(config_path: str = "configs/config.yaml", num_batches: int = 20):
    print("=" * 65)
    print("FaceSense AI - CPU Benchmark")
    print("=" * 65)

    config = load_yaml_config(config_path)
    dl_cfg = config.get("dataloader", {})
    cpu_cfg = config.get("cpu", {})

    # Apply thread setting
    num_threads = int(cpu_cfg.get("num_threads", torch.get_num_threads()))
    torch.set_num_threads(num_threads)
    use_compile = bool(cpu_cfg.get("use_compile", False))

    print(f"num_workers       : {dl_cfg.get('num_workers', 0)}")
    print(f"pin_memory        : {dl_cfg.get('pin_memory', False)}")
    print(f"persistent_workers: {dl_cfg.get('persistent_workers', False)}")
    print(f"prefetch_factor   : {dl_cfg.get('prefetch_factor', 'N/A')}")
    print(f"batch_size        : {dl_cfg.get('batch_size', 64)}")
    print(f"num_threads       : {num_threads}")
    print(f"torch.compile     : {use_compile}")
    print("-" * 65)

    # ── 1. DataLoader creation ───────────────────────────────────────
    t0 = time.perf_counter()
    train_loader, _, _, metadata = build_dataloaders(config)
    dl_create_time = time.perf_counter() - t0
    print(f"[1] DataLoader creation time : {dl_create_time:.3f}s")
    print(f"    Train samples: {metadata['train_count']:,}")

    # ── 2. Batch fetch ───────────────────────────────────────────────
    avg_batch = time_batches(train_loader, num_batches)
    print(f"[2] Avg batch fetch time ({num_batches} batches): {avg_batch * 1000:.1f}ms")

    # ── 3. Eager model forward + backward ────────────────────────────
    model_eager = build_model_from_config(config)
    images, targets = next(iter(train_loader))
    fwd_eager, bwd_eager, err = time_forward_backward(model_eager, images, targets, n=5)
    compute_eager = fwd_eager + bwd_eager
    print(f"[3] EAGER  Forward: {fwd_eager*1000:.1f}ms  Backward: {bwd_eager*1000:.1f}ms  "
          f"Total: {compute_eager*1000:.1f}ms")

    # ── 4. Compiled model forward + backward ─────────────────────────
    best_compute = compute_eager
    if use_compile and hasattr(torch, "compile"):
        print(f"[4] Compiling model (first call triggers AOT compilation)...")
        try:
            model_compiled = torch.compile(build_model_from_config(config))
            fwd_cmp, bwd_cmp, err = time_forward_backward(model_compiled, images, targets, n=5)
            if err:
                print(f"    [!] torch.compile failed: {err}")
                print(f"        Note: MSVC cl.exe required on Windows for inductor backend.")
            else:
                compute_cmp = fwd_cmp + bwd_cmp
                speedup = compute_eager / compute_cmp if compute_cmp > 0 else 1.0
                print(f"    COMPILED Forward: {fwd_cmp*1000:.1f}ms  Backward: {bwd_cmp*1000:.1f}ms  "
                      f"Total: {compute_cmp*1000:.1f}ms  [{speedup:.2f}x speedup]")
                best_compute = compute_cmp
        except Exception as e:
            print(f"    [!] torch.compile unavailable: {type(e).__name__}")
            print(f"        To enable: install Visual Studio Build Tools (MSVC cl.exe)")
    else:
        if not use_compile:
            print("[4] torch.compile: disabled in config (use_compile: false)")
        else:
            print("[4] torch.compile: not available in this PyTorch build")

    # ── 5. Projected epoch time ──────────────────────────────────────
    batches_per_epoch = len(train_loader)
    # With persistent workers, fetch overlaps with compute; bottleneck is the max
    bottleneck = max(avg_batch, best_compute)
    bottleneck_label = "COMPUTE" if best_compute >= avg_batch else "DATA-LOADING"
    projected_s = bottleneck * batches_per_epoch

    print("-" * 65)
    print(f"[5] Batches per epoch       : {batches_per_epoch}")
    print(f"    Fetch per batch         : {avg_batch * 1000:.1f}ms")
    print(f"    Best compute per batch  : {best_compute * 1000:.1f}ms")
    print(f"    Bottleneck              : {bottleneck_label}")
    print(f"    ~Projected epoch time   : {projected_s:.0f}s  ({projected_s/60:.1f} min)")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FaceSense AI CPU Benchmark")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--num-batches", type=int, default=20, help="Number of batches to time")
    args = parser.parse_args()
    benchmark(args.config, args.num_batches)
