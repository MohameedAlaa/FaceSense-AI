"""
FaceSense AI - Dataset Inspection Script
Fast & robust dataset statistics calculator, dimension inspector, and validator.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
from PIL import Image

def inspect_dataset(data_dir: str | Path = "Data(FER2013)"):
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Dataset directory '{data_path}' does not exist.")
        return

    print("=" * 65)
    print(f"FaceSense AI - Dataset Inspection: {data_path.resolve()}")
    print("=" * 65)

    splits = [d.name for d in data_path.iterdir() if d.is_dir()]
    if not splits:
        print("No subdirectories/splits found in dataset path.")
        return

    stats = {
        "total_samples": 0,
        "splits": {},
        "global_class_counts": defaultdict(int),
        "dimensions": set(),
        "color_modes": set(),
        "invalid_files": [],
    }

    for split in sorted(splits):
        split_path = data_path / split
        split_classes = sorted([d.name for d in split_path.iterdir() if d.is_dir()])
        split_counts = {}
        split_total = 0

        for cls_name in split_classes:
            cls_folder = split_path / cls_name
            files = [f for f in cls_folder.iterdir() if f.is_file()]
            valid_count = 0
            
            for file_path in files:
                try:
                    # Open without redundant verify for faster single-pass validation
                    with Image.open(file_path) as img:
                        stats["dimensions"].add(img.size)
                        stats["color_modes"].add(img.mode)
                    valid_count += 1
                except Exception as e:
                    stats["invalid_files"].append((str(file_path), str(e)))

            split_counts[cls_name] = valid_count
            split_total += valid_count
            stats["global_class_counts"][cls_name] += valid_count

        stats["splits"][split] = {
            "total": split_total,
            "classes": split_counts
        }
        stats["total_samples"] += split_total

    print(f"\nTotal Samples: {stats['total_samples']:,}")
    print(f"Image Dimensions (W x H): {list(stats['dimensions'])}")
    print(f"Color Modes: {list(stats['color_modes'])}")
    print(f"Corrupted / Invalid Files: {len(stats['invalid_files'])}")

    print("\n" + "-" * 65)
    print(f"{'Split':<8} | {'Class':<12} | {'Count':<8} | {'Percentage of Split':<20}")
    print("-" * 65)
    for split, data in stats["splits"].items():
        for cls_name, count in data["classes"].items():
            pct = (count / data["total"] * 100) if data["total"] > 0 else 0
            print(f"{split:<8} | {cls_name:<12} | {count:<8} | {pct:>6.2f}%")
        print(f"{split:<8} | {'TOTAL':<12} | {data['total']:<8} | 100.00%")
        print("-" * 65)

    print("\nOverall Class Distribution Across Dataset:")
    print("-" * 50)
    for cls_name, count in sorted(stats["global_class_counts"].items()):
        pct = (count / stats["total_samples"] * 100) if stats["total_samples"] > 0 else 0
        print(f"  - {cls_name:<12}: {count:>6} ({pct:>6.2f}%)")
    print("-" * 50)
    print(f"  - {'TOTAL':<12}: {stats['total_samples']:>6} (100.00%)")
    print("=" * 65)

    return stats

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "Data(FER2013)"
    inspect_dataset(target)
