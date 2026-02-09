#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild summary.csv from existing result CSVs.")
    parser.add_argument("--out-dir", type=str, default="out/go2w_reg_ablation_5x5")
    return parser.parse_args()


def parse_meta(path: Path) -> dict:
    meta = {
        "radius": None,
        "speed": None,
        "weight": None,
        "yaw0": None,
        "completed": None,
        "lin_err_mean": None,
        "yaw_err_mean": None,
        "final_pos_err": None,
        "final_yaw_err": None,
    }
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#"):
                break
            if ":" not in line:
                continue
            key, val = line[1:].split(":", 1)
            key = key.strip()
            val = val.strip()
            if key in {"radius", "speed", "weight", "yaw0", "lin_err_mean", "yaw_err_mean", "final_pos_err", "final_yaw_err"}:
                try:
                    meta[key] = float(val)
                except ValueError:
                    meta[key] = None
            elif key == "completed":
                try:
                    meta[key] = int(val)
                except ValueError:
                    meta[key] = None
    return meta


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    csv_files = sorted([p for p in out_dir.glob("*.csv") if p.name != "summary.csv"])
    summary_path = out_dir / "summary.csv"

    with summary_path.open("w", encoding="utf-8", newline="") as fsum:
        writer = csv.writer(fsum)
        writer.writerow([
            "radius",
            "weight",
            "vx_cmd",
            "wz_cmd",
            "yaw0",
            "completed",
            "lin_err_mean",
            "yaw_err_mean",
            "final_pos_err",
            "final_yaw_err",
            "file",
        ])
        for p in csv_files:
            meta = parse_meta(p)
            if meta["radius"] is None or meta["weight"] is None or meta["speed"] is None:
                continue
            radius = meta["radius"]
            weight = meta["weight"]
            vx_cmd = meta["speed"]
            wz_cmd = vx_cmd / radius if radius and abs(radius) > 1e-9 else 0.0
            writer.writerow([
                radius,
                weight,
                vx_cmd,
                wz_cmd,
                meta["yaw0"],
                meta["completed"],
                meta["lin_err_mean"],
                meta["yaw_err_mean"],
                meta["final_pos_err"],
                meta["final_yaw_err"],
                p.as_posix(),
            ])
    print(f"re-summarized: {summary_path}")


if __name__ == "__main__":
    main()
