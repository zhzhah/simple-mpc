#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablation runner for go2w_kinodynamics_reg.py")
    parser.add_argument("--r-count", type=int, default=20)
    parser.add_argument("--w-count", type=int, default=21)
    parser.add_argument("--speed-max", type=float, default=0.4)
    parser.add_argument("--wz-max", type=float, default=0.5)
    parser.add_argument("--init-yaws", type=str, default="0,1.5708,3.1416,-1.5708")
    parser.add_argument("--out-root", type=str, default="out/go2w_reg_ablation")
    parser.add_argument("--batch-steps", type=int, default=2000)
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args()


def compute_cmd(vx_max: float, radius: float, wz_max: float) -> tuple[float, float]:
    vx = vx_max
    wz = vx / radius
    if abs(wz) > wz_max:
        wz = math.copysign(wz_max, wz)
        vx = abs(wz) * radius
    return vx, wz


def ensure_dir(root: Path) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = root / ts
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=False)
        return out_dir
    suffix = 1
    while True:
        candidate = root / f"{ts}_{suffix}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        suffix += 1


def run_and_check(cmd: list[str], out_file: Path) -> bool:
    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        print("[warn] run failed:", ret.returncode)
        return False
    if not out_file.exists():
        print("[warn] output missing:", out_file)
        return False
    if out_file.stat().st_size <= 0:
        print("[warn] output empty:", out_file)
        return False
    return True


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.out_root))
    r_min = 0.1
    r_max_target = 2.0
    w_min = 0.0
    w_max_target = 400.0
    if args.r_count <= 0:
        raise ValueError("r-count must be > 0")
    if args.w_count <= 0:
        raise ValueError("w-count must be > 0")
    if args.r_count == 1:
        r_step = 0.1
    else:
        r_step_raw = (r_max_target - r_min) / (args.r_count - 1)
        r_step = max(0.1, round(r_step_raw, 1))
    radii = [round(r_min + r_step * i, 1) for i in range(args.r_count)]
    if args.w_count == 1:
        w_step = 20.0
    else:
        w_step_raw = w_max_target / (args.w_count - 1)
        w_step = max(20.0, round(w_step_raw / 20.0) * 20.0)
    weights = [round(w_min + w_step * i, 1) for i in range(args.w_count)]
    yaws = [float(x) for x in args.init_yaws.split(",") if x.strip() != ""]

    summary = out_dir / "summary.csv"
    with summary.open("w", encoding="utf-8") as fsum:
        fsum.write(
            "radius,weight,vx_cmd,wz_cmd,yaw0,completed,lin_err_mean,yaw_err_mean,final_pos_err,final_yaw_err,file\n"
        )
        fsum.flush()
        last_cmd: list[str] | None = None
        last_out_file: Path | None = None
        for r in radii:
            for w in weights:
                vx_cmd, wz_cmd = compute_cmd(args.speed_max, r, args.wz_max)
                for yaw0 in yaws:
                    if last_out_file is not None:
                        missing_prev = (not last_out_file.exists()) or (last_out_file.stat().st_size <= 0)
                        if missing_prev:
                            for attempt in range(3):
                                print("[retry]", " ".join(last_cmd or []))
                                if last_cmd is not None and run_and_check(last_cmd, last_out_file):
                                    break
                            if not last_out_file.exists() or last_out_file.stat().st_size <= 0:
                                print("[warn] previous output still missing:", last_out_file)
                    out_file = out_dir / f"r{r:.1f}_w{w:.0f}_yaw{yaw0:.3f}.csv"
                    cmd = [
                        sys.executable,
                        os.path.join(os.path.dirname(__file__), "go2w_kinodynamics_reg.py"),
                        "--batch",
                        "--radius",
                        str(r),
                        "--speed",
                        str(vx_cmd),
                        "--weight",
                        str(w),
                        "--yaw0",
                        str(yaw0),
                        "--batch-steps",
                        str(args.batch_steps),
                        "--out",
                        str(out_file),
                    ]
                    if args.gui:
                        cmd.append("--gui")
                    print("[run]", " ".join(cmd))
                    if not run_and_check(cmd, out_file):
                        last_cmd = cmd
                        last_out_file = out_file
                        continue
                    last_cmd = cmd
                    last_out_file = out_file
                    # parse output file for summary
                    completed = 0
                    lin_err_mean = float("nan")
                    yaw_err_mean = float("nan")
                    final_pos_err = float("nan")
                    final_yaw_err = float("nan")
                    with out_file.open("r", encoding="utf-8") as f:
                        for line in f:
                            if line.startswith("# completed:"):
                                completed = int(line.split(":")[1].strip())
                            elif line.startswith("# lin_err_mean:"):
                                lin_err_mean = float(line.split(":")[1].strip())
                            elif line.startswith("# yaw_err_mean:"):
                                yaw_err_mean = float(line.split(":")[1].strip())
                            elif line.startswith("# final_pos_err:"):
                                final_pos_err = float(line.split(":")[1].strip())
                            elif line.startswith("# final_yaw_err:"):
                                final_yaw_err = float(line.split(":")[1].strip())
                            if not line.startswith("#"):
                                break
                    fsum.write(
                        f"{r},{w},{vx_cmd},{wz_cmd},{yaw0},{completed},{lin_err_mean},{yaw_err_mean},{final_pos_err},{final_yaw_err},{out_file}\n"
                    )
                    fsum.flush()


if __name__ == "__main__":
    main()
