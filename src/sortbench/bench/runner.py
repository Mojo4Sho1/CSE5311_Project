"""
Experiment runner: orchestrates a full benchmarking sweep from a YAML config.

Usage (from repo root):
    python -m sortbench.bench.runner experiments/configs/01_random_scaling.yaml

Outputs in a new run directory:
    - config_resolved.yaml    # the config we actually used
    - meta.json               # environment info (python, numpy, cpu/ram, git commit)
    - results.jsonl           # one JSON line per successful timing sample
    - summary.csv             # median + IQR per (algo, n)
    - (console) rich/tqdm summaries

Design notes:
- For each size n, we generate ONE dataset and give the same input to every algorithm.
- Harness handles warmup/GC; we keep timing clean.
- On timeout/error for an algorithm at size n, we skip larger sizes for that algo.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
import yaml

# QoL (optional, but present in our environment)
try:
    from rich.console import Console
    from rich.table import Table
    _RICH = True
    _console = Console()
except Exception:  # pragma: no cover
    _RICH = False
    _console = None  # type: ignore

try:
    from tqdm import tqdm
    _TQDM = True
except Exception:  # pragma: no cover
    _TQDM = False

# Project imports
from sortbench.datasets import make_dataset
from sortbench.bench.measure import time_sort_call


# ------------------------- data structures ------------------------- #

@dataclass(frozen=True)
class AlgoSpec:
    name: str
    sort_fn: Any
    config: Dict[str, Any]


# ------------------------- helpers: IO & meta ------------------------- #

def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(obj: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _append_jsonl(obj: Dict[str, Any], path: Path) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))
        f.write("\n")


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_run_dir(base_dir: Path, experiment_name: str) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dir = base_dir / f"{_timestamp()}_{experiment_name}"
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def _git_commit_short() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _gather_meta() -> Dict[str, Any]:
    import platform
    meta = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "psutil": psutil.__version__,
        "git_commit": _git_commit_short(),
        "machine": {
            "cpu": platform.processor() or platform.machine(),
            "cores_logical": psutil.cpu_count(logical=True),
            "cores_physical": psutil.cpu_count(logical=False),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "platform": platform.platform(),
        },
        "start_time": _dt.datetime.now().isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "cwd": str(Path.cwd()),
    }
    return meta


def _resolve_algorithms(cfg_algos: List[Dict[str, Any]]) -> List[AlgoSpec]:
    specs: List[AlgoSpec] = []
    seen = set()
    for entry in cfg_algos:
        name = entry.get("name", None)
        if not name or not isinstance(name, str):
            raise ValueError("Each algorithm must have a string 'name' field")
        if name in seen:
            raise ValueError(f"Duplicate algorithm name in config: {name}")
        seen.add(name)

        try:
            mod = importlib.import_module(f"sortbench.algorithms.{name}")
        except Exception as e:
            raise ImportError(f"Could not import algorithm module 'sortbench.algorithms.{name}': {e!r}") from e

        if not hasattr(mod, "sort"):
            raise AttributeError(f"Algorithm module '{name}' must define a callable `sort(a, *, config=None)`")

        config = entry.get("config", {})
        if config is None:
            config = {}
        elif not isinstance(config, dict):
            raise ValueError(f"Algorithm '{name}': 'config' must be a dict if provided")

        specs.append(AlgoSpec(name=name, sort_fn=getattr(mod, "sort"), config=config))
    return specs


def _iqr_ns(group: pd.DataFrame) -> int:
    q1 = group["time_ns"].quantile(0.25)
    q3 = group["time_ns"].quantile(0.75)
    return int(q3 - q1)


def _aggregate_summary(jsonl_path: Path) -> pd.DataFrame:
    if not jsonl_path.exists():
        # Empty result
        return pd.DataFrame(columns=["algo", "n", "samples_ok", "median_ns", "iqr_ns", "min_ns", "max_ns"])
    df = pd.read_json(jsonl_path, lines=True)
    # Filter only successful samples (lines with time_ns present)
    df = df[df["time_ns"].notna()]
    if df.empty:
        return pd.DataFrame(columns=["algo", "n", "samples_ok", "median_ns", "iqr_ns", "min_ns", "max_ns"])

    agg = (
        df.groupby(["algo", "n"], as_index=False)
        .agg(
            samples_ok=("time_ns", "count"),
            median_ns=("time_ns", "median"),
            min_ns=("time_ns", "min"),
            max_ns=("time_ns", "max"),
        )
    )
    # Add IQR
    iqr_vals = (
        df.groupby(["algo", "n"])
        .apply(_iqr_ns)
        .rename("iqr_ns")
        .reset_index()
    )
    out = agg.merge(iqr_vals, on=["algo", "n"], how="left")
    # Convert medians to int for clean CSV (they may be float from pandas)
    out["median_ns"] = out["median_ns"].astype("int64")
    out[["min_ns", "max_ns", "iqr_ns"]] = out[["min_ns", "max_ns", "iqr_ns"]].astype("int64")
    return out.sort_values(["algo", "n"], ignore_index=True)


def _print_rich_summary(summary: pd.DataFrame, sizes: List[int]) -> None:
    if not _RICH:
        # Fallback: plain print a few lines
        print("\n=== Summary (median ns) ===")
        if summary.empty:
            print("(no samples)")
            return
        for algo in summary["algo"].unique():
            s = summary[summary["algo"] == algo]
            last = s.sort_values("n").tail(1)
            med = int(last["median_ns"].values[0]) if not last.empty else None
            print(f"{algo:20s}  last_n={int(last['n'].values[0]) if not last.empty else '—'}  median_ns={med}")
        return

    # Rich table with first/middle/last n (if present)
    table = Table(title="Benchmark Summary (median ± IQR in ms)")
    table.add_column("Algorithm", style="bold")
    picks: List[Tuple[str, Optional[int]]] = []
    if sizes:
        first = sizes[0]
        mid = sizes[len(sizes)//2]
        last = sizes[-1]
        picks = [("n=" + str(first), first), ("n=" + str(mid), mid), ("n=" + str(last), last)]
        for hdr, _ in picks:
            table.add_column(hdr, justify="right")
    else:
        table.add_column("median (all)", justify="right")

    def _format_cell(median_ns: Optional[int], iqr_ns: Optional[int]) -> str:
        if median_ns is None:
            return "—"
        # convert to ms for display
        median_ms = median_ns / 1e6
        if iqr_ns is None:
            return f"{median_ms:.2f}"
        iqr_ms = iqr_ns / 1e6
        return f"{median_ms:.2f} ± {iqr_ms:.2f}"

    for algo in summary["algo"].unique():
        row = [f"[bold]{algo}[/]"]
        if picks:
            for _, npick in picks:
                s = summary[(summary["algo"] == algo) & (summary["n"] == npick)]
                if s.empty:
                    row.append("—")
                else:
                    row.append(_format_cell(int(s["median_ns"].values[0]), int(s["iqr_ns"].values[0])))
        else:
            s = summary[summary["algo"] == algo]
            if s.empty:
                row.append("—")
            else:
                med = int(s["median_ns"].median())
                iqr = int(s["iqr_ns"].median())
                row.append(_format_cell(med, iqr))
        table.add_row(*row)
    _console.print()
    _console.print(table)
    _console.print()


# ------------------------- core runner ------------------------- #

def run_experiment(config_path: Path) -> Path:
    cfg = _load_yaml(config_path)

    # Required keys & basic validation
    required = ["experiment_name", "output_dir", "seed", "repeats", "warmup", "disable_gc", "timeout_seconds", "dataset", "sizes", "algorithms"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    experiment_name: str = str(cfg["experiment_name"])
    output_dir = Path(cfg["output_dir"])
    sizes: List[int] = list(cfg["sizes"])
    repeats: int = int(cfg["repeats"])
    warmup: bool = bool(cfg["warmup"])
    disable_gc: bool = bool(cfg["disable_gc"])
    timeout_seconds: float = float(cfg["timeout_seconds"])
    dataset_spec: Dict[str, Any] = dict(cfg["dataset"])
    algos_cfg: List[Dict[str, Any]] = list(cfg["algorithms"])

    if not sizes:
        raise ValueError("Config 'sizes' must be a non-empty list of positive integers")

    run_dir = _ensure_run_dir(output_dir, experiment_name)
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.csv"
    meta_path = run_dir / "meta.json"
    cfg_resolved_path = run_dir / "config_resolved.yaml"

    # Persist resolved config early
    _write_yaml(cfg, cfg_resolved_path)

    # Meta info
    meta = _gather_meta()
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Algorithms
    algos: List[AlgoSpec] = _resolve_algorithms(algos_cfg)

    # Seeded RNG
    rng = np.random.default_rng(int(cfg["seed"]))

    # Per-algorithm skip flags (set on timeout/error)
    per_algo_skip = {a.name: False for a in algos}

    if _RICH:
        _console.print(f"[bold green]Run directory:[/bold green] {run_dir}")
        _console.print(f"[bold]Experiment:[/bold] {experiment_name}")
        _console.print(f"[bold]Algorithms:[/bold] {', '.join(a.name for a in algos)}")
        _console.print()

    size_iter = sizes
    if _TQDM:
        size_iter = tqdm(sizes, desc="Sizes", unit="n")

    for n in size_iter:
        # Generate the single dataset for this size
        base_a = make_dataset(int(n), dataset_spec, rng)

        algo_iter = algos
        if _TQDM:
            # Avoid nested noisy bars; a light inner status line instead
            pass  # keep single bar for clarity

        for a_spec in algo_iter:
            if per_algo_skip[a_spec.name]:
                continue

            res = time_sort_call(
                algo_name=a_spec.name,
                algo_fn=a_spec.sort_fn,
                a=base_a,
                config=a_spec.config,
                repeats=repeats,
                warmup=warmup,
                disable_gc=disable_gc,
                timeout_seconds=timeout_seconds,
                defensive_copy=True,  # safer during development
            )

            # Emit sample lines
            for trial_idx, t_ns in enumerate(res["samples_ns"]):
                _append_jsonl(
                    {
                        "algo": a_spec.name,
                        "n": int(n),
                        "dataset": dataset_spec,
                        "trial": int(trial_idx),
                        "time_ns": int(t_ns),
                        "config": a_spec.config,
                    },
                    results_path,
                )

            # Handle timeout/error
            status = res.get("status", "ok")
            if status == "timeout":
                per_algo_skip[a_spec.name] = True
                _append_jsonl(
                    {
                        "algo": a_spec.name,
                        "n": int(n),
                        "status": "timeout",
                        "timed_out_on_repeat": res.get("timed_out_on_repeat"),
                        "config": a_spec.config,
                    },
                    results_path,
                )
            elif status == "error":
                per_algo_skip[a_spec.name] = True
                _append_jsonl(
                    {
                        "algo": a_spec.name,
                        "n": int(n),
                        "status": "error",
                        "error": res.get("error"),
                        "config": a_spec.config,
                    },
                    results_path,
                )

    # Aggregate → summary.csv
    summary_df = _aggregate_summary(results_path)
    summary_df.to_csv(summary_path, index=False)

    # Console summary
    _print_rich_summary(summary_df, sizes)

    if _RICH:
        _console.print(f"[bold green]Done.[/bold green] Wrote:")
        _console.print(f" - {results_path}")
        _console.print(f" - {summary_path}")
        _console.print(f" - {meta_path}")
        _console.print(f" - {cfg_resolved_path}")

    return run_dir


# ------------------------- CLI ------------------------- #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a sorting benchmark experiment from a YAML config.")
    p.add_argument("config", type=str, help="Path to YAML experiment config")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")
    try:
        run_experiment(config_path)
    except Exception as e:
        # Print a friendly message and exit non-zero
        if _RICH:
            _console.print(f"[bold red]Runner failed:[/bold red] {e!r}")
        else:
            print(f"Runner failed: {e!r}")
        raise


if __name__ == "__main__":
    main()
