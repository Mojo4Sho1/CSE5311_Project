# CSE5311_Project - Sorting Algorithm Benchmark

Compare classic sorting algorithms under controlled conditions to answer:

- **How do running times change with data size?**
- **How do speeds compare across algorithms at different sizes?**
- **Can simple, principled tweaks improve running time?**
- **Which algorithm is better under which conditions?**

This repo emphasizes clean experimental design, reproducibility, and presentation-ready figures.

---

## Quick Start

### Environment

Create from file (recommended):

```bash
conda env create -f environment.yml
conda activate sortbench
```

Or install directly:

```bash
conda create -n sortbench python=3.11 -y
conda activate sortbench
conda install -y numpy pyyaml pandas matplotlib jupyter pytest hypothesis psutil
conda install -y tqdm rich
# optional:
# conda install -y numba        # or cython
# conda install -y black ruff pre-commit mypy orjson cpuinfo pytest-benchmark
```

### Repository Layout

```
sorting-bench/
├─ README.md
├─ environment.yml
├─ .gitignore
├─ src/
│  └─ sortbench/
│     ├─ __init__.py
│     ├─ algorithms/
│     │  ├─ __init__.py
│     │  ├─ insertion.py
│     │  ├─ quicksort.py
│     │  ├─ mergesort.py
│     │  ├─ heapsort.py
│     │  ├─ radix.py
│     │  └─ builtin_timsort.py
│     ├─ datasets/
│     │  ├─ __init__.py
│     │  ├─ generators.py
│     │  └─ transforms.py
│     ├─ bench/
│     │  ├─ __init__.py
│     │  ├─ runner.py
│     │  ├─ measure.py
│     │  └─ metadata.py
│     ├─ validate/
│     │  ├─ __init__.py
│     │  ├─ oracle.py
│     │  └─ properties.py
│     └─ utils/
│        ├─ __init__.py
│        ├─ types.py
│        └─ io.py
├─ experiments/
│  ├─ configs/
│  │  └─ 01_random_scaling.yaml
│  └─ runs/                 # auto-created by runner
├─ notebooks/
│  ├─ 01_random_scaling.ipynb
│  └─ 02_sensitivity.ipynb
├─ tests/
│  ├─ test_correctness.py
│  └─ test_stability.py
└─ slides/
   └─ talk-outline.md
```

---

## Interfaces (Contracts)

These interfaces are fixed so implementations can be swapped without plumbing changes.

### 1) Algorithm API (uniform across all sorters)

```python
def sort(a: list[int], *, config: dict | None = None) -> list[int]
```

- **Pure function**: do **not** mutate the input; return a new sorted list.
- `config` enables controlled variations (“discovery” toggles), e.g.:
  - quicksort: `{"pivot":"median3"|"random"|"first", "insertion_cutoff":24, "three_way":true}`
  - mergesort: `{"detect_runs":false, "small_run_insertion":24}`
  - heapsort: `{}`
  - insertion: `{}`
  - radix: `{"digits_per_pass_bits":8, "signed":false}`

Each algorithm module exports this `sort` function and documents its config keys.

### 2) Dataset Generator

```python
def make_dataset(n: int, spec: dict, rng) -> list[int]
```

Supported `spec` forms (examples):

```yaml
# Random integers (default)
dist: random
params:
  range: [0, 4294967295]

# Nearly-sorted: start from sorted, then perform a fraction of random swaps
dist: nearly_sorted
params:
  swap_frac: 0.05   # 0.0 → already sorted; 1.0 → fully randomized

# Few-uniques: draw from a small set of values (heavy duplicates)
dist: few_uniques
params:
  k: 100

# Small-range integers (e.g., byte values)
dist: small_range
params:
  max_val: 255

# Reversed order
dist: reversed
params: {}
```

### 3) Benchmark Record (one JSON object per trial)

Each timed trial is logged as a single JSON line:

```json
{
  "algo": "quicksort",
  "n": 100000,
  "dataset": {"dist":"random","params":{"range":[0,4294967295]}},
  "trial": 3,
  "time_ns": 123456789,
  "comparisons": 1456789,
  "moves": 987654,
  "config": {"pivot":"median3","insertion_cutoff":24},
  "python": "3.11.9",
  "machine": {"cpu":"Intel(...)", "cores_logical":16, "ram_gb":64},
  "git_commit": "abcdef1",
  "notes": null
}
```

- `comparisons`/`moves` are optional (only if instrumentation is enabled).
- A run also emits a compact `summary.csv` (medians/IQR) and `meta.json`.

---

## Experiment Configuration Schema (YAML)

Runner reads a single YAML file to define an experiment sweep.

```yaml
# experiments/configs/01_random_scaling.yaml
experiment_name: "01_random_scaling"
output_dir: "experiments/runs"
seed: 42
repeats: 7             # independent repeats per (algo, n)
warmup: true           # one untimed call before timing
disable_gc: true       # disable GC during timing blocks
timeout_seconds: 60    # per-trial timeout; skip if exceeded

dataset:
  dist: "random"       # random | nearly_sorted | few_uniques | small_range | reversed
  params:
    range: [0, 4294967295]

sizes: [1000, 2000, 5000, 10000, 20000, 50000, 100000]

algorithms:
  - name: "builtin_timsort"
    config: {}

# Later, add more:
# - name: "quicksort"
#   config: { pivot: "median3", insertion_cutoff: 24, three_way: false }
# - name: "mergesort"
#   config: { detect_runs: false, small_run_insertion: 0 }
# - name: "heapsort"
#   config: {}
# - name: "insertion"
#   config: {}
# - name: "radix"
#   config: { digits_per_pass_bits: 8 }
```

---

## Timing Methodology (fairness & repeatability)

- Clock: `time.perf_counter_ns()`; report **median** over `repeats` and optionally IQR.  
- GC: disabled during the timed block if `disable_gc: true`.  
- Warmup: one untimed call per (algo, n) if `warmup: true`.  
- Input reuse: for each (n, trial), generate **one** dataset and provide **the same data** to every algorithm (copy per call).  
- Stop rule: if a trial exceeds `timeout_seconds`, mark as timeout and skip larger `n` for that algorithm in this run.  
- Metadata: record Python version, CPU model/cores, RAM, and current git commit in `meta.json`.

---

## Validation & Correctness

- **Oracle check**: `validate/oracle.py` compares each algorithm’s output to Python’s `sorted(a)`.  
- **Properties**:
  - Output is nondecreasing.  
  - Output is a permutation of input (same multiset).  
  - (Optional) Stability tests for algorithms that claim stability (mergesort, radix in certain modes).
- **Tests** (pytest): randomized arrays and edge cases (empty, length 1, all equal, many duplicates, reversed).

---

## Running an Experiment

Once `runner.py` exists:

```bash
python -m sortbench.bench.runner experiments/configs/01_random_scaling.yaml
```

Artifacts appear under:

```
experiments/runs/<timestamp>_01_random_scaling/
  ├─ results.jsonl     # one line per trial
  ├─ summary.csv       # grouped medians/IQR by (algo, n)
  └─ meta.json         # environment & repo details
```

---

## Analysis Notebooks (figures for slides)

Work in `notebooks/`:

- `01_random_scaling.ipynb`: load `summary.csv` → plot time vs `n` (per algorithm).  
- `02_sensitivity.ipynb`: plots for nearly-sorted (`swap_frac`) and duplicates/value-range sweeps.

Recommended figures:
- Scaling curves (time vs `n`) on random data.  
- Nearly-sorted crossover plot (different `swap_frac`).  
- Duplicates / value-range figure (few-uniques `k`, small-range `max_val`).  
- Ablation bars for “discovery” toggles (e.g., quicksort pivot & insertion cutoff).

---

## “Discovery” Levers (simple, defensible tweaks)

- **Quicksort**: pivot selection (`first`/`random`/`median3`), **insertion-sort cutoff** for small partitions, optional **3-way partition** for heavy duplicates.  
- **Mergesort**: detect natural runs; optional small-run insertion.  
- **Heapsort**: Floyd heapify vs sift-up.  
- **Radix**: LSD vs MSD; **digits per pass** (8 vs 16 bits); handling signed ints.

Report each tweak’s median speedup and the conditions where it helps.

---

## Slide Outline

1. Problem & questions (assignment bullets)  
2. Algorithms at a glance (asymptotics; one concise slide)  
3. Experimental design (datasets, sizes, repeats, timing hygiene, machine)  
4. Results: scaling with size (main plot)  
5. Results: nearly-sorted sensitivity (mini multipanel)  
6. Results: duplicates/value-range (mini multipanel)  
7. Discovery ablation(s) with clear % gains  
8. “Which is better when?” summary table  
9. Limitations & next steps

---

## Roadmap

- **Step 1 (done):** Fix interfaces and README; scaffold files.  
- **Step 2:** Minimal end-to-end with `builtin_timsort` only (random dataset) → produce first `results.jsonl`, `summary.csv`, `meta.json`.  
- **Step 3:** Implement quicksort (median-of-three + insertion cutoff) + correctness tests; rerun `01_random_scaling`.  
- **Step 4:** Add mergesort, heapsort, insertion, radix; add sensitivity configs.

---

## License

TBD (e.g., MIT). Add a `LICENSE` file when decided.

---

## Acknowledgments

Designed for small, transparent experiments with a config-driven runner and strict interfaces so that algorithms and datasets can be swapped without refactoring.
