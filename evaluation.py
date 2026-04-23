# Metrics covered:
#   1. Computational Performance  — runtime, peak memory, motif count
#   2. Scalability Benchmark      — run each algo at growing network sizes
#   3. Embedding Similarity       — cosine similarity matrix across datasets
#   4. Information Loss (Hybrid)  — fixed vs adaptive binning comparison
#   5. Hyperparameter Sensitivity — sweep Δt (k) and sample_rate
#   6. Null-Model Statistics      — alpha, beta, gamma thresholds on motif frequencies

import time
import tracemalloc
import random
import math
import importlib
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Import your four algorithm modules ───────────────────────────────────────
import etn
import improved_etn
import hetn
import hybrid

ALGOS = {
    "ETN":          etn,
    "Improved ETN": improved_etn,
    "HETN":         hetn,
    "Hybrid":       hybrid,
}

# DATASET LOADING

def load_dataset(file_path):
    edges = []
    timestamps = []

    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue

            u, v, t = map(int, line.strip().split()[:3])
            edges.append((u, v, t))
            timestamps.append(t)

    # STEP 1: normalize timestamps to start from 0
    t_min = min(timestamps)
    edges = [(u, v, t - t_min) for (u, v, t) in edges]

    # STEP 2: compress timestamps (VERY IMPORTANT)
    # map large gaps → small consecutive integers
    unique_times = sorted(set(t for _, _, t in edges))
    time_map = {t: i for i, t in enumerate(unique_times)}

    edges = [(u, v, time_map[t]) for (u, v, t) in edges]

    return edges

DATASET_FILES = { 
    "CollegeMsg": "CollegeMsg.txt",
    "Email": "email-Eu-core-temporal.txt",
    "SuperUser": "sx-superuser.txt"
}

DATASETS = {
    name: load_dataset(path)
    for name, path in DATASET_FILES.items()
}

# HELPER UTILITIES

def run_with_metrics(algo_module, edges, **kwargs):
    """
    Run algo_module.compute_embedding(edges, **kwargs).
    Returns (embedding_dict, elapsed_seconds, peak_memory_MB).
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    embedding = algo_module.compute_embedding(edges, **kwargs)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return embedding, elapsed, peak / 1024 / 1024   # bytes to MB


def embedding_to_vector(embedding, all_keys):
    """Project an embedding dict onto a fixed key space to numpy vector."""
    vec = np.array([embedding.get(k, 0.0) for k in all_keys], dtype=float)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def null_model_frequencies(edges, n_randomizations=30, seed=0):
    """
    Build a null-model distribution by randomly shuffling timestamps.
    Returns a dict: motif_key → list of frequencies across randomizations.
    Used for alpha / beta / gamma thresholds.
    """
    rng = random.Random(seed)
    timestamps = [t for _, _, t in edges]
    freq_map = defaultdict(list)

    for _ in range(n_randomizations):
        rng.shuffle(timestamps)
        shuffled = [(u, v, t) for (u, v, _), t in zip(edges, timestamps)]
        emb = etn.compute_embedding(shuffled)   # use ETN as reference
        for k, v in emb.items():
            freq_map[k].append(v)

    return freq_map


def adaptive_bin(x, thresholds):
    """Bin x using a sorted list of threshold boundaries."""
    for i, thr in enumerate(thresholds):
        if x < thr:
            return i
    return len(thresholds)


# METRIC 1 — COMPUTATIONAL PERFORMANCE

def metric_computational_performance(datasets=DATASETS):
    """
    For every (algo, dataset) pair measure:
      - runtime (seconds)
      - peak memory (MB)
      - number of distinct motifs discovered
    """
    print("\n" + "="*70)
    print("METRIC 1 - Computational Performance (Runtime . Memory . Motif Count)")
    print("="*70)

    rows = []
    results = {}   # store embeddings for later metrics

    for algo_name, algo_mod in ALGOS.items():
        for ds_name, edges in datasets.items():
            emb, elapsed, mem_mb = run_with_metrics(algo_mod, edges)
            motif_count = len(emb)
            rows.append([algo_name, ds_name,
                         f"{elapsed:.4f}s",
                         f"{mem_mb:.2f} MB",
                         motif_count])
            results[(algo_name, ds_name)] = emb

    headers = ["Algorithm", "Dataset", "Runtime", "Peak Memory", "Motif Count"]
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    return results


# METRIC 2 — SCALABILITY BENCHMARK
def metric_scalability():
    """
    Grow network size by sampling edges from a real dataset.
    Record runtime for each algo at each scale.
    """
    print("\n" + "="*70)
    print("METRIC 2 - Scalability Benchmark (runtime vs. network size)")
    print("="*70)

    sizes = [50, 100, 250, 500, 1000, 2000]
    scale_results = {name: [] for name in ALGOS}
    TIMEOUT = 30.0   # seconds

    # ✅ Pick one real dataset (Email is best for scalability)
    base_edges = DATASETS["Email"]

    # Optional: shuffle once to avoid bias
    import random
    random.shuffle(base_edges)

    for n_edges in sizes:
        if n_edges > len(base_edges):
            print(f"  Skipping {n_edges} (dataset too small)")
            for algo_name in ALGOS:
                scale_results[algo_name].append(None)
            continue

        # ✅ Take subset of real edges
        edges = base_edges[:n_edges]

        for algo_name, algo_mod in ALGOS.items():
            prev = scale_results[algo_name]

            # skip if already timed-out
            if prev and prev[-1] is None:
                scale_results[algo_name].append(None)
                continue

            _, elapsed, _ = run_with_metrics(algo_mod, edges)

            if elapsed > TIMEOUT:
                print(f"  ⚠  {algo_name} timed out at {n_edges} edges "
                      f"({elapsed:.1f}s > {TIMEOUT}s limit)")
                scale_results[algo_name].append(None)
            else:
                scale_results[algo_name].append(round(elapsed, 4))

    # Print table
    rows = []
    for algo_name, times in scale_results.items():
        row = [algo_name] + [
            f"{t:.4f}s" if t is not None else "TIMEOUT"
            for t in times
        ]
        rows.append(row)

    headers = ["Algorithm"] + [f"{s} edges" for s in sizes]
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    return sizes, scale_results

# METRIC 3 — EMBEDDING SIMILARITY & CROSS-DATASET STABILITY

def metric_embedding_similarity(perf_results):
    """
    For each algorithm, compute cosine similarity between every pair of
    dataset embeddings. Good clustering = School ↔ School high, School ↔ Workplace low.
    """
    print("\n" + "="*70)
    print("METRIC 3 - Embedding Similarity & Cross-Dataset Stability")
    print("="*70)

    ds_names = list(DATASETS.keys())

    for algo_name in ALGOS:
        # collect all keys seen across datasets for this algo
        all_keys = set()
        for ds_name in ds_names:
            all_keys.update(perf_results[(algo_name, ds_name)].keys())
        all_keys = sorted(all_keys, key=str)

        if not all_keys:
            print(f"\n  {algo_name}: no motifs found — skipping similarity.\n")
            continue

        vecs = np.array([
            embedding_to_vector(perf_results[(algo_name, ds_name)], all_keys)
            for ds_name in ds_names
        ])

        sim_matrix = cosine_similarity(vecs)

        print(f"\n  [{algo_name}] Cosine Similarity Matrix (1.0 = identical)")
        header = [""] + ds_names
        rows = []
        for i, ds_i in enumerate(ds_names):
            row = [ds_i] + [f"{sim_matrix[i, j]:.3f}" for j in range(len(ds_names))]
            rows.append(row)
        print(tabulate(rows, headers=header, tablefmt="grid"))

        # Stability score = avg same-type similarity minus avg cross-type similarity
        school_idx    = [i for i, d in enumerate(ds_names) if "School"    in d]
        workplace_idx = [i for i, d in enumerate(ds_names) if "Workplace" in d]

        if len(school_idx) >= 2:
            intra = np.mean([sim_matrix[i, j]
                             for i in school_idx for j in school_idx if i != j])
            if workplace_idx:
                inter = np.mean([sim_matrix[i, j]
                                 for i in school_idx for j in workplace_idx])
                print(f"Stability score (School intra={intra:.3f} vs "
                      f"School - Workplace inter={inter:.3f}): "
                      f"gap = {intra - inter:.3f}")


# METRIC 4 — INFORMATION LOSS (HYBRID: fixed vs adaptive binning)

def metric_information_loss(dataset_edges=None):
    """
    Compare the Hybrid model's fixed bin_size() function against
    an adaptive (percentile-based) binning strategy.
    Measures: motif count, entropy of distribution, cosine similarity between
    the two embeddings (high similarity = low information loss).
    """
    print("\n" + "="*70)
    print("METRIC 4 - Information Loss: Fixed vs Adaptive Binning (Hybrid)")
    print("="*70)

    if dataset_edges is None:
        dataset_edges = {name: edges for name, edges in DATASETS.items()}

    def compute_hybrid_fixed(edges, k=2):
        """Original hybrid with fixed thresholds (5, 20)."""
        return hybrid.compute_embedding(edges, k=k)

    def compute_hybrid_adaptive(edges, k=2):
        """Hybrid with percentile-based (33rd / 66th) adaptive thresholds."""
        from collections import defaultdict
        from itertools import combinations

        snap = defaultdict(list)
        for u, v, t in edges:
            snap[t].append((u, v))

        # Collect all node-count and edge-count values to compute percentiles
        all_node_counts, all_edge_counts, all_tri_counts = [], [], []
        for t, layer_edges in snap.items():
            nodes = set(n for e in layer_edges for n in e)
            adj = defaultdict(set)
            for u, v in layer_edges:
                adj[u].add(v); adj[v].add(u)
            tri = sum(1 for u in adj
                      for v, w in __import__('itertools').combinations(adj[u], 2)
                      if w in adj[v])
            all_node_counts.append(len(nodes))
            all_edge_counts.append(len(layer_edges))
            all_tri_counts.append(tri)

        def pct_thresholds(values):
            arr = sorted(values)
            n = len(arr)
            lo = arr[max(0, int(n * 0.33) - 1)]
            hi = arr[max(0, int(n * 0.66) - 1)]
            return [lo, hi] if lo != hi else [lo, lo + 1]

        nt = pct_thresholds(all_node_counts)
        et = pct_thresholds(all_edge_counts)
        tt = pct_thresholds(all_tri_counts)

        counts = defaultdict(int)
        for t, layer_edges in snap.items():
            if t % 5 != 0:
                continue
            active = set(n for e in layer_edges for n in e)
            for ego in active:
                signature = []
                for dt in range(k):
                    layer = snap.get(t + dt, [])
                    nodes = set(n for e in layer for n in e)
                    adj = defaultdict(set)
                    for u, v in layer:
                        adj[u].add(v); adj[v].add(u)
                    tri = sum(1 for u in adj
                              for v2, w in combinations(adj[u], 2)
                              if w in adj[v2])
                    signature.append((
                        adaptive_bin(len(nodes), nt),
                        adaptive_bin(len(layer), et),
                        adaptive_bin(tri, tt),
                    ))
                counts[tuple(signature)] += 1

        total = sum(counts.values()) or 1
        return {k: v / total for k, v in counts.items()}

    def entropy(emb):
        probs = np.array(list(emb.values()))
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def direct_cosine(emb_a, emb_b):
        """Both embeddings share the same key format; compute cosine directly."""
        all_keys = sorted(set(emb_a) | set(emb_b), key=str)
        if not all_keys:
            return 0.0
        va = np.array([emb_a.get(k, 0.0) for k in all_keys])
        vb = np.array([emb_b.get(k, 0.0) for k in all_keys])
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(va / na, vb / nb))

    rows = []
    for ds_name, edges in dataset_edges.items():
        emb_fixed    = compute_hybrid_fixed(edges)
        emb_adaptive = compute_hybrid_adaptive(edges)

        if not emb_fixed and not emb_adaptive:
            continue

        sim = direct_cosine(emb_fixed, emb_adaptive)

        rows.append([
            ds_name,
            len(emb_fixed),
            f"{entropy(emb_fixed):.4f}",
            len(emb_adaptive),
            f"{entropy(emb_adaptive):.4f}",
            f"{sim:.4f}",
            "LOW" if sim > 0.90 else ("MEDIUM" if sim > 0.70 else "HIGH"),
        ])

    headers = ["Dataset",
               "Fixed Motifs", "Fixed Entropy",
               "Adaptive Motifs", "Adaptive Entropy",
               "Cosine Sim", "Info Loss"]
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print("  Info Loss interpretation: cosine similarity between fixed & adaptive embeddings.")
    print("  HIGH loss (sim < 0.70) means fixed thresholds lose significant structural detail.")


# METRIC 5 — HYPERPARAMETER SENSITIVITY

def metric_hyperparameter_sensitivity():
    """
    Sweep:
      - k (neighborhood sequence length / Δt window): 1, 2, 3, 4
      - sample_rate (for HETN and Hybrid): 1, 3, 5, 10

    For each combination, record motif count and runtime.
    """
    print("\n" + "="*70)
    print("METRIC 5 - Hyperparameter Sensitivity (k and sample_rate sweeps)")
    print("="*70)

    edges = DATASETS["CollegeMsg"]

    # ── 5a: k sweep (all algos) ──────────────────────────────────────────────
    print("\n  [5a] k (delta t window / neighborhood sequence length) sweep")
    k_values = [1, 2, 3, 4]
    rows_k = []
    for algo_name, algo_mod in ALGOS.items():
        row = [algo_name]
        for k in k_values:
            emb, elapsed, _ = run_with_metrics(algo_mod, edges, k=k)
            row.append(f"{len(emb)} motifs / {elapsed:.3f}s")
        rows_k.append(row)
    headers_k = ["Algorithm"] + [f"k={k}" for k in k_values]
    print(tabulate(rows_k, headers=headers_k, tablefmt="grid"))

    # ── 5b: sample_rate sweep (HETN only — Hybrid uses hardcoded t%5 internally)
    print("\n  [5b] sample_rate sweep (HETN only)")
    sample_rates = [1, 3, 5, 10]
    rows_sr = []
    for algo_name, algo_mod in [("HETN", hetn)]:
        row = [algo_name]
        for sr in sample_rates:
            emb, elapsed, _ = run_with_metrics(algo_mod, edges, sample_rate=sr)
            row.append(f"{len(emb)} motifs / {elapsed:.3f}s")
        rows_sr.append(row)
    headers_sr = ["Algorithm"] + [f"rate={r}" for r in sample_rates]
    print(tabulate(rows_sr, headers=headers_sr, tablefmt="grid"))
    print("  Note: Hybrid uses a hardcoded sample_rate of 5 (t % 5). To make it")
    print("  configurable, add 'sample_rate=5' as a parameter to hybrid.compute_embedding().")

    print("\n  Interpretation: larger k reveals longer temporal routines but increases")
    print("  motif space and compute time. Higher sample_rate reduces both.")


# METRIC 6 — NULL-MODEL STATISTICAL THRESHOLDS (alpha, beta, gamma)

def metric_null_model(alpha=0.05, beta=2.0, gamma=0.01, n_rand=30):
    """
    For each dataset, compare ETN motif frequencies against a null model
    (timestamp-shuffled networks).

    A motif is considered SIGNIFICANT if ALL three conditions hold:
      alpha (over-representation p-value) : p < alpha
      beta (frequency ratio vs null)     : observed / mean_null > beta
      gamma (minimum frequency)           : observed frequency > gamma

    Reports: total motifs, significant motifs, and the top-10 significant ones.
    """
    print("\n" + "="*70)
    print(f"METRIC 6 - Null-Model Statistical Thresholds"
          f"(alpha={alpha}, beta={beta}, gamma={gamma})")
    print("="*70)

    summary_rows = []

    for ds_name, edges in DATASETS.items():
        real_emb   = etn.compute_embedding(edges)
        null_freqs = null_model_frequencies(edges, n_randomizations=n_rand)

        significant = {}
        for motif, obs_freq in real_emb.items():
            null_vals = null_freqs.get(motif, [0.0] * n_rand)
            null_arr  = np.array(null_vals + [0.0] * (n_rand - len(null_vals)))

            # alpha: fraction of null samples that are >= observed (empirical p-value)
            p_val = float(np.mean(null_arr >= obs_freq))

            # beta: ratio of observed to null mean
            null_mean = float(np.mean(null_arr)) if null_arr.mean() > 0 else 1e-9
            ratio     = obs_freq / null_mean

            # Apply all three gates
            if p_val < alpha and ratio > beta and obs_freq > gamma:
                significant[motif] = {
                    "obs_freq": obs_freq,
                    "null_mean": null_mean,
                    "ratio": ratio,
                    "p_val": p_val,
                }

        summary_rows.append([ds_name, len(real_emb), len(significant),
                              f"{len(significant)/max(len(real_emb),1)*100:.1f}%"])

        # Print top-10 most over-represented significant motifs
        top = sorted(significant.items(), key=lambda x: -x[1]["ratio"])[:10]
        if top:
            print(f"\n  [{ds_name}] Top significant motifs:")
            detail_rows = []
            for motif, stats in top:
                detail_rows.append([
                    str(motif)[:45],
                    f"{stats['obs_freq']:.4f}",
                    f"{stats['null_mean']:.4f}",
                    f"{stats['ratio']:.2f}×",
                    f"{stats['p_val']:.3f}",
                ])
            detail_headers = ["Motif (truncated)", "Obs Freq",
                              "Null Mean", "beta ratio", "p-value (alpha)"]
            print(tabulate(detail_rows, headers=detail_headers,
                           tablefmt="grid"))

    print("\n  Null-Model Summary:")
    print(tabulate(summary_rows,
                   headers=["Dataset", "Total Motifs",
                             "Significant Motifs", "% Significant"],
                   tablefmt="grid"))
    print(f"\n  A motif is SIGNIFICANT when: p < {alpha}  AND  "
          f"freq_ratio > {beta}  AND  freq > {gamma}")


# PLOTS

def generate_plots(scale_sizes, scale_results, perf_results):
    """Generate a 2×2 summary figure and save to evaluation_plots.png."""
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    colors = {"ETN": "#2196F3", "Improved ETN": "#F44336",
              "HETN": "#FF9800", "Hybrid": "#4CAF50"}

    # ── Plot 1: Scalability ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for algo_name, times in scale_results.items():
        valid_x, valid_y = [], []
        for x, t in zip(scale_sizes, times):
            if t is not None:
                valid_x.append(x); valid_y.append(t)
        ax1.plot(valid_x, valid_y, marker="o", label=algo_name,
                 color=colors[algo_name])
    ax1.set_title("Scalability: Runtime vs Network Size", fontsize=11)
    ax1.set_xlabel("Number of Edges"); ax1.set_ylabel("Runtime (s)")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # ── Plot 2: Motif counts per dataset ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ds_names   = list(DATASETS.keys())
    algo_names = list(ALGOS.keys())
    x = np.arange(len(ds_names))
    width = 0.2
    for i, algo_name in enumerate(algo_names):
        counts = [len(perf_results[(algo_name, ds)]) for ds in ds_names]
        ax2.bar(x + i * width, counts, width, label=algo_name,
                color=colors[algo_name], alpha=0.85)
    ax2.set_title("Motif Count per Dataset", fontsize=11)
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(ds_names, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("Motif Count"); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3, axis="y")

    # ── Plot 3: Runtime per dataset (bar) ────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    runtime_data = {algo: [] for algo in algo_names}
    for algo_name, algo_mod in ALGOS.items():
        for ds_name, edges in DATASETS.items():
            _, elapsed, _ = run_with_metrics(algo_mod, edges)
            runtime_data[algo_name].append(elapsed)
    for i, algo_name in enumerate(algo_names):
        ax3.bar(x + i * width, runtime_data[algo_name], width,
                label=algo_name, color=colors[algo_name], alpha=0.85)
    ax3.set_title("Runtime per Dataset", fontsize=11)
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels(ds_names, rotation=20, ha="right", fontsize=8)
    ax3.set_ylabel("Runtime (s)"); ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3, axis="y")

    # ── Plot 4: k sensitivity (motif count) ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    k_values = [1, 2, 3, 4]
    edges_sample = DATASETS["CollegeMsg"]
    for algo_name, algo_mod in ALGOS.items():
        motif_counts = []
        for k in k_values:
            emb, _, _ = run_with_metrics(algo_mod, edges_sample, k=k)
            motif_counts.append(len(emb))
        ax4.plot(k_values, motif_counts, marker="s", label=algo_name,
                 color=colors[algo_name])
    ax4.set_title("Hyperparameter Sensitivity: k vs Motif Count", fontsize=11)
    ax4.set_xlabel("k (Δt window)"); ax4.set_ylabel("Motif Count")
    ax4.set_xticks(k_values); ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

    plt.suptitle("Temporal Network Motif — Evaluation Summary", fontsize=13, fontweight="bold")
    out_path = "evaluation_plots.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [OK] Plots saved to {out_path}")


# MAIN

def main():
    # print("\n" + "█"*70)
    print("TEMPORAL NETWORK MOTIF - FULL EVALUATION SUITE")
    # print("█"*70)
    print("  Algorithms : ETN | Improved ETN | HETN | Hybrid (Optimized)")
    print(f"  Datasets   : {', '.join(DATASETS.keys())}")
    print("  Metrics    : Computational . Scalability . Similarity .")
    print("               Info Loss . Hyperparams . Null-Model Stats")
    # print("█"*70)

    # ── Run all metrics in order ─────────────────────────────────────────────
    perf_results                = metric_computational_performance()
    scale_sizes, scale_results  = metric_scalability()
    metric_embedding_similarity(perf_results)
    metric_information_loss()
    metric_hyperparameter_sensitivity()
    metric_null_model(alpha=0.05, beta=2.0, gamma=0.01, n_rand=30)

    # ── Generate summary plots ───────────────────────────────────────────────
    print("\n" + "="*70)
    print("PLOTS")
    print("="*70)
    generate_plots(scale_sizes, scale_results, perf_results)

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()