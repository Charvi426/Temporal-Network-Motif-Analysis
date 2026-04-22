# comparison.py

import matplotlib.pyplot as plt

import etn
import improved_etn
import hetn
import hybrid


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data(path):
    edges = []
    with open(path) as f:
        for line in f:
            u, v, t = map(int, line.strip().split())
            edges.append((u, v, t))
    return edges


# -----------------------------
# PLOT FUNCTION
# -----------------------------
def plot_single_method(names, values, title):
    plt.figure()

    plt.bar(names, values)

    # Add value labels
    for i, v in enumerate(values):
        plt.text(i, v, str(v), ha='center', va='bottom')

    plt.title(title)
    plt.ylabel("Number of Motifs")


# -----------------------------
# MAIN
# -----------------------------
def main():
    datasets = {
        "CollegeMsg": "CollegeMsg.txt",
        "Email": "email-Eu-core-temporal.txt",
        "SuperUser": "sx-superuser.txt"
    }

    names = list(datasets.keys())

    etn_vals = []
    imp_vals = []
    hetn_vals = []
    hybrid_vals = []

    # -----------------------------
    # COMPUTE EMBEDDINGS
    # -----------------------------
    for name, path in datasets.items():
        print(f"\nProcessing {name}...")

        edges = load_data(path)

        e_etn = etn.compute_embedding(edges)
        e_imp = improved_etn.compute_embedding(edges)
        e_hetn = hetn.compute_embedding(edges)
        e_hybrid = hybrid.compute_embedding(edges)

        etn_vals.append(len(e_etn))
        imp_vals.append(len(e_imp))
        hetn_vals.append(len(e_hetn))
        hybrid_vals.append(len(e_hybrid))

    # =============================
    # 📊 GRAPH 1: ETN
    # =============================
    plot_single_method(names, etn_vals, "ETN Across Datasets")

    # =============================
    # 📊 GRAPH 2: IMPROVED ETN
    # =============================
    plot_single_method(names, imp_vals, "Improved ETN Across Datasets")

    # =============================
    # 📊 GRAPH 3: HETN
    # =============================
    plot_single_method(names, hetn_vals, "HETN Across Datasets")

    # =============================
    # 📊 GRAPH 4: HYBRID
    # =============================
    plot_single_method(names, hybrid_vals, "Hybrid Model Across Datasets")

    plt.show()


if __name__ == "__main__":
    main()