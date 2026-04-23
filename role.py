import os
from collections import defaultdict
from itertools import combinations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import numpy as np

def load_temporal_edges(filepath, delta_t_seconds):
    """Parses SNAP temporal edges into (u, v, discrete_layer)"""
    print(f"Loading temporal edges from {filepath}...")
    raw_edges = []
    min_t = float('inf')
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or line.startswith('%'): continue
            parts = line.strip().split()
            if len(parts) >= 3:
                u, v, t = int(parts[0]), int(parts[1]), float(parts[2])
                raw_edges.append((u, v, t))
                if t < min_t: min_t = t
                    
    discrete_edges = []
    for u, v, t in raw_edges:
        discrete_t = int((t - min_t) / delta_t_seconds)
        discrete_edges.append((u, v, discrete_t))
        
    print(f"Loaded {len(discrete_edges)} edges. Discretized into layers.")
    return discrete_edges

def load_department_labels(filepath):
    """Parses SNAP ground-truth community labels"""
    print(f"Loading department labels from {filepath}...")
    node_departments = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or line.startswith('%'): continue
            parts = line.strip().split()
            if len(parts) >= 2:
                node_id, dept_id = int(parts[0]), int(parts[1])
                node_departments[node_id] = dept_id
    print(f"Loaded labels for {len(node_departments)} users.")
    return node_departments

def load_department_labels_from_dept_files(dept_filepaths):
    """Build department labels from dept-wise temporal edge files."""
    print("Loading department labels from dept-wise edge files...")
    node_departments = {}

    for dept_id, filepath in enumerate(dept_filepaths, start=1):
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('#') or line.startswith('%'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    u, v = int(parts[0]), int(parts[1])
                    node_departments[u] = dept_id
                    node_departments[v] = dept_id

    print(f"Loaded labels for {len(node_departments)} users across {len(dept_filepaths)} departments.")
    return node_departments

# ==========================================
# 2. Optimized Hybrid Algorithm (Node-Level)
# ==========================================
def build_snapshots(edges):
    snap = defaultdict(list)
    for u, v, t in edges:
        snap[t].append((u, v))
    return snap

def bin_size(x):
    if x < 5: return 0
    elif x < 20: return 1
    else: return 2

def compute_node_embeddings(edges, k=2):
    print(f"Computing Hybrid embeddings for k={k}...")
    snap = build_snapshots(edges)
    node_counts = defaultdict(lambda: defaultdict(int))

    for t in snap:
        if t % 5 != 0:   # Timestamp sampling optimization
            continue

        active_nodes = set()
        for u, v in snap[t]:
            active_nodes.add(u)
            active_nodes.add(v)

        for ego in active_nodes:
            layers = []
            for dt in range(k):
                layers.append(snap.get(t + dt, []))

            signature = []
            for layer in layers:
                nodes = set()
                adj = defaultdict(set)

                for u, v in layer:
                    nodes.add(u)
                    nodes.add(v)
                    adj[u].add(v)
                    adj[v].add(u)

                # Limit neighborhood size optimization
                nodes = set(list(nodes)[:10])

                num_nodes = len(nodes)
                num_edges = len(layer)

                tri = 0
                for u_node in adj:
                    for v_node, w_node in combinations(adj[u_node], 2):
                        if w_node in adj[v_node]:
                            tri += 1

                signature.append((
                    bin_size(num_nodes),
                    bin_size(num_edges),
                    bin_size(tri)
                ))

            node_counts[ego][tuple(signature)] += 1

    # Normalize embeddings
    node_embeddings = {}
    for ego, counts in node_counts.items():
        total = sum(counts.values())
        node_embeddings[ego] = {sig: count / total for sig, count in counts.items()}
        
    print(f"Successfully extracted profiles for {len(node_embeddings)} individuals.")
    return node_embeddings

# ==========================================
# 3. Evaluation and Visualization
# ==========================================
def visualize_roles(user_embeddings, labels):
    print("Preparing feature vectors for MDS visualization...")
    
    # Identify all unique motifs to create a standard feature space
    all_motifs = set()
    for emb in user_embeddings.values():
        all_motifs.update(emb.keys())
    all_motifs = list(all_motifs)
    
    X = []
    colors = []
    
    # Build the feature matrix and assign colors based on the department
    for user, emb in user_embeddings.items():
        if user in labels:
            vec = [emb.get(m, 0.0) for m in all_motifs]
            X.append(vec)
            colors.append(labels[user])
            
    X = np.array(X)
    
    print("Running Multidimensional Scaling (MDS) to project into 2D space...")
    # Using random_state for reproducibility
    mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42, max_iter=100)
    X_2d = mds.fit_transform(X)
    
    print("Plotting results...")
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, cmap='tab20', alpha=0.7, edgecolors='w', s=50)
    plt.title("Role Identification via Hybrid Motifs\n(Colored by Department)")
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    
    # Add a colorbar to show different departments
    cbar = plt.colorbar(scatter)
    cbar.set_label("Department ID")
    
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "role_visualization.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved role visualization to {out_path}")

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    edge_file = os.path.join(base_dir, 'email-Eu-core-temporal.txt')
    dept_files = [
        os.path.join(base_dir, 'email-Eu-core-temporal-Dept1.txt'),
        os.path.join(base_dir, 'email-Eu-core-temporal-Dept2.txt'),
        os.path.join(base_dir, 'email-Eu-core-temporal-Dept3.txt'),
        os.path.join(base_dir, 'email-Eu-core-temporal-Dept4.txt'),
    ]

    missing_files = [p for p in [edge_file, *dept_files] if not os.path.exists(p)]
    if missing_files:
        print("Error: Missing dataset files:")
        for missing in missing_files:
            print(f" - {missing}")
    else:
        # We use an aggregation window of 3600 seconds (1 hour) for email
        edges = load_temporal_edges(edge_file, delta_t_seconds=3600)
        labels = load_department_labels_from_dept_files(dept_files)
        
        # Compute embeddings for each user
        user_embeddings = compute_node_embeddings(edges, k=2)
        
        # Plot the results
        visualize_roles(user_embeddings, labels)