import os
import time

# Import the original graph-level embedding functions you wrote
from etn import compute_embedding as compute_etn
from hybrid import compute_embedding as compute_hybrid

# ==========================================
# 1. Data Parsing Function
# ==========================================
def load_temporal_edges(filepath, delta_t_seconds):
    """Parses SNAP temporal edges into (u, v, discrete_layer)"""
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
        
    return discrete_edges

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    # The three datasets representing different technologies
    datasets = {
        "Email (Thread-based)": "email-Eu-core-temporal.txt",
        "CollegeMsg (Direct Messaging)": "CollegeMsg.txt",
        "SuperUser (Q&A Forum)": "sx-superuser.txt"
    }
    
    # Standardize the aggregation window (1 hour) to compare them fairly
    delta_t = 3600 
    
    print("Starting Communication Channel Evaluation...\n")
    print(f"{'Dataset':<30} | {'Base ETN Motifs':<15} | {'Hybrid Motifs':<15}")
    print("-" * 65)

    for name, filename in datasets.items():
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found. Skipping...")
            continue
            
        # Load the edges
        edges = load_temporal_edges(filename, delta_t)
        
        # Run Base ETN
        etn_embeddings = compute_etn(edges, k=2)
        total_etn_motifs = len(etn_embeddings)
        
        # Run Optimized Hybrid Model
        hybrid_embeddings = compute_hybrid(edges, k=2)
        total_hybrid_motifs = len(hybrid_embeddings)
        
        # Print the results
        print(f"{name:<30} | {total_etn_motifs:<15} | {total_hybrid_motifs:<15}")

    print("\nEvaluation Complete.")