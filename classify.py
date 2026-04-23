import os
import math

# Import the original graph-level embedding function you wrote
from hybrid import compute_embedding

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
# 2. ETM-based Distance Math
# ==========================================
def calculate_cosine_distance(emb1, emb2):
    """Calculates the cosine distance between two graph embeddings"""
    # Identify all unique motifs across both graphs to create a shared feature space
    all_motifs = set(emb1.keys()).union(set(emb2.keys()))
    
    dot_product = 0.0
    norm1 = 0.0
    norm2 = 0.0
    
    for motif in all_motifs:
        v1 = emb1.get(motif, 0.0)
        v2 = emb2.get(motif, 0.0)
        dot_product += v1 * v2
        norm1 += v1 ** 2
        norm2 += v2 ** 2
        
    if norm1 == 0.0 or norm2 == 0.0:
        return 1.0 # Maximum distance if a graph has no motifs
        
    cosine_similarity = dot_product / (math.sqrt(norm1) * math.sqrt(norm2))
    return 1.0 - cosine_similarity

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    # We mix Q&A forums with unrelated networks to prove the clustering works
    datasets = {
        "MathOverflow": "sx-mathoverflow.txt",
        "SuperUser": "sx-superuser.txt",
        "Email": "email-Eu-core-temporal.txt",
        "CollegeMsg": "CollegeMsg.txt"
    }
    
    delta_t = 3600 # 1-hour aggregation window
    embeddings = {}
    
    print("Extracting behavioral signatures for each social environment...\n")
    for name, filepath in datasets.items():
        if os.path.exists(filepath):
            edges = load_temporal_edges(filepath, delta_t)
            embeddings[name] = compute_embedding(edges, k=2)
            print(f"[{name}] successfully processed.")
        else:
            print(f"Warning: {filepath} not found.")
            
    print("\n=======================================================")
    print("     ETM-BASED DISTANCE MATRIX (HYBRID MODEL)          ")
    print("=======================================================")
    
    names = list(embeddings.keys())
    
    # Print the table header
    header = f"{'':<15} | " + " | ".join([f"{n[:10]:<10}" for n in names])
    print(header)
    print("-" * len(header))
    
    # Print the table rows
    for name1 in names:
        row = f"{name1:<15} | "
        for name2 in names:
            dist = calculate_cosine_distance(embeddings[name1], embeddings[name2])
            # Format distance to 3 decimal places
            row += f"{dist:<10.3f} | "
        print(row)