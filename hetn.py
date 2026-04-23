from collections import defaultdict

def build_snapshots(edges):
    snap = defaultdict(list)
    for u, v, t in edges:
        snap[t].append((u, v))
    return snap

def compute_embedding(edges, k=2, sample_rate=5):
    snap = build_snapshots(edges)
    counts = defaultdict(int)

    for t in snap:
        if t % sample_rate != 0:
            continue

        active_nodes = set()
        for u, v in snap[t]:
            active_nodes.add(u)
            active_nodes.add(v)

        for ego in active_nodes:
            layers_neighbors = []
            layers_edges = []

            # 1. Track the sequence over 'k' time steps
            for dt in range(k):
                current_t = t + dt
                neigh = set()
                layer_adj = set()
                
                if current_t in snap:
                    for u, v in snap[current_t]:
                        # Track all edges in this layer to find triangles later
                        layer_adj.add((u, v))
                        layer_adj.add((v, u))
                        # Identify direct neighbors of the ego
                        if u == ego: neigh.add(v)
                        elif v == ego: neigh.add(u)
                        
                layers_neighbors.append(neigh)
                layers_edges.append(layer_adj)

            all_neighbors = set().union(*layers_neighbors)
            if not all_neighbors:
                continue

            # 2. First-Order Encoding (Individual Neighbors)
            first_order_sigs = {}
            for n in all_neighbors:
                # Create the binary sequence: 1 if present, 0 if absent for each time step
                sig = tuple(1 if n in layers_neighbors[dt] else 0 for dt in range(k))
                first_order_sigs[n] = sig

            # 3. Sort Neighbors Lexicographically
            # We sort the neighbors based on their first-order binary signature.
            # *Note on Isomorphism*: We use the node ID 'x' as a tie-breaker to keep the code fast. 
            sorted_neighbors = sorted(list(all_neighbors), key=lambda x: (first_order_sigs[x], x))
            
            final_first_order = tuple(first_order_sigs[n] for n in sorted_neighbors)

            # 4. Second-Order Encoding (Pairs forming Triangles with Ego)
            final_second_order = []
            
            # Iterate through all possible pairs following the EXACT sorting order established above
            for i in range(len(sorted_neighbors)):
                for j in range(i + 1, len(sorted_neighbors)):
                    a = sorted_neighbors[i]
                    b = sorted_neighbors[j]

                    # A second-order interaction (triangle) exists at time 'dt' if 'a' and 'b' 
                    # are both neighbors of the ego AND they interact with each other.
                    pair_sig = tuple(
                        1 if (a in layers_neighbors[dt] and 
                              b in layers_neighbors[dt] and 
                              (a, b) in layers_edges[dt]) else 0
                        for dt in range(k)
                    )
                    final_second_order.append(pair_sig)

            # 5. Concatenate to create the complete HETNS
            complete_signature = (final_first_order, tuple(final_second_order))
            counts[complete_signature] += 1

    total = sum(counts.values()) if counts else 1
    return {k: v / total for k, v in counts.items()}