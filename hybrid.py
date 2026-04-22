# hybrid_optimized.py

from collections import defaultdict
from itertools import combinations

def build_snapshots(edges):
    snap = defaultdict(list)
    for u, v, t in edges:
        snap[t].append((u, v))
    return snap


def bin_size(x):
    if x < 5: return 0
    elif x < 20: return 1
    else: return 2


def compute_embedding(edges, k=2):
    snap = build_snapshots(edges)
    counts = defaultdict(int)

    for t in snap:
        if t % 5 != 0:   # 🔥 sampling
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

                # limit neighbors
                nodes = set(list(nodes)[:10])

                # counts
                num_nodes = len(nodes)
                num_edges = len(layer)

                # triangles
                tri = 0
                for u in adj:
                    for v, w in combinations(adj[u], 2):
                        if w in adj[v]:
                            tri += 1

                signature.append((
                    bin_size(num_nodes),
                    bin_size(num_edges),
                    bin_size(tri)
                ))

            counts[tuple(signature)] += 1

    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}