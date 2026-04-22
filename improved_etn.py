# improved_etn.py

from collections import defaultdict

def build_snapshots(edges):
    snap = defaultdict(list)
    for u, v, t in edges:
        snap[t].append((u, v))
    return snap


def compute_embedding(edges, k=2):
    snap = build_snapshots(edges)
    counts = defaultdict(int)

    for t in snap:
        active_nodes = set()
        for u, v in snap[t]:
            active_nodes.add(u)
            active_nodes.add(v)

        for ego in active_nodes:
            layers = []

            for dt in range(k):
                edges_layer = snap.get(t + dt, [])
                layers.append(edges_layer)

            sig = []
            for layer in layers:
                nodes = set()
                for u, v in layer:
                    nodes.add(u)
                    nodes.add(v)
                sig.append(tuple(sorted(nodes)))

            counts[tuple(sig)] += 1

    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}