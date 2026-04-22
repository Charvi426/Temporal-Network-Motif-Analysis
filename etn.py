# etn.py

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
                neigh = set()
                if t + dt in snap:
                    for u, v in snap[t + dt]:
                        if u == ego:
                            neigh.add(v)
                        elif v == ego:
                            neigh.add(u)
                layers.append(neigh)

            all_nodes = set().union(*layers)
            if not all_nodes:
                continue

            sig = []
            for n in all_nodes:
                s = tuple(1 if n in layer else 0 for layer in layers)
                sig.append(s)

            sig.sort()
            counts[tuple(sig)] += 1

    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}