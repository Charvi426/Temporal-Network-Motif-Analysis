# hetn.py

from collections import defaultdict
from itertools import combinations

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

        adj = defaultdict(set)
        for u, v in snap[t]:
            adj[u].add(v)
            adj[v].add(u)

        triangles = set()
        for u in adj:
            for v, w in combinations(adj[u], 2):
                if w in adj[v]:
                    triangles.add(tuple(sorted((u, v, w))))

        for ego in adj:
            nodes = set([ego])
            nodes.update(adj[ego])

            first = []
            for n in nodes:
                if n == ego:
                    continue
                first.append(n)

            second = []
            for a, b in combinations(nodes, 2):
                if tuple(sorted((ego, a, b))) in triangles:
                    second.append((a, b))

            sig = (tuple(sorted(first)), tuple(sorted(second)))
            counts[sig] += 1

    total = sum(counts.values()) if counts else 1
    return {k: v / total for k, v in counts.items()}