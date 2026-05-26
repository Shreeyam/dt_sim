"""Cartoon scheduling-graph figures.

Three SVGs:
  1. scheduling_graph.svg       -- DAG of accesses with feasible-transition edges.
  2. scheduling_graph_chain.svg -- same graph, with one feasible chain highlighted
                                   in shreeyam.mplstyle pink.
  3. scheduling_graph_only.svg  -- just the highlighted chain on its own.

Cartoon look: small figure, high DPI, heavy black node outlines, thick edges.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MPL_STYLE = REPO_ROOT / "shreeyam.mplstyle"
OUT_DIR = Path(__file__).resolve().parents[1] / "figures"

plt.style.use(str(MPL_STYLE))

PINK = "#ea1a69"
NODE_FACE = "#ffffff"
EDGE_GREY = "#bdbdbd"
LIGHT_GREY = "#dcdcdc"
NODE_EDGE = "#000000"


def agility_dt(theta_a: float, theta_b: float, t_s: float = 6.0,
               t_dotsq: float = 1.5) -> float:
    """Bang-bang slew time between two pointing angles."""
    return t_s + t_dotsq * np.sqrt(abs(theta_b - theta_a))


def build_graph(seed: int = 11):
    rng = np.random.default_rng(seed)
    N = 14
    t = np.sort(rng.uniform(0.0, 60.0, N))
    theta = rng.uniform(-25.0, 25.0, N)
    nodes = np.column_stack([t, theta])

    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            dt = nodes[j, 0] - nodes[i, 0]
            if dt <= 0:
                continue
            if dt >= agility_dt(nodes[i, 1], nodes[j, 1]):
                edges.append((i, j))
    return nodes, edges


def relax_nodes(nodes: np.ndarray, chain: list[int],
                fig_w: float = 2.4, fig_h: float = 1.45,
                n_iter: int = 400, r_min: float = 0.28,
                step: float = 0.7, anchor: float = 0.008) -> np.ndarray:
    """Soft-repel near-overlapping nodes in visual space.

    Each step pushes pairs apart along their visual-space connector and pulls
    everything gently back toward its original position so the layout still
    resembles the input. Chain points get a stronger pull toward theta=0 to
    keep the schedule running through the visual middle.
    """
    nodes = nodes.copy()
    orig = nodes.copy()

    pad_x, pad_y = 4.0, 6.0
    x_span = (orig[:, 0].max() - orig[:, 0].min()) + 2 * pad_x
    y_span = (orig[:, 1].max() - orig[:, 1].min()) + 2 * pad_y
    sx = fig_w / x_span  # data -> inches
    sy = fig_h / y_span

    chain_set = set(chain)
    N = len(nodes)

    for _ in range(n_iter):
        forces = np.zeros_like(nodes)
        for i in range(N):
            for j in range(i + 1, N):
                dx = (nodes[j, 0] - nodes[i, 0]) * sx
                dy = (nodes[j, 1] - nodes[i, 1]) * sy
                d = np.hypot(dx, dy)
                if 1e-9 < d < r_min:
                    push = 0.5 * (r_min - d) / d
                    fx = dx * push / sx
                    fy = dy * push / sy
                    forces[i] -= [fx, fy]
                    forces[j] += [fx, fy]
        nodes += step * forces
        nodes += anchor * (orig - nodes)
        for k in chain_set:
            nodes[k, 1] *= 0.92  # gently centre chain on theta=0

    return nodes


def longest_chain(nodes: np.ndarray, edges: list[tuple[int, int]]) -> list[int]:
    """Longest path in the DAG, via DP in topological (time) order."""
    N = len(nodes)
    adj: dict[int, list[int]] = {i: [] for i in range(N)}
    for a, b in edges:
        adj[a].append(b)

    best_len = [1] * N
    best_next = [-1] * N
    for i in range(N - 1, -1, -1):
        for j in adj[i]:
            if best_len[j] + 1 > best_len[i]:
                best_len[i] = best_len[j] + 1
                best_next[i] = j

    start = int(np.argmax(best_len))
    chain = [start]
    while best_next[chain[-1]] != -1:
        chain.append(best_next[chain[-1]])
    return chain


def score_seed(seed: int) -> tuple[float, dict]:
    """Lower is better. Penalises uneven time spacing and chain offset from 0."""
    nodes, edges = build_graph(seed)
    chain = longest_chain(nodes, edges)

    # Even-spacing in time: variance of consecutive gaps (sorted).
    t_sorted = np.sort(nodes[:, 0])
    gaps = np.diff(t_sorted)
    spacing_score = gaps.std() / gaps.mean()

    # Chain centred on theta=0, low variance.
    chain_theta = nodes[chain, 1]
    centre_score = abs(chain_theta.mean()) / 25.0
    spread_score = chain_theta.std() / 25.0

    # Chain spans the time range.
    chain_t = nodes[chain, 0]
    span_score = 1.0 - (chain_t.max() - chain_t.min()) / (t_sorted.max() - t_sorted.min())

    # Want chain length >= 4 for a useful illustration.
    if len(chain) < 4:
        return 1e9, {}

    score = spacing_score + 1.5 * centre_score + spread_score + 0.7 * span_score
    return score, dict(
        seed=seed, chain_len=len(chain),
        spacing=spacing_score, centre=centre_score,
        spread=spread_score, span=span_score, total=score,
    )


def _setup_axes(ax, nodes: np.ndarray) -> None:
    pad_x = 4.0
    pad_y = 6.0
    ax.set_xlim(nodes[:, 0].min() - pad_x, nodes[:, 0].max() + pad_x)
    ax.set_ylim(nodes[:, 1].min() - pad_y, nodes[:, 1].max() + pad_y)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("auto")


def _draw_edges(ax, nodes, edges, color, lw, alpha=1.0, zorder=1):
    for a, b in edges:
        ax.annotate(
            "",
            xy=(nodes[b, 0], nodes[b, 1]),
            xytext=(nodes[a, 0], nodes[a, 1]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=lw,
                alpha=alpha,
                shrinkA=7.0,
                shrinkB=7.0,
                mutation_scale=10,
                joinstyle="round",
                capstyle="round",
            ),
            zorder=zorder,
        )


def _draw_nodes(ax, nodes, indices, face, edge, size=180, lw=1.6, zorder=3):
    pts = nodes[indices]
    ax.scatter(
        pts[:, 0], pts[:, 1],
        s=size, facecolor=face, edgecolor=edge,
        linewidths=lw, zorder=zorder,
    )


def _new_fig():
    fig, ax = plt.subplots(figsize=(2.4, 1.45), dpi=200)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    return fig, ax


def plot_full_graph(nodes, edges, out: Path) -> None:
    fig, ax = _new_fig()
    _setup_axes(ax, nodes)
    _draw_edges(ax, nodes, edges, color=EDGE_GREY, lw=1.0, zorder=1)
    _draw_nodes(ax, nodes, list(range(len(nodes))), face=NODE_FACE, edge=NODE_EDGE)
    fig.tight_layout(pad=0.1)
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_graph_with_chain(nodes, edges, chain, out: Path) -> None:
    chain_set = set(zip(chain[:-1], chain[1:]))
    other_edges = [e for e in edges if e not in chain_set]
    chain_edges = list(chain_set)

    fig, ax = _new_fig()
    _setup_axes(ax, nodes)
    _draw_edges(ax, nodes, other_edges, color=EDGE_GREY, lw=1.0, zorder=1)
    _draw_edges(ax, nodes, chain_edges, color=PINK, lw=2.4, zorder=2)

    non_chain = [i for i in range(len(nodes)) if i not in chain]
    _draw_nodes(ax, nodes, non_chain, face=NODE_FACE, edge=NODE_EDGE, lw=1.6, zorder=3)
    _draw_nodes(ax, nodes, chain, face=PINK, edge=NODE_EDGE, lw=1.8, zorder=4)

    fig.tight_layout(pad=0.1)
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_dot_layout(out: Path) -> None:
    """Static dot layout: 1 + 3 + 6 = 10 white dots, no edges/arrows.

    Top row: 1 dot. Middle row: 3 evenly spaced. Bottom row: 6 evenly spaced.
    Same canvas / dot styling as the other scheduling-graph figures.
    """
    bounds = np.array([[0.0, -25.0], [60.0, 25.0]])  # match previous figs

    cols = np.array([22.0, 30.0, 38.0])
    grid_y = np.array([10.0, 0.0, -10.0])
    grid = np.array([(x, y) for y in grid_y for x in cols])
    top = np.array([[cols[0], 20.0]])  # extra dot above the leftmost column
    all_dots = np.vstack([top, grid])

    fig, ax = _new_fig()
    _setup_axes(ax, bounds)
    ax.scatter(
        all_dots[:, 0], all_dots[:, 1],
        s=180, facecolor=NODE_FACE, edgecolor=NODE_EDGE,
        linewidths=1.6, zorder=3,
    )
    fig.tight_layout(pad=0.1)
    fig.savefig(out, format="svg")
    plt.close(fig)


def find_lookahead_repair(nodes, edges, chain):
    """Find a non-chain bypass for chain[2] or chain[3].

    Prefers chain[3] so that chain[0]->chain[1] and chain[1]->chain[2] survive
    in the repaired chain. Tries a strict 1-node bypass first; falls back to a
    cartoon bypass where the alt is reachable from chain[ci-1] and sits in time
    between chain[ci] and chain[ci+1] — the (alt -> chain[ci+1]) edge gets
    drawn as part of the chain even if it's not strictly slew-feasible.
    """
    edges_set = set(edges)
    chain_set = set(chain)
    ci = 2
    if not (1 <= ci < len(chain) - 1):
        return None
    p, c, s = chain[ci - 1], chain[ci], chain[ci + 1]
    for alt in range(len(nodes)):
        if alt in chain_set:
            continue
        if (p, alt) in edges_set and (alt, s) in edges_set:
            return ci, [alt]
    for alt in range(len(nodes)):
        if alt in chain_set:
            continue
        if (p, alt) in edges_set and nodes[c, 0] < nodes[alt, 0] < nodes[s, 0]:
            return ci, [alt]
    return None


def _draw_cloudy_scatter(ax, nodes, indices, face, edge, lw, zorder):
    if not indices:
        return
    pts = nodes[indices]
    ax.scatter(
        pts[:, 0], pts[:, 1],
        s=180, facecolor=face, edgecolor=edge,
        linewidths=lw, hatch="///", zorder=zorder,
    )


def plot_lookahead_chain(nodes, edges, chain, lookahead_set, cloudy, out: Path) -> None:
    """Render the chain figure with a lookahead window emphasised.

    Outside-lookahead nodes/edges are drawn in light grey; inside-lookahead
    non-chain nodes get a black outline. Cloudy nodes get a hatched fill;
    if a cloudy node is on the chain, its fill is pink + hatched.
    """
    cloudy = set(cloudy)
    chain_set = set(chain)
    chain_edges = list(zip(chain[:-1], chain[1:]))
    chain_edge_set = set(chain_edges)
    other_edges = [e for e in edges if e not in chain_edge_set]
    in_edges = [e for e in other_edges
                if e[0] in lookahead_set and e[1] in lookahead_set]
    out_edges = [e for e in other_edges
                 if not (e[0] in lookahead_set and e[1] in lookahead_set)]

    fig, ax = _new_fig()
    _setup_axes(ax, nodes)
    _draw_edges(ax, nodes, out_edges, color=LIGHT_GREY, lw=1.0, zorder=1)
    _draw_edges(ax, nodes, in_edges, color=EDGE_GREY, lw=1.0, zorder=1)
    _draw_edges(ax, nodes, chain_edges, color=PINK, lw=2.4, zorder=2)

    outside_plain = [i for i in range(len(nodes))
                     if i not in lookahead_set and i not in chain_set and i not in cloudy]
    _draw_nodes(ax, nodes, outside_plain, face=NODE_FACE, edge=LIGHT_GREY,
                lw=1.6, zorder=3)

    inside_plain = [i for i in lookahead_set
                    if i not in chain_set and i not in cloudy]
    _draw_nodes(ax, nodes, inside_plain, face=NODE_FACE, edge=NODE_EDGE,
                lw=1.6, zorder=3)

    cloudy_off_chain = [i for i in cloudy if i not in chain_set]
    _draw_cloudy_scatter(ax, nodes, cloudy_off_chain,
                         face=NODE_FACE, edge=NODE_EDGE, lw=1.6, zorder=3)

    cloudy_on_chain = [i for i in cloudy if i in chain_set]
    _draw_cloudy_scatter(ax, nodes, cloudy_on_chain,
                         face=PINK, edge=NODE_EDGE, lw=1.8, zorder=4)

    plain_chain = [i for i in chain if i not in cloudy]
    _draw_nodes(ax, nodes, plain_chain, face=PINK, edge=NODE_EDGE,
                lw=1.8, zorder=4)

    fig.tight_layout(pad=0.1)
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_mlp(out: Path, layer_sizes=(3, 5, 2)) -> None:
    """MLP cartoon: stacked layers of white circles with grey connections.

    White background (not transparent), matching node/edge styling of the
    other scheduling-graph figures.
    """
    bounds = np.array([[0.0, -25.0], [60.0, 25.0]])  # match previous figs

    n_layers = len(layer_sizes)
    layer_x = np.linspace(14.0, 46.0, n_layers)
    spacing = 11.5  # vertical gap between nodes within a layer

    layers: list[np.ndarray] = []
    for n, x in zip(layer_sizes, layer_x):
        ys = (np.arange(n) - (n - 1) / 2.0) * spacing
        layers.append(np.column_stack([np.full(n, x), ys]))

    fig, ax = _new_fig()
    fig.patch.set_alpha(1.0)
    ax.patch.set_alpha(1.0)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    _setup_axes(ax, bounds)

    for a, b in zip(layers[:-1], layers[1:]):
        for pa in a:
            for pb in b:
                ax.plot(
                    [pa[0], pb[0]], [pa[1], pb[1]],
                    color=NODE_EDGE, lw=1.4, zorder=1,
                    solid_capstyle="round",
                )

    pts = np.vstack(layers)
    ax.scatter(
        pts[:, 0], pts[:, 1],
        s=180, facecolor=NODE_FACE, edgecolor=NODE_EDGE,
        linewidths=1.6, zorder=3,
    )

    fig.tight_layout(pad=0.1)
    fig.savefig(out, format="svg", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_chain_only(nodes, chain, out: Path) -> None:
    chain_edges = list(zip(chain[:-1], chain[1:]))

    fig, ax = _new_fig()
    _setup_axes(ax, nodes)
    _draw_edges(ax, nodes, chain_edges, color=PINK, lw=2.4, zorder=2)
    _draw_nodes(ax, nodes, chain, face=PINK, edge=NODE_EDGE, lw=1.8, zorder=4)

    fig.tight_layout(pad=0.1)
    fig.savefig(out, format="svg")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Seed picked by sweeping seeds 0..299 against score_seed and choosing the
    # one with a 5-node chain spanning the middle with the most-even timeline.
    nodes, edges = build_graph(seed=240)
    chain = longest_chain(nodes, edges)
    # Light repulsion pass to break up clumps without changing the structure.
    nodes = relax_nodes(nodes, chain)
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            dt = nodes[j, 0] - nodes[i, 0]
            if dt > 0 and dt >= agility_dt(nodes[i, 1], nodes[j, 1]):
                edges.append((i, j))
    # Make sure the chain edges survive the relaxation.
    for e in zip(chain[:-1], chain[1:]):
        if e not in edges:
            edges.append(e)

    full = OUT_DIR / "scheduling_graph.svg"
    with_chain = OUT_DIR / "scheduling_graph_chain.svg"
    chain_only = OUT_DIR / "scheduling_graph_only.svg"

    plot_full_graph(nodes, edges, full)
    plot_graph_with_chain(nodes, edges, chain, with_chain)
    plot_chain_only(nodes, chain, chain_only)

    dots = OUT_DIR / "scheduling_graph_dots.svg"
    plot_dot_layout(dots)

    mlp = OUT_DIR / "scheduling_graph_mlp.svg"
    plot_mlp(mlp)

    repair = find_lookahead_repair(nodes, edges, chain)
    if repair is None:
        raise RuntimeError("no bypass found in chain[1..3]; pick another seed")
    ci, alts = repair
    cloudy_chain_node = chain[ci]
    repaired_chain = chain[:ci] + alts + chain[ci + 1:]

    # Lookahead window: chain[1..3] plus the alt(s), excluding chain[0]/[4].
    t_key = list(nodes[chain[1:4], 0]) + [nodes[a, 0] for a in alts]
    t_lo = min(t_key) - 1.5
    t_hi = max(t_key) + 1.5
    lookahead_set = {i for i in range(len(nodes))
                     if t_lo <= nodes[i, 0] <= t_hi}

    # Each non-chain, non-alt node in the lookahead is cloudy with prob 0.5.
    excluded = set(chain) | set(alts)
    rng = np.random.default_rng(0)
    cloudy_set = {cloudy_chain_node}
    for i in sorted(lookahead_set):
        if i in excluded:
            continue
        if rng.random() < 0.5:
            cloudy_set.add(i)

    lookahead_path = OUT_DIR / "scheduling_graph_lookahead.svg"
    cloudy_path = OUT_DIR / "scheduling_graph_cloudy.svg"
    repaired_path = OUT_DIR / "scheduling_graph_repaired.svg"
    plot_lookahead_chain(nodes, edges, chain, lookahead_set,
                         cloudy=set(), out=lookahead_path)
    plot_lookahead_chain(nodes, edges, chain, lookahead_set,
                         cloudy=cloudy_set, out=cloudy_path)
    plot_lookahead_chain(nodes, edges, repaired_chain, lookahead_set,
                         cloudy=cloudy_set, out=repaired_path)

    for p in (full, with_chain, chain_only, dots, mlp,
              lookahead_path, cloudy_path, repaired_path):
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
