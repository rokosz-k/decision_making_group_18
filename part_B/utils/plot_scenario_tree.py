import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_scenario_tree(root, figsize=(14, 8), show_values=False, show_probs=False):
    """
    Plot a graphical representation of a scenario tree.

    Parameters
    ----------
    root        : Node  – the root node returned by build_scenario_tree()
    figsize     : tuple – figure size in inches
    show_values : bool  – annotate nodes with (price, occ1, occ2)
    show_probs  : bool  – annotate edges with conditional probabilities
    """

    # ------------------------------------------------------------------ #
    #  1.  Collect all nodes and assign (x=stage, y=vertical position)    #
    # ------------------------------------------------------------------ #
    def collect_levels(root):
        """BFS – returns list of lists, one list per stage."""
        levels = []
        current = [root]
        while current:
            levels.append(current)
            nxt = []
            for n in current:
                nxt.extend(n.children)
            current = nxt
        return levels

    levels = collect_levels(root)
    pos = {}  # node_id -> (x, y)

    for x, level in enumerate(levels):
        n_nodes = len(level)
        ys = np.linspace(0, 1, n_nodes) if n_nodes > 1 else [0.5]
        for node, y in zip(level, ys):
            pos[id(node)] = (x, y)

    # ------------------------------------------------------------------ #
    #  2.  Draw                                                            #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#F7F9FC")
    fig.patch.set_facecolor("#F7F9FC")

    num_stages = len(levels)

    # Colour map: root → leaf fades from steel-blue to coral
    stage_colors = plt.cm.coolwarm(np.linspace(0.15, 0.85, num_stages))

    # --- edges first (drawn below nodes) ---
    for level in levels:
        for node in level:
            xp, yp = pos[id(node)]
            for child in node.children:
                xc, yc = pos[id(child)]
                ax.plot(
                    [xp, xc], [yp, yc],
                    color="#AABCD0", linewidth=1.2, zorder=1, alpha=0.7
                )
                if show_probs:
                    mx, my = (xp + xc) / 2, (yp + yc) / 2
                    ax.text(
                        mx, my,
                        f"{child.cond_prob:.2f}",
                        fontsize=6, color="#5A7A9A",
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="#F7F9FC",
                                  ec="none", alpha=0.75),
                        zorder=2,
                    )

    # --- nodes ---
    for x, level in enumerate(levels):
        color = stage_colors[x]
        is_leaf = (x == num_stages - 1)
        is_root = (x == 0)

        for node in level:
            xp, yp = pos[id(node)]

            marker = "D" if is_root else ("s" if is_leaf else "o")
            ms = 120 if is_root else (80 if is_leaf else 60)

            ax.scatter(
                xp, yp,
                s=ms, marker=marker,
                color=color, edgecolors="#2C4A6E",
                linewidths=0.8, zorder=3,
            )

            if show_values:
                p, o1, o2 = node.value[0], node.value[1], node.value[2]
                label = f"p={p:.1f}\no₁={o1:.2f}\no₂={o2:.2f}"
                ax.text(
                    xp, yp - 0.045,
                    label,
                    fontsize=5.5, color="#1C2E40",
                    ha="center", va="top", zorder=4,
                    fontfamily="monospace",
                )

    # ------------------------------------------------------------------ #
    #  3.  Axes & labels                                                   #
    # ------------------------------------------------------------------ #
    stage_labels = [f"τ+{x}" if x > 0 else "τ (root)" for x in range(num_stages)]
    ax.set_xticks(range(num_stages))
    ax.set_xticklabels(stage_labels, fontsize=9, color="#2C4A6E")
    ax.set_yticks([])
    ax.set_xlim(-0.4, num_stages - 0.6)
    ax.set_ylim(-0.08, 1.08)
    ax.set_xlabel("Stage", fontsize=10, color="#2C4A6E", labelpad=8)
    ax.set_title("Scenario Tree", fontsize=13, fontweight="bold",
                 color="#1C2E40", pad=12)

    # Spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.set_tick_params(length=0)

    # Legend
    legend_elements = [
        mpatches.Patch(color=stage_colors[0],  label="Root node"),
        mpatches.Patch(color=stage_colors[num_stages // 2], label="Intermediate node"),
        mpatches.Patch(color=stage_colors[-1], label="Leaf node"),
    ]
    #ax.legend(handles=legend_elements, loc="upper right",
    #          fontsize=8, framealpha=0.6, edgecolor="#AABCD0")

    plt.tight_layout()
    return fig, ax


# ------------------------------------------------------------------ #
#  Quick smoke-test with a toy tree (no environment needed)           #
# ------------------------------------------------------------------ #
if __name__ == "__main__":

    class Node:
        def __init__(self, value, cond_prob, parent, stage):
            self.value = value
            self.cond_prob = cond_prob
            self.parent = parent
            self.stage = stage
            self.children = []

        def scenario_probability(self):
            prob = self.cond_prob
            node = self
            while node.parent is not None:
                node = node.parent
                prob *= node.cond_prob
            return prob

        def path_from_root(self):
            path, node = [], self
            while node is not None:
                path.append(node)
                node = node.parent
            return list(reversed(path))

    rng = np.random.default_rng(0)

    def make_tree(B=3, L=3):
        root = Node(value=[100.0, 0.6, 0.7], cond_prob=1.0, parent=None, stage=0)
        current = [root]
        for stage in range(1, L):
            nxt = []
            for parent in current:
                probs = rng.dirichlet(np.ones(B))
                for k in range(B):
                    child = Node(
                        value=[parent.value[0] * rng.uniform(0.95, 1.05),
                               parent.value[1] * rng.uniform(0.9, 1.1),
                               parent.value[2] * rng.uniform(0.9, 1.1)],
                        cond_prob=float(probs[k]),
                        parent=parent,
                        stage=stage,
                    )
                    parent.children.append(child)
                    nxt.append(child)
            current = nxt
        return root

    root = make_tree(B=3, L=4)
    fig, ax = plot_scenario_tree(root, figsize=(14, 7))
    plt.show()