#!/usr/bin/env python3
"""Camera-ready system-architecture figure for the streaming colocated RL paper.

Renders the data plane described by ``arch_spec.py``:

    StreamingRouter (owns MigrationPolicy)
        -> SGLang inference engines
        -> StreamingWorkQueue (owns GrabPolicy)
             -> streaming trainers

Layout is explicit (no auto-layout) so the figure is deterministic and fits a
known column width exactly. Everything is drawn in a 0..100 x 0..100 axis; point
sizes scale off ``--width`` so the figure stays legible at final print size.

Usage
-----
    python paper_figures/make_arch_figure.py --out paper_figures/out
    python paper_figures/make_arch_figure.py --width 3.4 --out paper_figures/out
    python paper_figures/make_arch_figure.py --num-gpus 4 --train-tp 2 --infer-tp 1
    python paper_figures/make_arch_figure.py --grayscale
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arch_spec import (  # noqa: E402
    ELIDED,
    Topology,
    build_spec,
    golden_height,
    gpu_layout,
)

# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------

# Colourblind-safe hues (blue / green / purple / amber), all light enough to
# carry black text. The figure must ALSO read in grayscale, so no distinction
# rides on hue alone: node kinds differ by border style, edge kinds by dash
# pattern, and the planes are separated by position.
COLOR = {
    "router": ("#E8F0FA", "#2B5D8A"),
    "policy": ("#FFFFFF", "#C2571A"),
    "ray_actor": ("#E6F4EA", "#2E7D4F"),
    "engine": ("#F1EAFB", "#5B3E96"),
    "engine_chip": ("#E2D4F5", "#5B3E96"),
    "trainer": ("#FDF3E0", "#A9741A"),
    "trainer_chip": ("#F8E3BC", "#A9741A"),
    "queue_slot": ("#CFE7D8", "#2E7D4F"),
    "gpu": ("#ECECEC", "#5A5A5A"),
    "gpu_band": ("#E9E9E9", "#E9E9E9"),
}

GRAY = {
    "router": ("#F0F0F0", "#000000"),
    "policy": ("#FFFFFF", "#000000"),
    "ray_actor": ("#F0F0F0", "#000000"),
    "engine": ("#F6F6F6", "#000000"),
    "engine_chip": ("#DDDDDD", "#000000"),
    "trainer": ("#F6F6F6", "#000000"),
    "trainer_chip": ("#DDDDDD", "#000000"),
    "queue_slot": ("#CCCCCC", "#000000"),
    "gpu": ("#E4E4E4", "#000000"),
    "gpu_band": ("#EDEDED", "#EDEDED"),
}

# Border linestyle per node kind -- the grayscale-safe part of the encoding.
NODE_LS = {
    "router": "-",
    "policy": (0, (4, 2)),
    "ray_actor": "-",
    "engine": "-",
    "trainer": "-",
}

# Dash pattern per edge kind.
EDGE_LS = {
    "dispatch": "-",
    "http_migrate": (0, (5, 2)),
    "produce": "-",
    "consume": "-",
    "colocate": (0, (1, 2)),
}


class Style:
    """Point sizes and palette, scaled to the requested figure width."""

    def __init__(self, width_in: float, grayscale: bool, edge_labels: bool = False):
        s = width_in / 7.0
        self.scale = s
        self.palette = GRAY if grayscale else COLOR
        self.grayscale = grayscale
        self.edge_labels = edge_labels
        self.fs_title = 9.5 * s
        self.fs_sub = 6.6 * s
        self.fs_note = 6.0 * s
        self.fs_chip = 6.8 * s
        self.fs_edge = 6.8 * s
        self.fs_edge_detail = 5.8 * s
        self.fs_foot = 6.0 * s
        self.lw_box = 1.15 * s
        self.lw_chip = 0.8 * s
        self.lw_edge = 1.25 * s
        self.arrow_ms = 9.0 * s

    def colors(self, key: str) -> tuple[str, str]:
        return self.palette[key]

    def accent(self, hue: str) -> str:
        """Annotation ink. Collapses to near-black under --grayscale so the
        ownership and migration callouts don't stay coloured in a mono render."""
        return "#333333" if self.grayscale else hue


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------


def box(ax, st: Style, x0, y0, x1, y1, key, ls="-", lw=None, z=2, radius=1.6):
    face, edge = st.colors(key)
    patch = FancyBboxPatch(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw if lw is not None else st.lw_box,
        linestyle=ls,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def text(ax, x, y, s, size, *, weight="normal", ha="center", va="center",
         color="#111111", z=5, style="normal", linespacing=1.35):
    return ax.text(
        x, y, s, fontsize=size, fontweight=weight, ha=ha, va=va, color=color,
        zorder=z, fontstyle=style, linespacing=linespacing,
    )


def arrow(ax, st: Style, p0, p1, kind, *, connectionstyle="arc3,rad=0",
          color=None, z=4, double=False):
    edge_color = color or ("#000000" if st.grayscale else "#333333")
    style = "<|-|>" if double else "-|>"
    patch = FancyArrowPatch(
        p0, p1,
        arrowstyle=f"{style},head_width={0.22 * st.scale + 0.14},"
                   f"head_length={0.40 * st.scale + 0.24}",
        connectionstyle=connectionstyle,
        linewidth=st.lw_edge,
        linestyle=EDGE_LS[kind],
        color=edge_color,
        mutation_scale=st.arrow_ms,
        shrinkA=0, shrinkB=0,
        zorder=z,
        joinstyle="miter",
    )
    ax.add_patch(patch)
    return patch


#: Vertical advance, in axis units, between the main edge label and the first
#: detail line, and between successive detail lines. Callers position labels by
#: their TOP edge (``y_top``) so collision budgeting is just arithmetic:
#: a label with n detail lines occupies ``[y_top - LABEL_LEAD - n*DETAIL_LEAD, y_top]``.
LABEL_LEAD = 3.0
DETAIL_LEAD = 2.15


def label_height(n_detail: int) -> float:
    """Axis-unit height of an :func:`edge_label` block. Used to budget corridors."""
    return LABEL_LEAD + n_detail * DETAIL_LEAD


def edge_label(ax, st: Style, x, y_top, label, detail=(), *, ha="center"):
    """Main label plus smaller detail lines, on an opaque plate.

    Two artists rather than one, because matplotlib cannot mix font sizes inside
    a single Text. Anchored by the block's top edge; see :func:`label_height`.

    The white plate matters: it keeps a label readable where it crosses a
    connector, which is otherwise the most common legibility failure here.
    """
    plate = dict(boxstyle="round,pad=0.26", facecolor="white", edgecolor="none",
                 alpha=0.95)
    t = ax.text(x, y_top, label, fontsize=st.fs_edge, ha=ha, va="top", zorder=6,
                color="#111111", linespacing=1.45)
    t.set_bbox(plate)
    if detail:
        d = ax.text(x, y_top - LABEL_LEAD, "\n".join(detail),
                    fontsize=st.fs_edge_detail, ha=ha, va="top", zorder=6,
                    color="#3A3A3A", linespacing=1.45)
        d.set_bbox(dict(plate, pad=0.22))
    return t


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------

# Frame coordinates in the 0..100 axis, chosen so every connector has a clear
# corridor and every label has an empty rectangle to sit in. Left column carries
# the colocated hardware (engines above trainers, tied by the dotted band); the
# right column carries the queue; the router spans the top.
# The flow reads as a loop: down the left column (router -> engines), right into
# the queue, then back left into the trainers, which share the engines' GPUs.
# Keeping the colocated pair in one column is what lets the GPU lane sit between
# them; the queue gets its own column so both of its connectors have room to be
# labelled.
ROUTER = (3.0, 80.0, 97.0, 96.0)
ENGINES = (3.0, 52.0, 54.0, 73.0)
TRAINERS = (3.0, 18.0, 54.0, 38.0)
QUEUE = (62.0, 30.0, 97.0, 68.0)

X_DISPATCH = 10.0      # router -> engines
Y_ENG_OUT = 70.0       # engines -> queue, leaves the engines box here
X_QUEUE_IN = 79.0      # engines -> queue, enters the queue's top edge here
X_PULL = 88.0          # queue <-> trainers, vertical leg
Y_PULL = 24.0          # queue <-> trainers, horizontal leg


def draw(spec, st: Style, ax) -> None:
    topo = spec.topology

    # ---------------- StreamingRouter (owns MigrationPolicy) ----------------
    rx0, ry0, rx1, ry1 = ROUTER
    box(ax, st, rx0, ry0, rx1, ry1, "router", ls=NODE_LS["router"])

    router = spec.node("router")
    text(ax, rx0 + 4.0, (ry0 + ry1) / 2, router.label, st.fs_title, weight="bold",
         ha="left")

    pol = spec.node("migration_policy")
    _draw_policy_box(ax, st, pol, rx1 - 30.0, ry0 + 2.5, rx1 - 4.0, ry1 - 2.5,
                     "owned by the router")

    # ---------------- SGLang inference engines ----------------
    ex0, ey0, ex1, ey1 = ENGINES
    box(ax, st, ex0, ey0, ex1, ey1, "engine", ls=NODE_LS["engine"])
    eng = spec.node("engines")
    text(ax, ex0 + 3.0, ey1 - 3.4, eng.label, st.fs_title * 0.92, weight="bold", ha="left")

    # One shared column per physical GPU. The engine row, the GPU lane and the
    # train-group row are all laid out against it.
    layout = gpu_layout(topo)
    cols = _column_bounds(ex0 + 3.0, ex1 - 3.0, len(layout.columns))
    chip_centers = [(c[0] + c[1]) / 2 for c in cols]
    _draw_spans(ax, st, layout.engines, cols, ey0 + 2.0, ey0 + 7.5, "engine_chip")

    # Migration drawn where it actually happens: one prompt group leaves a
    # lagging engine and is re-prefilled on a still-busy one. It is a spec edge
    # like any other (engines -> engines); the arc is just how a self-edge is
    # rendered here.
    e_migrate = next(e for e in spec.edges if e.kind == "http_migrate")
    if len(chip_centers) >= 2:
        src_x, dst_x = chip_centers[-1], chip_centers[0]
        y_arc = ey0 + 8.0
        arrow(ax, st, (src_x, y_arc), (dst_x, y_arc), "http_migrate",
              connectionstyle="arc3,rad=0.20", z=5, color=st.accent("#B14A12"))
        # Rides on the arc apex; the plate keeps the curve from striking
        # through the text.
        text(ax, (src_x + dst_x) / 2, y_arc + 4.0, e_migrate.label,
             st.fs_edge_detail, color=st.accent("#8A3A0E"), style="italic",
             z=6).set_bbox(
            dict(boxstyle="round,pad=0.26", facecolor=st.colors("engine")[0],
                 edgecolor="none", alpha=1.0))

    # ---------------- StreamingWorkQueue (owns GrabPolicy) ----------------
    qx0, qy0, qx1, qy1 = QUEUE
    box(ax, st, qx0, qy0, qx1, qy1, "ray_actor", ls=NODE_LS["ray_actor"])
    wq = spec.node("work_queue")
    text(ax, qx0 + 3.0, qy1 - 3.4, wq.label, st.fs_title, weight="bold", ha="left")

    # The pending list, as slots -- makes "the queue holds prompt groups" literal
    # so the box needs no sentence explaining it.
    slot_y0, slot_y1 = qy1 - 12.4, qy1 - 7.4
    n_slots = 7
    span = (qx1 - 3.0) - (qx0 + 3.0)
    gap = span * 0.02
    w = (span - gap * (n_slots - 1)) / n_slots
    face, edger = st.colors("queue_slot")
    for i in range(n_slots):
        sx = qx0 + 3.0 + i * (w + gap)
        filled = i < 4
        ax.add_patch(FancyBboxPatch(
            (sx, slot_y0), w, slot_y1 - slot_y0,
            boxstyle="round,pad=0,rounding_size=0.7",
            facecolor=face if filled else "#FFFFFF",
            edgecolor=edger, linewidth=st.lw_chip,
            linestyle="-" if filled else (0, (2, 1.6)), zorder=3,
        ))
    text(ax, qx0 + 3.0, slot_y0 - 2.4, wq.sublabel, st.fs_note, ha="left",
         va="center", color="#3A3A3A", style="italic")

    gp = spec.node("grab_policy")
    _draw_policy_box(ax, st, gp, qx0 + 3.0, qy0 + 2.6, qx1 - 3.0, qy0 + 13.6,
                     "owned by the work queue")

    # ---------------- Streaming trainers ----------------
    tx0, ty0, tx1, ty1 = TRAINERS
    box(ax, st, tx0, ty0, tx1, ty1, "trainer", ls=NODE_LS["trainer"])
    tr = spec.node("trainers")
    text(ax, tx0 + 3.0, ty1 - 3.4, tr.label, st.fs_title * 0.92, weight="bold", ha="left")
    _draw_spans(ax, st, layout.train_groups, cols, ty0 + 2.5, ty0 + 8.0,
                "trainer_chip")

    # ---------------- Edges ----------------
    e_dispatch, _e_migrate, e_push, e_pull, e_colo = spec.edges
    # _e_migrate is the engine-to-engine arc drawn inside the engines box above.

    # The connectors carry no text by default -- the boxes name the components
    # and the GPU lane shows the colocation, which is all the figure claims.
    # --edge-labels puts the spec's labels back for a draft/annotated render.
    def maybe_label(x, y_top, edge, **kw):
        if st.edge_labels:
            edge_label(ax, st, x, y_top, edge.label, edge.detail, **kw)

    # router -> engines
    arrow(ax, st, (X_DISPATCH, ry0), (X_DISPATCH, ey1), "dispatch")
    maybe_label(X_DISPATCH + 1.8, ry0 - 1.8, e_dispatch, ha="left")

    # engines -> work queue : the completed generations leave the engines' right
    # edge and drop into the queue's top edge.
    arrow(ax, st, (ex1, Y_ENG_OUT), (X_QUEUE_IN, qy1), "produce",
          connectionstyle="angle,angleA=0,angleB=90,rad=3")
    maybe_label((ex1 + X_QUEUE_IN) / 2 + 6.0, Y_ENG_OUT + 8.0, e_push)

    # work queue <-> trainers : pull-based, one double-headed elbow routed under
    # the queue.
    arrow(ax, st, (X_PULL, qy0), (tx1, Y_PULL), "consume", double=True,
          connectionstyle="angle,angleA=-90,angleB=0,rad=3")
    maybe_label((tx1 + X_PULL) / 2 + 3.0, Y_PULL - 1.6, e_pull)

    # engines / trainers : colocation, drawn as a shared GPU lane rather than
    # asserted with a label. Each column runs an unbroken band from the engine
    # chip above it down to the train-group chip below it, through a chip
    # carrying that GPU's physical id -- so "these two run on the same silicon"
    # is something the reader sees rather than reads.
    _draw_gpu_lane(ax, st, layout, cols, ey0, ty1, e_colo.label)

    # ---------------- Footnote ----------------
    # Just the topology, so the x8 / x4 counts in the box titles are grounded.
    text(ax, 3.0, 11.0, topo.describe(), st.fs_foot, ha="left", va="top",
         color="#3A3A3A")


def _draw_policy_box(ax, st: Style, node, x0, y0, x1, y1, ownership):
    """A composed policy object, rendered *inside* its owner.

    Containment rather than an arrow is the point: MigrationPolicy is a field of
    the router and GrabPolicy is constructed inside the work-queue actor, so
    drawing them as peers joined by an edge would misstate the design. The
    ``ownership`` caption names that relationship explicitly, since nesting alone
    can read as mere grouping.
    """
    box(ax, st, x0, y0, x1, y1, "policy", ls=NODE_LS["policy"], z=3, radius=1.2)
    text(ax, x0 + 1.6, y1 - 2.6, ownership, st.fs_note * 0.95, ha="left",
         color=st.accent("#8A5A2B"), style="italic")
    text(ax, (x0 + x1) / 2, (y0 + y1) / 2 - 1.5, node.label, st.fs_title * 0.85,
         weight="bold")


# Shrink chip fonts when columns get narrow, so a wide topology degrades into
# small-but-readable labels rather than text spilling over the chip borders.
# Calibrated on the canonical 8-column row, where a column is ~5.0 axis units.
CHIP_FS_PER_UNIT = 1.35


def _column_bounds(x_left, x_right, n_cols):
    """Equal-width GPU columns with a hairline gap. Returns (x0, x1) per column.

    Every row in the colocated stack -- engine chips, GPU lane, train-group chips
    -- is laid out against these same bounds. That shared geometry is what makes
    the figure's colocation claim literal: a chip sits over exactly the GPU
    columns it runs on.
    """
    span = x_right - x_left
    gap = span * 0.010
    w = (span - gap * (n_cols - 1)) / max(1, n_cols)
    return [(x_left + i * (w + gap), x_left + i * (w + gap) + w) for i in range(n_cols)]


def _draw_gpu_lane(ax, st: Style, layout, cols, y_top, y_bottom, caption):
    """The shared-GPU lane between the engine row and the train-group row.

    Per column: a tinted band spanning the whole gap (so the eye tracks a single
    piece of hardware from the engine chip down to the train-group chip) and, at
    its centre, a chip carrying that GPU's physical id.
    """
    band_face = st.colors("gpu_band")[0]
    chip_h = min(5.0, (y_top - y_bottom) * 0.42)
    cy = (y_top + y_bottom) / 2
    gy0, gy1 = cy - chip_h / 2, cy + chip_h / 2
    face, edger = st.colors("gpu")

    for i, gpu in enumerate(layout.columns):
        x0, x1 = cols[i]
        if gpu is ELIDED:
            text(ax, (x0 + x1) / 2, cy, "...", st.fs_chip, color="#555555")
            continue
        ax.add_patch(FancyBboxPatch(
            (x0, y_bottom), x1 - x0, y_top - y_bottom,
            boxstyle="round,pad=0,rounding_size=0.5",
            facecolor=band_face, edgecolor="none", zorder=1,
        ))
        ax.add_patch(FancyBboxPatch(
            (x0, gy0), x1 - x0, gy1 - gy0,
            boxstyle="round,pad=0,rounding_size=0.6",
            facecolor=face, edgecolor=edger, linewidth=st.lw_chip, zorder=3,
        ))
        text(ax, (x0 + x1) / 2, cy, str(gpu),
             min(st.fs_chip, CHIP_FS_PER_UNIT * (x1 - x0) * st.scale), z=5,
             color="#333333")

    # Caption goes *below* the GPU chips: the band above them is bounded by the
    # engines box, so a caption there would sit on its border.
    text(ax, cols[-1][1], gy0 - 1.6, caption, st.fs_edge_detail, ha="right",
         va="top", color="#444444", style="italic", z=6)


def _draw_spans(ax, st: Style, spans, cols, y0, y1, key):
    """Draw one chip per span across the columns it occupies."""
    face, edger = st.colors(key)
    for sp in spans:
        x0 = cols[sp.first_col][0]
        x1 = cols[sp.last_col][1]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if sp.is_elision:
            text(ax, cx, cy, "...", st.fs_chip, color="#555555")
            continue
        ax.add_patch(FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0,rounding_size=0.7",
            facecolor=face, edgecolor=edger, linewidth=st.lw_chip, zorder=3,
        ))
        text(ax, cx, cy, sp.label,
             min(st.fs_chip, CHIP_FS_PER_UNIT * (x1 - x0) * st.scale), z=5)


#: Drawn y-extent. Narrower than 0..100 so the saved figure crops to the content
#: instead of to the axes background -- bbox_inches="tight" counts the axes
#: patch, so an unused band at the bottom would ship as whitespace.
Y_LIM = (6.0, 98.0)


def build_figure(spec, width_in: float, grayscale: bool, edge_labels: bool = False):
    st = Style(width_in, grayscale, edge_labels)
    # Scale the height with the drawn y-extent so shapes keep their designed
    # proportions regardless of Y_LIM.
    height_in = golden_height(width_in) * (Y_LIM[1] - Y_LIM[0]) / 100.0
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    ax.set_xlim(0, 100)
    ax.set_ylim(*Y_LIM)
    ax.set_aspect("auto")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    draw(spec, st, ax)
    return fig


def _script_path() -> str:
    """This script's path, repo-relative when the caller is inside the repo.

    Prefer the relative form so the string stays meaningful after the PDF is
    copied into a paper repo or emailed around; fall back to absolute when the
    file is not under the working directory.
    """
    absolute = os.path.abspath(__file__)
    try:
        rel = os.path.relpath(absolute, os.getcwd())
    except ValueError:  # different drive on Windows
        return absolute
    return absolute if rel.startswith(os.pardir) else rel


def _provenance(argv, topo) -> tuple[dict, dict]:
    """(pdf_metadata, png_metadata) recording how this figure was produced.

    The PDF side is limited to the keys matplotlib's PDF backend accepts; the PNG
    side takes free-form tEXt chunks, so it gets an explicit ``Script`` key too.
    """
    script = _script_path()
    command = " ".join(["python", script, *argv])
    title = "slime streaming colocated RL - system architecture (data plane)"
    subject = f"Generated by {script}. Regenerate with: {command}"
    keywords = (
        f"{script}; {topo.describe()}; "
        "slime; streaming colocated RL; system architecture"
    )
    pdf_meta = {
        "Title": title,
        "Subject": subject,
        "Keywords": keywords,
        "Creator": script,
    }
    png_meta = {
        "Title": title,
        "Description": subject,
        "Software": script,
        "Script": script,
        "Command": command,
        "Topology": topo.describe(),
    }
    return pdf_meta, png_meta


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="paper_figures/out",
                   help="output directory (created if absent)")
    p.add_argument("--name", default="arch",
                   help="basename for the emitted files")
    p.add_argument("--num-gpus", type=int, default=8)
    p.add_argument("--train-tp", type=int, default=2)
    p.add_argument("--infer-tp", type=int, default=1)
    p.add_argument("--width", type=float, default=7.0,
                   help="figure width in inches (7.0 = two-column, 3.4 = single)")
    p.add_argument("--grayscale", action="store_true",
                   help="render without hue; distinctions ride on line style only")
    p.add_argument("--edge-labels", action="store_true",
                   help="annotate the connectors with the spec's edge labels "
                        "(off by default; useful for a draft render)")
    p.add_argument("--dpi", type=int, default=400)
    args = p.parse_args(argv)

    topo = Topology(total_gpus=args.num_gpus, train_tp=args.train_tp,
                    infer_tp=args.infer_tp)
    spec = build_spec(topo)

    os.makedirs(args.out, exist_ok=True)
    fig = build_figure(spec, args.width, args.grayscale, args.edge_labels)

    pdf_meta, png_meta = _provenance(
        list(argv) if argv is not None else sys.argv[1:], topo
    )

    stem = os.path.join(args.out, args.name)
    pdf, png = f"{stem}.pdf", f"{stem}.png"
    fig.savefig(pdf, format="pdf", bbox_inches="tight", pad_inches=0.02,
                metadata=pdf_meta)
    fig.savefig(png, format="png", dpi=args.dpi, bbox_inches="tight",
                pad_inches=0.02, metadata=png_meta)
    plt.close(fig)

    print(f"[arch] {topo.describe()}")
    print(f"[arch] wrote {pdf}  (Creator={pdf_meta['Creator']})")
    print(f"[arch] wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
