"""Declarative spec for the streaming-colocated-RL system architecture figure.

Single source of truth shared by both renderers:

  * ``make_arch_figure.py``  — matplotlib, explicit layout, camera-ready PDF/PNG
  * ``make_arch_dot.py``     — graphviz ``dot``, auto-layout, structural cross-check

Keeping the nodes/edges here means the two renderers cannot drift apart: if an
edge is missing from one figure and present in the other, that is a bug in a
renderer, not in the spec.

This module imports nothing but the standard library on purpose — it must stay
usable from either renderer (and from a future timeline/sequence figure).

Scope
-----
This is a **data-plane** figure. It deliberately omits the driver loop
(``train_streaming.py``), ``RayElasticGroup``, and the ``GroupSwitchController``:
those are control-plane objects and belong in a separate diagram. What is drawn:

    StreamingRouter (owns MigrationPolicy)
        -> SGLang inference engines
        -> StreamingWorkQueue (owns GrabPolicy)
             -> streaming trainers

Every edge carries a ``source`` field with the ``file:line`` it was read from, so
the figure stays auditable as the code changes. ``README.md`` renders these as a
provenance table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Topology
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Topology:
    """GPU/engine/train-group geometry.

    Mirrors the derivations the driver performs at ``train_streaming.py:99-109``
    (and, identically, ``slime/ray/elastic_actor.py:75-79``). Keeping the same
    arithmetic here means the figure's engine and trainer counts always match a
    real launch config rather than being hand-drawn.

    The evaluated benchmark config is ``total_gpus=8, train_tp=2, infer_tp=1``
    (see ``tests/streaming/test_streaming_8xGPU_tp_train2_tp_infer1_deepseek_r1_8b_
    10rollout_train_group_proactive_GRADUATED_BENCHMARK.py``), giving 8 engines
    and 4 train groups with 2 engines pinned to each train group's GPU pair.
    """

    total_gpus: int = 8
    train_tp: int = 2
    infer_tp: int = 1

    def __post_init__(self) -> None:
        if self.total_gpus <= 0:
            raise ValueError(f"total_gpus must be > 0, got {self.total_gpus}")
        for name, tp in (("train_tp", self.train_tp), ("infer_tp", self.infer_tp)):
            if tp <= 0:
                raise ValueError(f"{name} must be > 0, got {tp}")
            if self.total_gpus % tp != 0:
                raise ValueError(
                    f"{name}={tp} must divide total_gpus={self.total_gpus}"
                )
        if self.train_tp % self.infer_tp != 0:
            raise ValueError(
                f"infer_tp={self.infer_tp} must divide train_tp={self.train_tp} "
                "(engines_per_train_group must be a whole number)"
            )

    @property
    def num_train_groups(self) -> int:
        """== the training DP size (``train_streaming.py:110``)."""
        return self.total_gpus // self.train_tp

    @property
    def num_infer_engines(self) -> int:
        return self.total_gpus // self.infer_tp

    @property
    def engines_per_train_group(self) -> int:
        return self.train_tp // self.infer_tp

    def engines_for_train_group(self, group: int) -> list[int]:
        """``slime/ray/elastic_actor.py:226-230``."""
        start = group * self.engines_per_train_group
        return list(range(start, start + self.engines_per_train_group))

    def train_group_for_engine(self, engine: int) -> int:
        """``slime/router/streaming_router.py:101-102``, and the identical
        rollup in ``slime/ray/streaming_work_queue.py:120``."""
        return engine // self.engines_per_train_group

    def gpus_for_engine(self, engine: int) -> list[int]:
        """``slime/ray/elastic_actor.py:232-242``."""
        start = engine * self.infer_tp
        return list(range(start, start + self.infer_tp))

    def gpus_for_train_group(self, group: int) -> list[int]:
        """``slime/ray/elastic_actor.py:244-252``."""
        start = group * self.train_tp
        return list(range(start, start + self.train_tp))

    def describe(self) -> str:
        return (
            f"{self.total_gpus} GPUs, train_tp={self.train_tp}, infer_tp={self.infer_tp} "
            f"-> {self.num_infer_engines} engines, {self.num_train_groups} train groups "
            f"({self.engines_per_train_group} engine(s) per train group)"
        )


# --------------------------------------------------------------------------
# Nodes and edges
# --------------------------------------------------------------------------

# Node kinds drive fill/border style. The Ray-actor vs in-process-object
# distinction is load-bearing and is the thing most often drawn wrong: the
# StreamingRouter is NOT a Ray actor and NOT an HTTP server -- it is a plain
# Python object living inside the StreamingRolloutManager actor
# (slime/router/streaming_router.py:6-9).
NODE_KINDS = ("inproc_obj", "ray_actor", "policy", "engine", "trainer")

# Edge kinds drive stroke style.
EDGE_KINDS = ("dispatch", "http_migrate", "produce", "consume", "colocate")


@dataclass(frozen=True)
class Node:
    id: str
    label: str
    kind: str
    sublabel: str = ""
    #: When set, this node renders *inside* the named node (composition, not an
    #: arrow). Used for MigrationPolicy-in-router and GrabPolicy-in-queue.
    owner: str | None = None
    #: Extra annotation lines rendered small inside the box.
    notes: tuple[str, ...] = ()
    source: str = ""

    def __post_init__(self) -> None:
        if self.kind not in NODE_KINDS:
            raise ValueError(f"unknown node kind {self.kind!r} on node {self.id!r}")


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    label: str
    kind: str
    source: str = ""
    #: Secondary lines under the main label, rendered smaller.
    detail: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in EDGE_KINDS:
            raise ValueError(
                f"unknown edge kind {self.kind!r} on edge {self.src}->{self.dst}"
            )


@dataclass
class ArchSpec:
    topology: Topology
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def node(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    def children_of(self, node_id: str) -> list[Node]:
        return [n for n in self.nodes if n.owner == node_id]

    def validate(self) -> None:
        ids = [n.id for n in self.nodes]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate node ids: {sorted(dupes)}")
        known = set(ids)
        for n in self.nodes:
            if n.owner is not None and n.owner not in known:
                raise ValueError(f"node {n.id!r} has unknown owner {n.owner!r}")
        for e in self.edges:
            for endpoint in (e.src, e.dst):
                if endpoint not in known:
                    raise ValueError(
                        f"edge {e.src}->{e.dst} references unknown node {endpoint!r}"
                    )

    def edge_keys(self) -> list[tuple[str, str, str]]:
        """Renderer-independent identity of the edge set.

        ``make_arch_dot.py --check`` diffs this against what it drew, so a spec
        edge silently dropped by one renderer is caught rather than eyeballed.
        """
        return sorted((e.src, e.dst, e.label) for e in self.edges)


# --------------------------------------------------------------------------
# The spec itself
# --------------------------------------------------------------------------

def build_spec(topology: Topology | None = None) -> ArchSpec:
    """Assemble the architecture spec for a given topology.

    Text is deliberately sparse: boxes carry the component name, edges carry the
    call that crosses them, and nothing else. Detail that belongs in prose does
    not belong on the figure -- the provenance table in README.md is where the
    fuller story lives.
    """
    topo = topology or Topology()

    nodes: list[Node] = [
        Node(
            id="router",
            label="StreamingRouter",
            kind="inproc_obj",
            source="slime/router/streaming_router.py:40",
        ),
        Node(
            id="migration_policy",
            label="MigrationPolicy",
            kind="policy",
            owner="router",
            source="slime/router/migration_policy.py:104",
        ),
        Node(
            id="engines",
            # No count in the label: the figure is not tied to one topology, and
            # the chip row plus the footnote already say how many there are.
            label="SGLang inference engines",
            kind="engine",
            source="slime/ray/elastic_actor.py:205-222",
        ),
        Node(
            id="work_queue",
            label="StreamingWorkQueue",
            kind="ray_actor",
            # The one caption that earns its place: it names what the pending
            # slots drawn inside the box actually hold.
            sublabel="pending prompt groups",
            source="slime/ray/streaming_work_queue.py:24",
        ),
        Node(
            id="grab_policy",
            label="GrabPolicy",
            kind="policy",
            owner="work_queue",
            source="slime/ray/grab_policy.py:50",
        ),
        Node(
            id="trainers",
            label="Streaming trainers",
            kind="trainer",
            source="slime/backends/megatron_utils/streaming_actor.py:34",
        ),
    ]

    edges: list[Edge] = [
        Edge(
            src="router",
            dst="engines",
            label="prompt groups",
            kind="dispatch",
            source="slime/router/streaming_router.py:124-134, 204-208",
        ),
        # Migration is engine-to-engine work movement: an in-flight group is
        # taken off a lagging engine and re-prefilled on a still-busy one. The
        # router mediates it, but the work moves sideways -- so it renders as an
        # arc inside the engine row, not as another arrow down from the router.
        Edge(
            src="engines",
            dst="engines",
            label="migrate",
            kind="http_migrate",
            source="slime/router/streaming_router.py:265-376",
        ),
        # Drawn engines -> queue because that is where the data comes from: the
        # completed generations. The router is what actually calls push_data();
        # the detail line keeps that honest without adding a second hop.
        Edge(
            src="engines",
            dst="work_queue",
            label="completed prompt groups",
            kind="produce",
            detail=("push_data(), via the router",),
            source="slime/router/streaming_router.py:428-430, 446, 356, 469",
        ),
        # Pull-based: the trainer calls, the queue answers. Drawn as one
        # double-headed connector rather than two arrows -- two parallel arrows
        # between the same pair of boxes reads as two mechanisms, and there is
        # only one.
        Edge(
            src="work_queue",
            dst="trainers",
            label="chunk",
            kind="consume",
            detail=("grab_available(), trainer pulls",),
            source="slime/backends/megatron_utils/streaming_actor.py:522-527, 650-658",
        ),
        Edge(
            src="engines",
            dst="trainers",
            label="shared GPUs",
            kind="colocate",
            source="slime/ray/streaming_work_queue.py:120-124",
        ),
    ]

    spec = ArchSpec(topology=topo, nodes=nodes, edges=edges)
    spec.validate()
    return spec


# --------------------------------------------------------------------------
# Sub-item layout helper (shared by both renderers)
# --------------------------------------------------------------------------


def engine_slots(topo: Topology, max_shown: int = 8) -> list[str]:
    """Labels for the individual engine chips, elided if the topology is large.

    Elision keeps a 16- or 32-GPU config legible. The colocation band is drawn
    per train group rather than per engine, so eliding engine chips does not
    break the engine <-> trainer alignment.
    """
    n = topo.num_infer_engines
    if n <= max_shown:
        return [f"E{i}" for i in range(n)]
    head = max_shown - 2
    return [f"E{i}" for i in range(head)] + ["...", f"E{n - 1}"]


def trainer_slots(topo: Topology, max_shown: int = 8) -> list[str]:
    n = topo.num_train_groups
    if n <= max_shown:
        return [f"G{i}" for i in range(n)]
    head = max_shown - 2
    return [f"G{i}" for i in range(head)] + ["...", f"G{n - 1}"]


#: Above this many train groups the GPU lane elides its middle. Chosen so the
#: canonical 8-GPU config (4 train groups) is never elided while a 32-GPU config
#: still leaves columns wide enough to read.
MAX_SHOWN_GROUPS = 6

#: Marks the elided column in a GPU layout.
ELIDED = None


@dataclass(frozen=True)
class Span:
    """A chip occupying GPU columns ``[first_col, last_col]`` inclusive."""

    label: str
    first_col: int
    last_col: int

    @property
    def is_elision(self) -> bool:
        return self.label == "..."


@dataclass(frozen=True)
class GpuLayout:
    """One shared column per physical GPU, with what sits on it above and below.

    This is what makes colocation drawable rather than merely asserted: an engine
    and a train group that occupy the same GPUs occupy the same columns, so the
    figure can put them in one vertical stack over a labelled GPU lane.

    ``columns`` holds physical GPU ids in display order; an ``ELIDED`` entry marks
    where the middle was dropped for a large topology.
    """

    columns: list[int | None]
    engines: list[Span]
    train_groups: list[Span]


def gpu_layout(topo: Topology, max_groups: int = MAX_SHOWN_GROUPS) -> GpuLayout:
    """Build the shared-GPU column layout for ``topo``.

    Columns are enumerated train group by train group, so a train group is always
    a contiguous run and the elision (when one is needed) falls on a group
    boundary rather than splitting a group in half.
    """
    groups = list(range(topo.num_train_groups))
    if len(groups) > max_groups:
        head = max_groups - 2
        shown: list[int | None] = [*groups[:head], ELIDED, groups[-1]]
    else:
        shown = list(groups)

    columns: list[int | None] = []
    train_groups: list[Span] = []
    for g in shown:
        if g is ELIDED:
            columns.append(ELIDED)
            train_groups.append(Span("...", len(columns) - 1, len(columns) - 1))
            continue
        start = len(columns)
        columns.extend(topo.gpus_for_train_group(g))
        train_groups.append(Span(f"G{g}", start, len(columns) - 1))

    col_of = {gpu: i for i, gpu in enumerate(columns) if gpu is not ELIDED}
    engines: list[Span] = []
    for e in range(topo.num_infer_engines):
        gpus = topo.gpus_for_engine(e)
        if not all(g in col_of for g in gpus):
            continue  # this engine sits in the elided middle
        idxs = [col_of[g] for g in gpus]
        engines.append(Span(f"E{e}", min(idxs), max(idxs)))
    # Re-insert the elision marker between the engines that straddle it.
    for i, col in enumerate(columns):
        if col is ELIDED:
            pos = sum(1 for s in engines if s.last_col < i)
            engines.insert(pos, Span("...", i, i))

    return GpuLayout(columns=columns, engines=engines, train_groups=train_groups)


def golden_height(width_in: float) -> float:
    """Default figure height for a given width."""
    return width_in / math.sqrt(2.0)
