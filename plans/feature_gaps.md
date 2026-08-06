# Feature Gaps

> **Status: proposed, not scheduled.** None of these block v1.0. The v1.0 plan
> is explicit that the remaining work is hardening, not new features, so treat
> this file as the backlog that opens *after* 1.0 ships.

Cellestial covers every plot family it set out to cover. These four are the
gaps that a user coming from scanpy or scvelo can still notice. They are
ordered by how much of the field they unlock, not by effort.

Composition and proportion bars are deliberately absent from this list.
`cl.bar(data, mapping=aes('sample', fill='leiden'), position='fill')` already
works, because `bar` forwards `**geom_kwargs` to `geom_bar`.

## 1. Trajectory and pseudotime

The one missing plot family. RNA velocity is covered by `cl.stream()`, but
there is nothing for trajectory topology or for expression ordered along
pseudotime.

Two plots, and they are unrelated to each other:

- **Graph over an embedding.** Nodes at per-cluster embedding centroids, edges
  weighted by connectivity, drawn as a layer on top of an existing dimensional
  plot rather than as a standalone function. `cluster_outlines` and
  `ondata_legend` already compute per-cluster centroids on the built frame, so
  the node placement is solved; the new part is the edge frame.
- **Expression along pseudotime.** Pseudotime on x, expression on y, one line
  or band per gene, grouped or faceted. Structurally this is `xyplot` with a
  smoother, so the question is whether it earns a name of its own or is
  documented as a recipe.

Open questions:

- Where the connectivity comes from. Reading a stored graph keeps the
  backend-agnostic rule intact if the lookup lives in a helper under
  `if isinstance(data, AnnData):`, the same shape `_get_dendrogram` uses.
- Whether the layer form conflicts with `DeferredLayer`, which reads the built
  frame after the plot call. A trajectory layer needs cluster identity and
  embedding coordinates, both of which the frame already carries, so it
  probably fits the deferred pattern directly.

## 2. `tracksplot`

The heatmap family has every scanpy sibling except this one:
`heatmap`, `dotplot`, `matrixplot`, `stacked_violin`, `annotated_heatmap`.

Stacked per-gene track panels sharing an x axis of cells ordered by group, with
the group bar underneath. The cheapest parity win in the repo: the marker-gene
resolution, the group ordering, the dendrogram, and the shared-axis grid
machinery are all already written and used by the four siblings.

Lands in `cellestial/single/heatmap/`, exported from that package's `__init__`
and from `cellestial/__init__.py` alongside the rest.

Open question: the vertical stack is `gggrid` with `align=True`, which is the
same layout `annotated_heatmap` fought with. Read the annotated_heatmap
overflow notes in [MEMORY.md](../MEMORY.md) before assuming the alignment,
the shared axis, or the legend placement is free.

## 3. Embedding density

No equivalent to `sc.pl.embedding_density`. Fits the architecture as a **layer**
over an existing dimensional plot, not as a new plot function: it reads the
embedding columns off the built frame exactly as `stream` reads velocity
embeddings, so it should be a `DeferredLayer`.

Two variants worth deciding between before writing anything:

- Contours over all cells, one set per group. Cheap, and lets-plot draws it.
- A filled density per group computed on that group's cells but evaluated on
  the full embedding extent, which is what scanpy actually does and what makes
  the panels comparable across groups. More work, and the normalization choice
  is user-visible.

Do not auto-promote a `group_by` into the density aesthetic. The user wires it,
consistent with the rest of the library.

## 4. Co-expression blend

Two genes on one embedding, each mapped to a color channel, blended per cell.
Common ask, and there is no current path to it because every gradient scale in
the library maps a single continuous column.

This one is the least certain of the four and may not belong in Cellestial at
all. It needs a per-cell RGB value computed outside the grammar and handed to
lets-plot as a literal color column, which means the resulting plot has no
usable legend and no scale the user can modify with a `+ scale_*` call. That
breaks the modularity principle in a way the other three do not.

If it is built, the honest form is a small frame helper that returns the blended
color column and a companion 2D legend, leaving the user to call `scatter` or
`dimensional` themselves. Not a plot function.
