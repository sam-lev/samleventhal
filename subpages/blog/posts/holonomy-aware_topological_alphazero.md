# Holonomy-Aware Topological AlphaZero

*Geodesics* takes the rules at their word and plays them on other surfaces:
geodesic and Goldberg spheres, tori, cylinders, Möbius bands, Klein bottles,
the real projective plane, and lattices in three dimensions — tiled with
triangles, squares, or hexagons. The rules transfer verbatim. What does *not*
transfer is the machinery that modern Go AIs are built from.

To the best of what the literature shows, the intersection this occupies —
agents, on non-orientable and arbitrary-genus generalized regular and irregular meshes, in settings of high heterophily, with
reflection-aware equivariant message passing — was simply empty before. The
full architecture, proofs, and verification are written up in the paper
([PDF](https://samleventhal.com/papers/Holonomy_Aware_Topological_AlphaZero.pdf)).

## Why the strong engines can't come along

AlphaZero and KataGo are convolutional neural networks. A convolution is an
operation on a rectangular grid of pixels: it slides a fixed stencil over rows
and columns. That stencil is the entire reason those networks are strong and
efficient — and it is exactly what stops them at the edge of the plane. You
cannot lay a grid of rows and columns over a sphere without a seam or a
singularity; a Möbius band has no consistent "up"; a Klein bottle doesn't
embed in ordinary space at all. There is no rectangular tensor to hand the
network. The board can't even be represented as input, let alone played well.

The usual trick these engines lean on makes the mismatch concrete: AlphaZero
augments its training data with the 8 symmetries of the square (4 rotations ×
2 reflections). That group is baked into the architecture's worldview. But a
geodesic sphere has 60 rotational symmetries, a torus has a continuous family
of translations, and a Klein bottle's symmetries include *orientation-
reversing* ones with no analogue on any flat board. Train the square-board
trick on a sphere and it is simply wrong.

There's a deeper obstruction, too, and it's the one that shaped the
architecture. On a non-orientable surface — Möbius, Klein, RP² — you cannot
even consistently define "clockwise." Carry a little coordinate frame once
around the band and it comes back mirror-flipped. Recent work in geometric
deep learning (Weiler and collaborators) proves this isn't a nuisance to be
smoothed over: any network that processes such a surface consistently *must*
be built to be steerable under reflections. A model that ignores it develops
a discontinuous seam — it plays one way on one side of the twist and a
contradictory way on the other. Standard Go networks have no concept of this
because on a flat board it never arises.

## What HATZ does instead

HATZ — Holonomy-Aware Topological AlphaZero — keeps the AlphaZero *method*
(self-play, a policy-and-value network guided by Monte-Carlo tree search) but
replaces the convolutional network with one built from the ground up for
arbitrary topology. Four ideas do the work.

**It reads the board as a cell complex, not a picture.** Messages pass along
the graph's own incidence structure — vertices, edges, and faces — so the
network's operations are defined by the board's connectivity rather than by
rows and columns. This is what lets one network play a triangular sphere and a
hexagonal torus and a cubic lattice without change.

**It carries the orientation structure the mathematics demands.** From the
board's combinatorics alone, HATZ computes where the orientation "seam" lives
— the edges you cross to arrive mirror-flipped — and passes information across
those edges through a different, reflection-aware transported map. Crossing the
twist of a Möbius band genuinely changes how the network reasons, which is the
point. (This is the discrete, reflection part of the full continuous structure
— the honest scope of a first version — and the transported maps are kept
orthogonal so that "transport" really means a rigid change of frame rather than
arbitrary mixing.)

**It is exactly equivariant to each board's real symmetries.** Rather than
augmenting data with the square's 8 symmetries, HATZ is built so that whatever
symmetries a particular board actually has, the network respects them exactly
— verified to machine precision, including the seam-preserving reflections of
the Möbius band. Symmetry that would otherwise have to be learned from scratch
is free, which matters enormously for sample efficiency in self-play.

**It pools through its own sense of territory.** Strong play needs to reason at
the scale of whole regions, not just individual points. HATZ predicts an
ownership map — who is likely to control each point — and then coarsens the
board along the connected regions of that prediction: likely-Black territory,
contested frontier, likely-White territory. Because the pooling follows a
quantity the network is *supervised* to get right, it sharpens as the model
improves, rather than being a fixed or arbitrary grouping. Alongside this, a
learned filtration over edges — supervised by whether two adjacent points end
up owned by the same player — lets the network attend at several scales at
once, drawing on a companion line of work on topological filtration learning.

The whole thing is about 38,000 parameters and depends on nothing but a small
numerical library — small enough that the lighter models in the same project
run directly in the browser, and HATZ itself plays through a local bridge.

## Where this sits, and what's honest about it

To the best of what the literature shows, the intersection this occupies —
game-playing agents, on non-orientable and arbitrary-genus boards, with
reflection-aware equivariant message passing — was simply empty before. There
was no prior agent to inherit from, which is the appeal and also the catch:
there are no baselines, so the model's own progression and a battery of
correctness checks are the yardstick for now.

And there are honest edges. This first version implements the reflection part
of the surface's structure but not yet the full continuous rotational part; its
gauge-consistency is trained in rather than guaranteed; and while it plays
every surface, it is not yet a *strong* player — reaching real strength on the
larger boards is a question of training compute and search depth, which on a
laptop is the genuine wall. What it does establish is that the architecture is
sound: the symmetries hold exactly, the seam is handled where the mathematics
says it must be, and the gradients are all verified. The board was never the
point — and now, at last, the player isn't tied to the grid.