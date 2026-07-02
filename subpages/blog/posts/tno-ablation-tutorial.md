I put together a self-contained, runnable tutorial walking through the central experiment behind **Topological Neural Operators (TNOs)** — a paper-style notebook you can read top-to-bottom and then execute cell by cell.

The motivating question is easy to state and awkward to answer: *does the Hodge decomposition give a neural operator the right inductive bias for solving PDEs on domains with holes?* A TNO represents physical fields as cochains on a combinatorial complex and routes message passing through the gradient, curl, and harmonic channels of the discrete Hodge decomposition — and that harmonic channel exists *only* because the domain has holes (its dimension is the first Betti number, β₁).

## What the notebook does

It develops just enough discrete exterior calculus to make the layer precise, implements the paper's per-layer update, and then runs a controlled ablation. A strict capability hierarchy — from an MLP, up through a geometry-conditioned GNN, to a family of TNO variants that each flip a single channel or design choice on or off — is trained across a ladder of domains with increasing topological complexity (β₁ = 0, 1, 1, 2, …). Because each variant isolates one capability, any change in test error can be attributed to that capability.

## Opening the black box

The final sections read out the network's *own* learned channel gates and then use per-Hodge-channel, gradient-based attribution to ask, mechanistically, whether the harmonic channel is really attending to the holes that define each domain's topology — with confound controls built in so the analysis doesn't fool itself.

[Read the full tutorial →](https://samleventhal.com/subpages/TNO/Gradient_Ablation_Tutorial)
