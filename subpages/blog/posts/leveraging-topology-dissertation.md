*My PhD dissertation, [**Leveraging Topology to Advance Machine Learning Models and Methods**](https://samleventhal.com/papers/Dissertation_Leventhal.pdf) (University of Utah, Kahlert School of Computing, December 2024), is finished and online. This post is a short tour of what it argues and why. The full document is [here as a PDF](https://samleventhal.com/papers/Dissertation_Leventhal.pdf).*

## The one-sentence version

Most attempts to combine topology with machine learning use topology as a *feature extractor*: compute a persistence diagram, a Betti curve, or some other summary, vectorize it, and hand it to a model that lives in ordinary Euclidean space. My dissertation takes a different route — it keeps the learning problem **entirely within the topological domain**. The topological structure isn't a preprocessing step feeding a pixel model; it *is* the object the model learns over.

Concretely, the recurring construction is a **learnable, model-agnostic graph** built from the cells of a Morse–Smale complex, one that records not just which cells are adjacent but how they connect across scales. Once a problem is posed in those terms — "classify the cells of this complex" rather than "classify these pixels" — segmentation and node classification both become graph-learning problems with topological priors baked into the connectivity itself.

## Why topology, and why stay inside it

Increasingly, the data worth learning from is naturally described by *discrete elements and their relationships* rather than by dense grids of numbers. When that's true, throwing the structure away to fit a pixel- or vector-shaped model is a real cost: you spend model capacity and training data relearning geometry you already knew.

The **Morse–Smale complex (MSC)** is the workhorse here. Given a scalar field, its integral lines partition the domain into cells of uniform gradient-flow behavior — maxima, minima, saddles, and the ascending/descending arcs between them. Those cells frequently *are* the semantic features a scientist cares about: valley- and ridge-like structures, junctions in a foam, vessels in a retinal scan, neuronal processes in a microscopy volume. Discrete Morse theory makes the MSC computable on real, noisy, grid-sampled data, and **persistence** gives a principled dial for simplifying away discretization and noise artifacts to recover coarse-scale behavior.

The dissertation's bet is that if these cells are the features, then the learning should happen on the cells directly.

## Segmentation as classifying topological cells

The first main contribution reframes image segmentation as **classifying arcs of a Morse–Smale complex** instead of labeling individual pixels. From the MSC I build a *priors graph*: each topological prior (an arc between non-degree-2 vertices) becomes a node, with edges between incident priors. A model trained over that graph segments an image by deciding which priors belong to the object.

Two things fall out of this that pixel-level methods don't get for free:

- **Efficiency and data economy.** Because the priors graph is dramatically smaller than the pixel grid and already encodes the object's structure, the approach reaches accuracy comparable to pixel-level segmentation with *marginal training data* and faster execution. Across the Neuron, Retinal, and Foam datasets, learning on priors is competitive with pixel baselines while training on a small fraction of the annotation.
- **A genuinely interactive workflow.** Labeling arcs is far cheaper than painting pixels — a handful of clicks with a shortest-path or free-form-stroke tool can annotate structures that would take hundreds of pixel-level strokes. I built an interactive tool around this: choose a persistence threshold so the MSC covers the object, label a few priors, train, infer over the rest, then *correct the model's mistakes and retrain*. The corrected labels become a more robust training set, and the loop tightens. That workflow is meant to be usable by practitioners in medical imaging, neuroscience, and materials science, not just by topologists.

## Hierarchy: learning across scales of connectivity

Persistence doesn't give you one complex — it gives you a *sequence* of increasingly simplified ones. The second contribution turns that sequence into a learning signal.

I define the **hierarchical priors graph (HPG)**: a sequence of graphs built from Morse–Smale complexes at increasing levels of simplification. The **reduced hierarchical priors graph (RHPG)** trims this to the arcs relating cells across resolution levels. On top of the RHPG come two training ideas:

- **Hierarchical Successive Training (HST)** — train a GNN sequentially on the different prior graphs, so it learns from multiple scales of connectivity in turn rather than from a single fixed neighborhood.
- **Hierarchical Joint Training (HJT)** — a message-passing scheme that passes messages *within* each prior graph and *across* graphs of different scales. The novel piece is **across-neighborhood aggregation**: information from cells of arbitrary rank and dimension, combined across the whole multi-scale sequence, rather than being confined to the geometric neighborhoods of a single complex.

The payoff is that a node surviving the filtration is common to every graph in the sequence but sits in a different neighborhood at each level — so the model gets both local detail and global context from the same target cell.

## Filtration learning for graph neural networks

The last contribution steps back from imaging to a general GNN problem: **heterophily**, the setting where connected nodes tend to have *different* labels. Standard message-passing GNNs, which assume connected nodes are similar, degrade badly here and also suffer from oversmoothing as they deepen.

The dissertation treats a graph as a **simplicial complex** and learns from the class relationships between simplices, using **persistence filtration** to produce an ordered sequence of nested subgraphs that expose local and global structure at once. Learning over this multi-scale sequence (the MsST and MsJT methods) yields a hierarchical GNN that is resilient to oversmoothing and stays competitive across the whole homophily–heterophily spectrum — outperforming standard GNNs, heterophily-specific architectures, and other topological-deep-learning baselines on most of the datasets tested, in both accuracy and efficiency.

## The thread

Read together, the chapters make one argument from three angles: **topological priors are most useful when the model learns on them directly.** A Morse–Smale complex gives you the cells; a priors graph turns those cells into something a network can reason over; a hierarchy of such graphs, tied together by persistence, lets the network learn across scales; and the same filtration idea generalizes to hard graph problems that have nothing to do with images. Throughout, the aim was frameworks that are *accessible and useful* — interactive tools, model-agnostic structures, and methods that pay their way in training cost — rather than topology for its own sake.

If any of this is relevant to your work, the full dissertation — background, methods, experiments, and all the figures and tables — is available here:

**[Leveraging Topology to Advance Machine Learning Models and Methods (PDF)](https://samleventhal.com/papers/Dissertation_Leventhal.pdf)**

*Committee: Valerio Pascucci (chair), Christopher R. Johnson, Dave Pugmire, Bei Wang Phillips, and Aditya Bhaskara.*
