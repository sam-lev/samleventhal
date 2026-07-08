This is one of the earliest pieces of my topology-and-learning work — a 2019 project that asked a question I've kept circling back to ever since: *can a topological summary of an image do the labor a human annotator usually does?* The [full write-up is here](https://github.com/sam-lev/Topological_UNet/blob/master/UNsup_MSC_UNET_MSC_GRAPH_LEARNING.png), and the [code is on GitHub](https://github.com/sam-lev/Topological_UNet).

The setting is blood-vessel segmentation in retinal images (the DRIVE and STARE datasets), but the segmentation target is only ever an excuse. The real object of study is the **Morse–Smale complex (MSC)** — a topological decomposition of an image treated as a scalar field — and whether it can stand in for the hand-drawn ground truth that supervised segmentation depends on. Building those labels by hand is slow, expensive, and exactly the bottleneck you'd want to remove.

I attacked it from two directions at once.

## Reading the MSC as a graph

The first half is a graph-learning problem. Compute the MSC over an image and you get a network of *arcs* — 1-cells that trace ridge lines between critical points, following the bright filaments that, in a retinal scan, happen to be vessels. Hand-label a small subset of those arcs as vessel or background, and the question becomes: can a model learn the rest?

Because each arc covers many pixels, it carries a rich feature vector — per-arc statistics (min, max, mean, a Gaussian fit, standard deviation) computed from the pixels beneath it. To make those features more robust I compute the MSC once on the original image, then project the same arcs onto a stack of filtered versions (difference-of-Gaussians, Sobel, Laplace, structure, blur) and re-sample the same statistics under each. It's essentially topology-guided data augmentation: the arc geometry is fixed by the original image, but its feature description is enriched by every filter.

This is naturally an *inductive* problem — you want to train on one region and generalize to unseen arcs — so I built on **GraphSAGE**, which learns to aggregate neighbor features rather than memorizing a fixed graph. Training runs on the **dual** of the MSC (arcs become nodes, so the richer features live where the learning happens), with an unsupervised loss over node pairs drawn from random walks and Centroidal Voronoi Tessellation sampling to spread the training seeds across the complex. A two-layer mean-pool aggregator with 16-sample neighborhoods and 32-dimensional embeddings, followed by linear regression on the embeddings, classifies the remaining arcs. On a held-out test of the entire image and MSC, it lands an **F1 of 0.79**.

The more sobering result came from an ablation. I pitted pure graph learners against plain classifiers on the same arc features:

| Approach | F1 |
|---|---|
| CAMLP (label propagation) | 0.687 |
| Random Forest + DeepWalk embeddings | 0.647 |
| Random Forest + raw arc features | 0.909 |
| Random Forest + DeepWalk + features | 0.913 |
| Random Forest + locally filtered features | 0.830 |

The graph machinery buys almost nothing here — a random forest on the raw features alone beats it, and adding DeepWalk embeddings on top improves things by a mere 0.004. The honest reading is that this segmentation task is *geometric*, not relational: the discriminative signal lives in each arc's own feature vector, and label-propagation-style similarity between neighbors doesn't have much left to contribute. That conclusion is a big part of why my later work moved toward genuinely geometric graph learning rather than neighborhood-similarity methods.

## Letting the MSC teach a U-Net

The second half is the one the project is named for, and the more ambitious idea: skip hand-labels entirely and train a **U-Net** on MSC segmentations as pseudo-ground-truth.

The catch is that "the MSC of an image" isn't a single thing — it depends on **persistence**, the threshold that decides how prominent a feature must be to survive. Low persistence captures fine tangential vessels but keeps noise; high persistence keeps only the trunk. Rather than pick one, I hand the U-Net *several* — segmentations at persistence 7, 10, 12, 15, 20, and 23 — stacked as multiple training targets per image.

This is where it gets interesting, and where it breaks in an instructive way. A pixel on a fine vessel might be labeled positive at low persistence and vanish at high persistence, so the network sees contradictory targets for the same location. Its response is to hedge — producing a soft, blurred prediction with a visible "halo" around the vessels. When the loss leans too hard on the MSC's closed-cell structure, the network mimics the *complex* instead of segmenting the *vessel*. The result: **F1 of 0.46** on test, with training that overfits (validation loss creeps up while training loss keeps falling), and a sharp collapse to 0.28 if you drop the low-persistence segmentations that carry the thin vessels.

That's not a strong segmentation score, and I don't want to dress it up as one. But the failure mode is the actual finding. It says the network is learning the geometry of the MSC rather than the anatomy of the vessel, and it points cleanly at the fix I proposed in the conclusion: a **spatially weighted loss** that down-weights pixels according to their instability along the MSC — telling the network, per pixel, how much to trust the topological label and how much to look past it. Let the "forgettable" pixels be forgotten, and the ground truth stops being a cage.

## Why I still think about this one

Two takeaways have aged well. First, **topology is a fantastic organizer of an image but a mediocre feature space on its own** — the MSC gives you the right *structure* to reason over, but the discriminative power was in the geometry all along, which reframed how I think about where topological priors actually help. Second, **unsupervised labels are only as good as your ability to tell the model which of them to ignore** — a persistence-varying MSC is a source of free supervision *and* a source of contradiction, and the whole game is managing the second to keep the first.

It's rough around the edges, and it's very much a snapshot of where my thinking started. But the questions it raised — how to marry topological computation with learning, and how to let a network keep what's useful in a topological prior while discarding what isn't — are the ones I'm still chasing.

*The paper and full code, including the C++ Morse–Smale library, the `topoml` graph-learning package, and the unsupervised U-Net notebook, are [on GitHub](https://github.com/sam-lev/Topological_UNet).*
