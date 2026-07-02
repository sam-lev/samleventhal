A persistence diagram looks intimidating the first time you see it: a scatter of dots above a diagonal line, each one supposedly meaningful. But the reading rule is simple, and once it clicks the whole picture opens up.

## The one rule

Every dot is a feature — a connected component, a loop, a void — that appears at some scale and disappears at another. The horizontal axis is **birth** (when the feature shows up) and the vertical axis is **death** (when it merges away or fills in).

> Distance from the diagonal is the whole story. A dot far above the line lived for a long time across scales; a dot hugging the line blinked in and out.

That distance is what we usually call *persistence*. Long-lived features are the structure; short-lived ones are mostly noise.

## What the dots are not

A common misread is to treat a high-persistence dot as a single, locatable object in the data. It isn't — it's a *topological* summary across a whole filtration. Two datasets that look very different can produce nearly identical diagrams, which is exactly what makes them useful as a stable signature.

## A tiny code sketch

If you want to generate one yourself, the usual starting point is a Vietoris–Rips filtration:

```python
import numpy as np
from ripser import ripser

points = np.random.rand(200, 2)
result = ripser(points, maxdim=1)
diagrams = result["dgms"]   # H0 and H1
```

Each entry in `diagrams` is an array of `(birth, death)` pairs — precisely the dots you'd plot.

## Reading the dimensions

The diagram is usually split by homological dimension:

| Dimension | Counts | Intuition |
| --- | --- | --- |
| H₀ | Connected components | How the data clusters |
| H₁ | Loops | Holes you could lasso |
| H₂ | Voids | Enclosed empty regions |

A clean way to start is to ignore everything near the diagonal, look at the two or three dots that stand apart, and ask what scale they were born and died at. That's almost always where the interesting structure is.

---

*This is a sample post. Replace it with your own — the file lives at `posts/field-notes-persistent-homology.md`.*
