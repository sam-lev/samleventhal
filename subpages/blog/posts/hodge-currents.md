*A Hodge-theoretic view of molecular dynamics.*

A small, runnable study of what the *shape* of a dynamical system tells you about
the way it moves. Everything here is plain NumPy and SciPy, runs on a laptop in
well under a minute, and is meant to be read top to bottom.

The premise is simple. When you watch a molecule (or any overdamped system) hop
between states, two different questions tend to come up, and they are usually
answered with two different toolkits:

- *Which way does reactive flux flow, and how committed is a given configuration
  to one end-state or the other?* That is the committor, and the machinery of
  transition path theory.
- *Is that flow conservative, or does it circulate?* At equilibrium it cannot
  circulate; out of equilibrium it can, and the circulation is a different object
  that no scalar reaction coordinate captures.

The claim here, which the code demonstrates rather than argues, is that one piece
of mathematics (the discrete Hodge decomposition on a cell complex) answers both
at once, and throws in a third reading, the metastable basins, for free. Which is
to say: the same operator family that hands you the committor also tells you
whether a circulating current exists, and also counts the basins and the cycles.

## What you get

Three results on two analytic potentials, then the same pipeline on real
molecular dynamics.

**Result 1.** At equilibrium, the reactive current is a gradient flow, and its
potential is the committor. On Müller–Brown, the harmonic (circulating) component
of the reactive current is exactly zero, and the committor is recovered to machine
precision (correlation 1.000) as the potential of the current, read through the
electrical-network analogy: the committor is a voltage, the reactive current is the
corresponding electrical current, and Kirchhoff's voltage law is the statement that
it is curl-free.

**Result 2.** Break detailed balance and a harmonic current switches on. On a
driven torus, the steady-state current grows a globally circulating (winding)
component that has no potential, is invisible to any committor, and rises
monotonically with the driving (the harmonic energy fraction climbs from about 0 to
about 0.5, the current magnitude grows several-fold). You need a topological cycle
(`b1 > 0`) to even represent such a current, which is the whole point. The torus is
not an arbitrary choice: backbone-dihedral `(phi, psi)` space *is* a torus, so a
winding current here is exactly the kind of cyclic, non-equilibrium flux one worries
about in driven molecular systems.

**Result 3.** One operator, two readouts. The low-lying spectrum of the same Markov
model recovers and localizes the metastable basins (the data-driven shadow of
Morse/Witten theory, in which small eigenvalues correspond to minima), while the
harmonic dimension of the 1-Laplacian, the first Betti number `b1`, counts the
independent cycles. Müller–Brown comes back with three basins and `b1 = 0`; the
torus had `b1 = 2`. Same toolkit, dynamics and topology in one pass.

## The setup, briefly

Everything sits on four standard pieces, assembled so each one is visible:

**Overdamped Langevin.** We integrate `dX = (-grad V + F) dt + sqrt(2/beta) dW` on
two analytic 2D potentials, so the whole pipeline runs fast and every object can be
drawn. `V` is Müller–Brown (three wells, two saddles) or a periodic torus landscape;
`F` is an optional non-gradient force used only to push the system out of
equilibrium in Result 2.

**A Markov state model.** We cluster the sampled configurations into microstates
(the 0-cells), count transitions at a lag to get a row-stochastic transition matrix
`T` and its stationary distribution `pi`, and from those read off the committor (a
linear solve) and the reactive probability current via discrete transition path
theory. This is the data-driven step that carries over unchanged to real molecular
trajectories.

**A cell complex.** The microstates become the vertices of a 2D complex: oriented
edges (1-cells) connect neighbors, triangles (2-cells) fill it in. For Müller–Brown
we triangulate the sampled points (Delaunay, with long exterior triangles dropped)
so the complex hugs the explored region; for the torus we use a structured periodic
mesh so the topology (`b1 = 2`) is exact. The two boundary operators `B1` (nodes by
edges) and `B2` (edges by triangles) satisfy the one identity that makes all of this
work, `B1 @ B2 = 0`.

**The Hodge decomposition.** Any edge flow `J` (a 1-cochain: a number per oriented
edge) splits orthogonally into three pieces,

```
J  =  grad(s)        +     curl*(phi)      +     harmonic
   =  B1^T s         +     B2 phi          +     h
      (irrotational)       (locally            (global cycles;
                            circulating)         no potential)
```

The gradient part is the boundary of a node potential; the curl part is the boundary
of a triangle potential and circulates locally; the harmonic part is what is left,
and it is both divergence-free and curl-free. It lives in the kernel of the
1-Hodge-Laplacian `L1 = B1^T B1 + B2 B2^T`, whose dimension is exactly the first
Betti number `b1`. In a sense the harmonic component is the only part of the flow
that "knows" the global shape of the space: it is nonzero only when there is a loop
to wind around, and that is what makes it both a topological invariant and the right
diagnostic for non-equilibrium circulation.

## Reading the figures

**Result 1.**

![Müller-Brown committor and curl-free reactive current](media/result1_committor.png)

*Equilibrium reactive current on Müller–Brown: the committor (top right), the current
funneling through the saddle (bottom left), and a clean gradient / curl / harmonic
split of 100 / 0 / 0, with the recovered potential exactly on the diagonal against
the committor.*

The committor solved on the MSM (top right) runs smoothly from the lower-right basin
to the upper-left one. The reactive current (bottom left) funnels through the saddle,
brightest where the flux concentrates. Decomposed as an electrical current, it is
100% gradient, 0% curl, 0% harmonic, and its potential lies exactly on the diagonal
against the committor. A note on honesty: the raw probability flux is a *weighted*
gradient, `c * grad(q)`, and because the stationary weight `c` spans orders of
magnitude across the landscape, an unweighted split would leak energy into curl. The
voltage form `J / c` removes that weighting and gives the physically correct,
curl-free decomposition. The harmonic component is zero either way, because the
complex has no holes.

**Result 2.**

![Driven torus harmonic current](media/result2_harmonic.png)

*A constant tilt on the torus organizes the steady-state current into a coherent
winding (top right); its harmonic part (bottom left) is a clean uniform flow around
the loop, and the harmonic fraction and current magnitude both climb with the driving
(bottom right).*

At zero driving the steady-state current is sampling noise around zero (detailed
balance). Turn on a constant tilt (a torque the torus cannot unwind into a periodic
potential) and the current organizes into a coherent winding (top right), whose
harmonic part (bottom left) is a clean, uniform flow all the way around the torus.
The curve (bottom right) shows the harmonic fraction and the current magnitude both
climbing with the driving. This circulating current has no scalar potential, so no
committor, Deep-TICA coordinate, or other learned scalar reaction coordinate can
represent it: it is a cohomology class, not a function.

**Result 3.**

![Metastable basins and the Markov spectrum](media/result3_spectrum.png)

*The implied-timescale spectrum (top left) with a dominant gap, the two leading slow
eigenvectors separating the basins, and a metastable decomposition recovering all
three wells. b1 = 0: no cycles, as expected for a disk.*

The implied-timescale spectrum (top left) shows a dominant gap after the slowest mode
(escape from the deep basin), with finer slow modes resolving the rest. The two
leading slow eigenvectors (top right, bottom left) separate the basins, and a
metastable decomposition (bottom right) recovers all three wells. The first Betti
number is 0: no cycles, as expected for a landscape whose accessible region is a disk.

## The same machinery on real molecular dynamics

The toy potentials make every object visible, but the pipeline is not tied to them.
A short script runs actual MD of alanine dipeptide (the standard small-molecule
benchmark, whose backbone dihedrals `(phi, psi)` are the torus of Result 2) with
OpenMM, and feeds the trajectory through exactly the same adapter, so Results 1 and 2
reappear on data from a real force field.

**Result 4 (equilibrium).**

![Alanine dipeptide committor on real MD](media/result4_ala2_committor.png)

*Six nanoseconds of vacuum MD: the free-energy map, the committor between C7eq and
alpha_R, the reactive current along it, and a harmonic fraction of about zero despite
a real conformational loop (b1 = 1).*

Six nanoseconds of vacuum MD at 400 K, discretized on the `(phi, psi)` torus. The
committor between the C7eq/C5 basin and the alpha_R basin runs cleanly from zero to
one, the reactive current flows along it, and the harmonic fraction is about zero
even though the explored region carries a genuine conformational loop (`b1 = 1`). The
Result 1 statement holds on real data: a cycle existing in the accessible space does
not, by itself, put a current in it.

**Result 5 (driven).**

![Driven alanine dipeptide harmonic winding](media/result5_ala2_driven.png)

*A constant torque on phi breaks detailed balance: the molecule winds (0 to about 330
turns), the gradient part collapses to about 1%, and a harmonic winding current
switches on and grows with the torque. Equilibrium has none.*

A constant torque on the phi dihedral (a `CustomTorsionForce` with energy
`-tau*theta`, the molecular analog of the torus tilt) breaks detailed balance. As the
torque grows the molecule winds around phi (0 to about 330 net turns), the gradient
part collapses toward zero (a genuine divergence-free steady state), and a harmonic
component of the steady-state current switches on and grows, pointing coherently in
+phi. Equilibrium (tau = 0) has none. This is Result 2 on a molecule.

One honest caveat, and a reason the learned-operator direction below is attractive:
the harmonic on the molecule is noisier and less dominant than on the clean torus
(real psi-coupled motion and a heterogeneous Ramachandran landscape inject genuine
curl), and recovering it at all requires fine time resolution and unit lag in the
overdamped regime. A constant torque at ordinary friction spins phi fast enough that
it jumps several grid cells between saved frames, and the directed flux then hides
where a fixed adjacency cannot represent it (the current looks detailed-balanced and
the harmonic collapses to zero). A metric adapted to the data (a conductance-weighted
or learned Hodge operator) would not need the grid hand-tuned to the winding speed.

## Where this goes

The point of keeping the pipeline this small is that every piece of it is a layer
you could *learn* rather than hand-build. The committor is the gradient potential of
a current; the non-equilibrium signal is a harmonic class; the basins are a spectrum.
A natural next step, and the direction my research points at, is an operator that
respects this structure by construction: message passing on the complex via the Hodge
Laplacians, with the gradient / curl / harmonic split kept explicit, so that the
learned object inherits the conservation laws and the topological bookkeeping instead
of having to rediscover them from data. That is the throughline to my Topological
Neural Operators work (Hodge-theoretic message passing on simplicial and cellular
complexes; NeurIPS submission, under review), of which this is a deliberately minimal,
fully transparent cousin.

## Relation to prior work

This assembles well-established ideas and claims no new method; its value is clarity
and the explicit three-way reading. The committor-as-voltage and the discrete
reactive current are transition path theory (Metzner, Schütte and Vanden-Eijnden,
2009) and the classical electrical-network picture of reversible Markov chains (Doyle
and Snell). The decomposition is combinatorial, discrete-exterior-calculus Hodge
theory. The metastable-basin reading is standard Markov state modelling (PCCA-style
coarse-graining; the Witten/Morse connection between small Laplacian eigenvalues and
potential minima). Recent learning-based work on reactive flux and on Hodge and
spectral operator design is the closer neighborhood the flagship direction above
lives in; exact citations there are left for the writeup rather than asserted here,
since I would rather under-claim than misattribute.

## The code

Everything is plain NumPy, SciPy, and Matplotlib, seeded so the figures regenerate
bit for bit. The toolkit lives in `hodgemd/` (potentials, Langevin dynamics, the
Markov model and transition-path currents, the cell complex and Hodge split, the
spectral readouts), and the experiments in `experiments/` (the three toy results, an
adapter that runs the pipeline on any `(phi, psi)` trajectory, and the OpenMM
alanine-dipeptide driver). The toy results run in well under a minute:

```bash
pip install -r requirements.txt
cd experiments && python run_all.py     # Results 1-3
python ala2_openmm.py                    # Results 4-5 (needs openmm + mdtraj)
```

The full code is on GitHub: [github.com/sam-lev/hodge-currents](https://github.com/sam-lev/hodge-currents).
