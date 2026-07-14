*Where a quantity lives is part of what it is.*

This post builds, from scratch, the computational primitives of structure-preserving discrete physics — the discrete exterior derivative, the codifferential, the Hodge star, the Hodge Laplacian, and the Hodge decomposition — and then places them inside the geometric- and topological-deep-learning program: graph neural networks, point-cloud and mesh learning, and learning on simplicial, cellular, and combinatorial complexes. No prior background in algebraic topology or graph learning is assumed; every object is constructed by hand on a small mesh, and every central claim is proved. Readers who already know the material can skim the constructions and read the proofs and the worked computations at the end, where a unit square and an annulus are pushed through the entire machinery numerically.

The single thread tying the two halves together is an identity: $L_0 = $ *the graph Laplacian*. Standard graph neural networks turn out to be the rank-0 slice of a much richer, degree-typed calculus — and that calculus is exactly what makes a learned operator conserve what physics says it must.

## 1. The big idea: degree typing and the de Rham complex

In the continuum, the quantities of physics are not all the same kind of object:

- a **temperature** or an **electric potential** is a scalar field you sample at *points*;
- a **force**, **velocity**, or **electric field** is something you integrate along a *curve* (a circulation, a work, a line integral);
- a **flux** — of fluid, heat, or magnetic field — is something you integrate over a *surface*;
- a **density** of charge or mass is something you integrate over a *volume*.

These four kinds of object are the differential forms of degree $0, 1, 2, 3$, and the language that organizes them is exterior calculus. Three operators move between them, and remarkably they are all the *same* operator $d$, the exterior derivative, acting on different degrees. In three dimensions:

$$
\underbrace{C^0}_{\text{scalars}} \;\xrightarrow{\;d_0 = \nabla\;}\; \underbrace{C^1}_{\text{circulations}} \;\xrightarrow{\;d_1 = \operatorname{curl}\;}\; \underbrace{C^2}_{\text{fluxes}} \;\xrightarrow{\;d_2 = \operatorname{div}\;}\; \underbrace{C^3}_{\text{densities}}.
$$

This is the **de Rham complex**. The fact that makes it a "complex" in the algebraic sense is that applying $d$ twice always gives zero:

$$
d_{k+1} \circ d_k = 0, \qquad \text{equivalently} \qquad \operatorname{curl}\nabla = 0 \;\text{ and }\; \operatorname{div}\operatorname{curl} = 0.
$$

You may remember those two vector-calculus identities as isolated facts. They are instances of one structural law, and this entire post is, in a sense, the discrete shadow of that law.

**The thesis in one line.** A field's degree — which $C^k$ it lives in — is intrinsic physical data, not a modeling convenience. Discretizations that respect degree (store circulations on edges, fluxes on faces) make conservation and compatibility laws like $\operatorname{div}\operatorname{curl} = 0$ hold *exactly*, as identities of integer incidence matrices, rather than approximately. Numerical analysts have known this for decades: whether a scheme conserves mass, charge, or circulation depends on *where a field is stored*. Deep learning has mostly dumped every physical field onto vertices as feature channels and hoped the network rediscovers the structure. Topological deep learning closes that gap, and the machinery below is how.

## 2. Part I — Discrete Exterior Calculus

We now build a discrete, computable version of the de Rham complex on a mesh. A **mesh** is a *cell complex*: a set of vertices (0-cells), edges (1-cells), faces (2-cells), and volumes (3-cells), glued so that the boundary of every cell is a union of lower cells. For most of this post a triangulated surface — a *simplicial complex* — suffices, and for the proofs we will use simplicial language, where a $k$-simplex is spanned by $k+1$ vertices: $\sigma = [v_0, v_1, \dots, v_k]$.

### 2.1 Orientation and cochains

Each $k$-cell carries an **orientation**: an edge $[a,b]$ has a direction $a \to b$; a triangle $[a,b,c]$ has a circulation sense. Reversing the order of two vertices reverses the orientation: $[b,a] = -[a,b]$. Orientation is what lets a circulation or a flux have a *sign* — flow along an edge versus against it, flux out of a face versus into it.

**Definition (cochain).** A **$k$-cochain** is an assignment of a number (or a feature vector) to every oriented $k$-cell of the mesh, with the convention that reversing a cell's orientation negates the value. The space of $k$-cochains is $C^k$, a vector space of dimension equal to the number of $k$-cells. A 0-cochain is a value per vertex; a 1-cochain a value per oriented edge; a 2-cochain a value per oriented face.

The mental model: a $k$-cochain stores the *integrals* of a degree-$k$ physical field over the $k$-cells. A 0-cochain samples a potential at vertices; a 1-cochain stores the circulation of a vector field along each edge; a 2-cochain stores the flux through each face. This "store the integral, not the pointwise value" convention is the entire trick — it is what turns calculus into exact linear algebra below.

In machine-learning terms, a cochain is just a feature matrix indexed by cells: $C^0 \cong \mathbb{R}^{n_v \times c}$, $C^1 \cong \mathbb{R}^{n_e \times c}$, and so on. The novelty relative to standard GNNs — which only ever use $C^0$, features on vertices — is that we will now also put features on edges and faces and let them interact through the geometry of the mesh.

### 2.2 Incidence matrices and the discrete exterior derivative

The discrete analogue of $d$ is pure combinatorics: **signed incidence matrices** record which cells bound which.

**Definition (boundary of a simplex).** For an oriented $k$-simplex $\sigma = [v_0, \dots, v_k]$, its boundary is the alternating sum of its facets, each obtained by deleting one vertex:

$$
\partial_k [v_0, \dots, v_k] \;=\; \sum_{i=0}^{k} (-1)^i \, [v_0, \dots, \widehat{v_i}, \dots, v_k],
$$

where $\widehat{v_i}$ means $v_i$ is omitted. For an edge, $\partial [a,b] = [b] - [a]$ (head minus tail). For a triangle, $\partial [a,b,c] = [b,c] - [a,c] + [a,b]$ — the three edges traversed consistently with the triangle's circulation sense.

**Definition (incidence matrix $B_k$).** $B_k$ is the matrix of $\partial_k$: its rows index $(k-1)$-cells, its columns index $k$-cells, and entry $(\tau, \sigma)$ is $+1$ or $-1$ according to whether $\tau$ appears in $\partial\sigma$ with matching or opposing orientation, and $0$ if it does not appear. For an edge $e = [a,b]$, column $e$ of $B_1$ has $+1$ in row $b$ and $-1$ in row $a$.

**Definition (discrete exterior derivative).** The discrete exterior derivative $d_k : C^k \to C^{k+1}$ is the *transpose* of the next incidence matrix:

$$
d_k = B_{k+1}^{\top}.
$$

Concretely:

- $d_0 = B_1^{\top}$ is the **discrete gradient**: $(d_0 f)(e) = f(b) - f(a)$ on the edge $e = [a,b]$ — the change of a vertex field along the edge.
- $d_1 = B_2^{\top}$ is the **discrete curl**: it sums a 1-cochain (a circulation) around the boundary of each face, with signs from orientation.
- $d_2 = B_3^{\top}$ is the **discrete divergence**: the net flux out of each volume.

**Why $d_k = B_{k+1}^\top$ *is* Stokes' theorem.** The continuous identity behind everything is the generalized Stokes theorem,

$$
\int_{\sigma} d\omega \;=\; \int_{\partial \sigma} \omega,
$$

the integral of a derivative over a region equals the integral of the original over the boundary. Recall that cochains store integrals. So the discrete statement "the value of $d\omega$ on cell $\sigma$" *should be* "the signed sum of the values of $\omega$ over the cells bounding $\sigma$" — and that signed sum is exactly the $\sigma$-row of $B_{k+1}^\top$ dotted with $\omega$. Defining $d = B^\top$ is therefore not an analogy or an approximation; it is Stokes' theorem transcribed as a sparse integer matrix, holding exactly on every mesh. This is why structure-preserving discretization is possible at all.

### 2.3 The fundamental identity $B_k B_{k+1} = 0$

**Theorem (the boundary of a boundary is empty).** For any simplicial (indeed any cell) complex,

$$
B_k B_{k+1} = 0 \qquad\Longleftrightarrow\qquad d_{k+1}\, d_k = 0.
$$

In particular $d_1 d_0 = 0$ ($\operatorname{curl}\nabla = 0$) and $d_2 d_1 = 0$ ($\operatorname{div}\operatorname{curl} = 0$) hold *exactly*, as identities of integer matrices, on any mesh of any quality.

**Proof (algebraic, for simplices).** It suffices to show $\partial_k \partial_{k+1} \sigma = 0$ on every $(k{+}1)$-simplex $\sigma = [v_0, \dots, v_{k+1}]$, since $B_k B_{k+1}$ is the matrix of $\partial_k \partial_{k+1}$ and $d d = (B B)^\top$. Apply the boundary formula twice:

$$
\partial_k \partial_{k+1} \sigma
= \partial_k \sum_{j=0}^{k+1} (-1)^j [\dots, \widehat{v_j}, \dots]
= \sum_{j} \sum_{i \neq j} (\pm 1)\, [\dots, \widehat{v_i}, \dots, \widehat{v_j}, \dots].
$$

Track the sign of the term in which $v_i$ and $v_j$ are both deleted, say with $i < j$. It arises in exactly two ways:

1. delete $v_j$ first (sign $(-1)^j$), then delete $v_i$ from the resulting $k$-simplex, where $v_i$ still sits in position $i$ (sign $(-1)^i$): total sign $(-1)^{i+j}$;
2. delete $v_i$ first (sign $(-1)^i$), then delete $v_j$, which has now *shifted down one position* to index $j-1$ (sign $(-1)^{j-1}$): total sign $(-1)^{i+j-1}$.

The two contributions are equal in magnitude and opposite in sign, so every doubly-deleted face cancels in pairs and the sum is zero. $\blacksquare$

**Proof (the picture, for $B_1 B_2 = 0$).** The boundary of a triangle is its three edges, oriented head-to-tail around the circulation sense. Take the boundary again: each vertex of the triangle is the *head* of exactly one boundary edge and the *tail* of exactly the next one, so its two contributions are $+1$ and $-1$ and cancel. "The boundary has no boundary." The algebraic proof above is this picture with bookkeeping.

**Why this identity matters so much.** Because $B_1 B_2 = 0$ is *exact*, a discrete field that is a discrete curl of something automatically has *zero* discrete divergence — not small divergence, zero, to the last bit. A numerical scheme built on these operators conserves the corresponding quantity to machine precision, on any mesh, rather than leaking it as discretization error. This one identity is the structural reason edge- and face-based finite element methods are stable for Maxwell's equations and incompressible flow where naive nodal schemes fail: place the electric field on edges and the magnetic flux on faces, and Gauss's law for magnetism ($\operatorname{div} B = 0$) holds *by construction*.

It is worth pausing on the contrast with the other way of getting conservation into a learned model: adding a physics-informed penalty term to the loss that punishes divergence of a curl. A soft penalty can be traded off against data fit and *can be violated* at inference time; an identity of integer matrices *cannot*. That structural-versus-soft distinction is the philosophical heart of everything that follows.

### 2.4 Adding metric: the Hodge star

Everything so far is purely combinatorial — it depends only on *which cells touch which*, not on any lengths or angles. Physics also needs **metric** information — lengths, areas, volumes — to compare quantities of different degrees and to define energy. This enters through the Hodge star.

**Definition (discrete Hodge star $M_k$).** $M_k$ is a (typically diagonal, positive-definite) *mass matrix* on $C^k$ encoding the geometry: $M_0$ holds vertex (dual-cell) volumes, $M_1$ holds edge weights (ratios of dual to primal edge length in the classical circumcentric-dual construction), $M_2$ holds face areas, $M_3$ holds cell volumes. It defines the discrete $L^2$ inner product on cochains,

$$
\langle u, v \rangle_{C^k} \;=\; u^{\top} M_k \, v,
$$

the discrete analogue of $\int_\Omega u\,v$. The Hodge star is where *where things live geometrically* — not just combinatorially — enters the calculus.

The division of labor is the secret of structure preservation, so it deserves emphasis:

- The **combinatorial** part ($B_k$) is exact and mesh-independent. Conservation laws ride on $B$ and hold as integer identities.
- The **metric** part ($M_k$) is where discretization quality and approximation error live. Accuracy rides on $M$.

Refine the mesh, and $M$ improves while $B$'s identities remain exactly true throughout. Conservation is never "converged to"; it holds from the first mesh onward.

### 2.5 The codifferential: the adjoint of $d$

$d$ goes *up* the complex (vertices → edges → faces). Once we have inner products, we can ask for its adjoint, which goes *down*.

**Definition (codifferential).** The codifferential $\delta_k : C^k \to C^{k-1}$ is the adjoint of $d_{k-1}$ with respect to the Hodge inner products. Explicitly,

$$
\delta_k \;=\; M_{k-1}^{-1}\, B_k\, M_k .
$$

**Proof that this formula gives the adjoint.** Adjointness means $\langle d_{k-1} u, \, v \rangle_{C^k} = \langle u, \, \delta_k v \rangle_{C^{k-1}}$ for all $u \in C^{k-1},\, v \in C^k$. Compute both sides:

$$
\langle d_{k-1} u, v \rangle_{C^k}
= (B_k^{\top} u)^{\top} M_k v
= u^{\top} B_k M_k v,
$$

$$
\langle u, \delta_k v \rangle_{C^{k-1}}
= u^{\top} M_{k-1} \left( M_{k-1}^{-1} B_k M_k \right) v
= u^{\top} B_k M_k v. \qquad\blacksquare
$$

This adjointness is the discrete **integration-by-parts** identity — the discrete counterpart of moving a derivative from one factor to the other at the cost of a sign and boundary terms. For $k=1$, $\delta_1$ sends an edge field to a vertex field measuring net outflow: it is the discrete divergence-like operator. Note that $\delta$, unlike $d$, *does* depend on the metric: it is combinatorics ($B_k$) sandwiched between geometry ($M^{-1}$ and $M$).

An immediate corollary of $dd = 0$ is $\delta\delta = 0$:

$$
\delta_k \delta_{k+1} = M_{k-1}^{-1} B_k M_k \, M_k^{-1} B_{k+1} M_{k+1} = M_{k-1}^{-1} (B_k B_{k+1}) M_{k+1} = 0.
$$

### 2.6 The Hodge Laplacian

Combining the up-map $d$ and the down-map $\delta$ gives the central operator of the whole subject.

**Definition (Hodge Laplacian).** The $k$-th Hodge Laplacian $L_k : C^k \to C^k$ is

$$
L_k \;=\; \underbrace{\delta_{k+1}\, d_k}_{L_k^{\uparrow}} \;+\; \underbrace{d_{k-1}\, \delta_k}_{L_k^{\downarrow}} .
$$

The **up-Laplacian** $L_k^\uparrow = \delta_{k+1} d_k$ couples a $k$-cochain to the $(k{+}1)$-cells above it (curl-like coupling); the **down-Laplacian** $L_k^\downarrow = d_{k-1}\delta_k$ couples it to the $(k{-}1)$-cells below (divergence-like coupling).

**Proposition.** $L_k$ is self-adjoint and positive semidefinite with respect to $\langle \cdot,\cdot\rangle_{C^k}$, and its quadratic form is

$$
\langle L_k u, u \rangle \;=\; \| d_k u \|^2 + \| \delta_k u \|^2 .
$$

**Proof.** Using adjointness twice,

$$
\langle L_k u, u\rangle
= \langle \delta_{k+1} d_k u, u \rangle + \langle d_{k-1}\delta_k u, u\rangle
= \langle d_k u, d_k u \rangle + \langle \delta_k u, \delta_k u \rangle
= \|d_k u\|^2 + \|\delta_k u\|^2 \;\ge\; 0 .
$$

Self-adjointness follows the same way: $\langle L_k u, v\rangle = \langle d_k u, d_k v\rangle + \langle \delta_k u, \delta_k v\rangle = \langle u, L_k v\rangle$. $\blacksquare$

**Corollary (harmonic $\Leftrightarrow$ closed and coclosed).** $L_k h = 0$ if and only if $d_k h = 0$ *and* $\delta_k h = 0$.

**Proof.** If $d_k h = 0 = \delta_k h$ then $L_k h = 0$ termwise. Conversely, if $L_k h = 0$ then $0 = \langle L_k h, h\rangle = \|d_k h\|^2 + \|\delta_k h\|^2$, a sum of two nonnegative terms, so both vanish; since $M$ is positive definite, $d_k h = 0$ and $\delta_k h = 0$. $\blacksquare$

Elements of $\ker L_k$ are called **harmonic** $k$-cochains: simultaneously curl-free and divergence-free.

### 2.7 The bridge to graph neural networks: $L_0$ *is* the graph Laplacian

At degree $k = 0$ there are no $(-1)$-cells, so the down term vanishes and

$$
L_0 = \delta_1 d_0 = M_0^{-1} B_1 M_1 B_1^{\top} \;\xrightarrow{\;M = I\;}\; B_1 B_1^{\top}.
$$

**Proposition.** With unit metric, $B_1 B_1^{\top} = D - A$, the ordinary graph Laplacian, where $D$ is the diagonal degree matrix and $A$ the adjacency matrix. (With general $M_1$ one gets the *weighted* graph Laplacian.)

**Proof.** Entry $(u,v)$ of $B_1 B_1^\top$ is $\sum_{e} B_1[u,e]\, B_1[v,e]$, a sum over edges. For $u = v$: each edge incident to $u$ contributes $(\pm 1)^2 = 1$, so the diagonal entry is $\deg(u)$. For $u \neq v$: only an edge $e = [u,v]$ (in either orientation) has both entries nonzero, contributing $(-1)(+1) = -1$; if $u,v$ are not adjacent the entry is $0$. Hence $B_1 B_1^\top = D - A$. Note the answer is independent of the edge orientations chosen — the signs square away on the diagonal and pair up off it. $\blacksquare$

This single identity is the precise sense in which **standard graph learning is the rank-0 corner of discrete exterior calculus**. Spectral graph theory, GCN's normalized Laplacian, diffusion on graphs, and the entire Weisfeiler–Lehman expressivity story (Part II) are statements about $L_0$. DEC says: there is also $L_1$ (on edges — with *no* graph analogue), $L_2$, and so on, and they carry physics the vertex Laplacian cannot see.

### 2.8 The Hodge decomposition and topological holes

The crowning structural fact — and the source of the "harmonic" features that topological deep learning exploits — is that the cochain spaces split into three orthogonal pieces.

**Theorem (discrete Hodge decomposition).** Every $k$-cochain space decomposes orthogonally with respect to the $M_k$-inner product as

$$
C^k \;=\; \underbrace{\operatorname{im}(d_{k-1})}_{\text{exact}} \;\oplus\; \underbrace{\operatorname{im}(\delta_{k+1})}_{\text{coexact}} \;\oplus\; \underbrace{\ker(L_k)}_{\text{harmonic}} .
$$

**Proof.** We use one standard fact from linear algebra over a finite-dimensional inner-product space: for any linear map $A$ into $C^k$, we have the orthogonal splitting $C^k = \operatorname{im}(A) \oplus \ker(A^{*})$, where $A^*$ is the adjoint. (Indeed $v \perp \operatorname{im}A \iff \langle Au, v\rangle = 0\;\forall u \iff \langle u, A^*v\rangle = 0\;\forall u \iff A^*v = 0$.)

*Step 1: the three subspaces are pairwise orthogonal.*
- Exact $\perp$ coexact: for $a \in C^{k-1}$, $b \in C^{k+1}$,
$$\langle d_{k-1}a,\; \delta_{k+1}b \rangle = \langle d_k d_{k-1} a,\; b\rangle = 0$$
by adjointness and $dd = 0$.
- Exact $\perp$ harmonic: $\langle d_{k-1}a, h\rangle = \langle a, \delta_k h\rangle = 0$ since harmonic implies $\delta_k h = 0$.
- Coexact $\perp$ harmonic: $\langle \delta_{k+1}b, h\rangle = \langle b, d_k h\rangle = 0$ since harmonic implies $d_k h = 0$.

*Step 2: the three subspaces fill $C^k$.* By the standard fact, the orthogonal complement of $\operatorname{im}(d_{k-1})$ is $\ker(\delta_k)$, and the orthogonal complement of $\operatorname{im}(\delta_{k+1})$ is $\ker(d_k)$. Hence

$$
\left( \operatorname{im}(d_{k-1}) \oplus \operatorname{im}(\delta_{k+1}) \right)^{\perp}
= \ker(\delta_k) \,\cap\, \ker(d_k)
= \ker(L_k),
$$

the last equality by the corollary in §2.6. So the exact and coexact parts, together with their joint orthogonal complement — the harmonic part — exhaust $C^k$. $\blacksquare$

**Theorem (harmonic dimension = Betti number).** $\dim \ker(L_k) = \beta_k$, the $k$-th Betti number of the complex — the number of independent $k$-dimensional "holes."

**Proof.** Recall (or take as the definition, with real coefficients) that the $k$-th cohomology of the complex is $H^k = \ker(d_k) / \operatorname{im}(d_{k-1})$, and $\beta_k = \dim H^k$. We exhibit an isomorphism $\varphi : \ker(L_k) \to H^k$, $\varphi(h) = [h]$.

*Well-defined:* harmonic cochains satisfy $d_k h = 0$, so $h \in \ker d_k$ and its class $[h]$ exists.

*Injective:* suppose $h$ is harmonic and $[h] = 0$, i.e. $h = d_{k-1}a$ is exact. Then $\|h\|^2 = \langle d_{k-1}a, h\rangle = \langle a, \delta_k h\rangle = 0$, so $h = 0$. (Harmonic and exact intersect trivially — that is just Step 1 orthogonality applied to a vector lying in both.)

*Surjective:* take any class $[z]$ with $d_k z = 0$. Hodge-decompose $z = d_{k-1}a + \delta_{k+1}b + h$. Apply $d_k$: $0 = d_k z = d_k \delta_{k+1} b$ (the exact term dies by $dd=0$, the harmonic term by definition). Then

$$
\|\delta_{k+1} b\|^2 = \langle \delta_{k+1}b, \delta_{k+1}b\rangle = \langle d_k \delta_{k+1} b, \, b \rangle = 0,
$$

so the coexact part of a cocycle vanishes and $z = d_{k-1}a + h$, i.e. $[z] = [h] = \varphi(h)$. $\blacksquare$

So each cohomology class contains exactly one harmonic representative — the discrete Hodge theorem. Since Betti numbers are topological invariants ($\beta_0 = $ number of connected components, $\beta_1 = $ independent loops/tunnels, $\beta_2 = $ enclosed voids), the harmonic subspace is a *purely topological* feature of the mesh, computed by ordinary numerical linear algebra (a nullspace of a sparse symmetric matrix).

**What the three pieces mean physically.** Decompose a flow field on a domain:

- The **exact** part is a pure gradient — it comes from a potential; irrotational flow.
- The **coexact** part is a pure curl — incompressible, divergence-free rotational flow.
- The **harmonic** part is what is left: simultaneously curl-free *and* divergence-free. On a simply connected domain this forces zero (the theorem above: $\beta_1 = 0$). But when the domain has holes, harmonic fields appear — precisely one dimension per hole — and they are *the circulation around a hole that no potential can produce*. Think of steady flow around an island, or the magnetic field circling a wire.

Harmonic features are therefore *global, topological signals*: they detect holes. A learned operator that has access to a harmonic channel can represent topologically constrained physics — a conserved circulation, a flux threading a handle — that a purely local scheme provably cannot, as we will make concrete next.

One honest caveat to keep in view: harmonic projection is a *global* operation (a nullspace computation coupling every cell), and on very large meshes it can become the memory bottleneck of the pipeline. The topology does not come for free; it comes for a solve.

## 3. Part II — Geometric and Topological Deep Learning

With the calculus in hand, we turn to the learning architectures that operate on graphs, meshes, and complexes. The organizing idea is **message passing**: features are updated by aggregating information from neighbors, where "neighbor" is defined by exactly the incidence structure built above.

### 3.1 Graph neural networks

**Definition (message-passing neural network, MPNN).** A graph has vertices $v$ with features $h_v$ and edges defining neighborhoods $\mathcal{N}(v)$. One layer updates every vertex by

$$
h_v^{(\ell+1)} \;=\; \phi\!\left( h_v^{(\ell)},\; \bigoplus_{u \in \mathcal{N}(v)} \psi\!\left( h_v^{(\ell)}, h_u^{(\ell)}, e_{vu} \right) \right),
$$

where $\psi$ is a learned *message* function, $\bigoplus$ is a permutation-invariant *aggregation* (sum, mean, max), and $\phi$ is a learned *update*. Permutation invariance of $\bigoplus$ is what makes the output independent of how neighbors are ordered — the basic inductive bias of graph learning: relabel the nodes and the computation is unchanged.

Four canonical instances, each a particular choice of $\psi, \bigoplus, \phi$:

- **GCN** (Kipf–Welling): a normalized-Laplacian smoother, $H^{(\ell+1)} = \sigma\!\left( \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2} H^{(\ell)} W^{(\ell)} \right)$ with $\tilde{A} = A + I$. Cheap, strong baseline — literally a learned diffusion on the graph, i.e. a polynomial in $L_0$. Notice how directly the rank-0 calculus shows up.
- **GraphSAGE** (Hamilton et al.): sample-and-aggregate, $h_v = \sigma\!\left( W \cdot [\, h_v \,\|\, \mathrm{AGG}\{h_u\} \,] \right)$; designed to scale by neighbor sampling and to generalize *inductively* to unseen nodes.
- **GIN** (Xu et al.): $h_v^{(\ell+1)} = \mathrm{MLP}\!\left( (1+\epsilon)\, h_v^{(\ell)} + \sum_{u \in \mathcal{N}(v)} h_u^{(\ell)} \right)$; the sum aggregator composed with an MLP is provably the most expressive simple scheme (next subsection).
- **GAT** (Veličković et al.): attention-weighted aggregation, $h_v = \sigma\!\left( \sum_u \alpha_{vu} W h_u \right)$ with learned, normalized attention $\alpha_{vu}$; lets the network learn *which* neighbors matter.

#### 3.1.1 Expressivity: the Weisfeiler–Lehman lens

How powerful is message passing? The answer is sharp, and worth understanding rather than memorizing.

The **1-dimensional Weisfeiler–Lehman (1-WL) test** is a classical graph-isomorphism heuristic: give every vertex an initial color; then repeatedly recolor each vertex by hashing the pair (its own color, the *multiset* of its neighbors' colors); stop when colors stabilize. Two graphs whose stable color histograms differ are certainly non-isomorphic; if the histograms agree the test is silent.

**Theorem (Xu et al., Morris et al.).** An MPNN can distinguish two graphs only if 1-WL distinguishes them; and an MPNN with *injective* aggregation and update (achieved by sum aggregation into an MLP — the GIN construction) matches this bound.

**Proof sketch.** *Upper bound:* one MPNN layer computes, at each vertex, a function of exactly the same data 1-WL hashes — the vertex's own state and the multiset of neighbor states. By induction on layers, if 1-WL assigns two vertices the same color at round $\ell$, any MPNN assigns them equal features at layer $\ell$; hence identically-colored graphs yield identical readouts. *Matching the bound:* the multiset of neighbor features must be encoded injectively for the network to simulate the WL hash. Mean and max lose multiplicity information (mean of $\{a,a,b,b\}$ equals mean of $\{a,b\}$; max forgets everything but the largest), while a *sum* of suitable (learned, MLP-composed) embeddings can be made injective on multisets over a countable domain — this is the deep-sets style argument at the core of GIN. $\blacksquare$

The connection back to DEC is direct: 1-WL operates through $L_0$-style neighborhoods, so it is blind to exactly the higher-order and topological structure ($L_1$, harmonics) that simplicial and cellular message passing adds. The classic failure example: 1-WL cannot distinguish two disjoint triangles from a single hexagon — both are 2-regular graphs with identical local views — yet they differ in $\beta_0$ and in their cycle structure. "Go beyond 1-WL" is one standing motivation for topological deep learning; "carry degree-typed physics" is the other, and they are the same mathematics.

#### 3.1.2 The two pathologies: over-smoothing and over-squashing

- **Over-smoothing.** Stacking many message-passing layers repeatedly averages features. In the linear caricature, $\ell$ layers of GCN-style propagation apply the $\ell$-th power of a fixed smoothing operator whose spectrum lies in $[-1,1]$; as $\ell \to \infty$, all components except the dominant eigenvector (constant on each connected component — note: the harmonic 0-cochains!) are damped geometrically, so all node representations converge to the same vector and discriminative signal is destroyed. This caps practical depth and is why long-range dependencies are hard for vanilla GNNs.
- **Over-squashing.** Information from a receptive field that grows exponentially with depth must be compressed ("squashed") through fixed-width vertex representations as it funnels through graph bottlenecks; gradients between distant nodes decay with the (small) number of bottleneck paths, so distant nodes cannot effectively communicate. The severity is controlled by the bottleneck/curvature structure of the graph.

Both pathologies matter enormously for physics on meshes: elliptic problems (steady heat flow, incompressible pressure solves) have *global* domains of dependence — poke the boundary anywhere and the solution moves everywhere — so a surrogate needs a global receptive field. But naive deep GNNs over-smooth before they reach global range. The standard fixes are hierarchy and spectra: coarsen the mesh into a multi-level pyramid (the learned analogue of multigrid) or work with global spectral latents.

#### 3.1.3 Pooling and practice

Hierarchical pooling schemes (DiffPool, Top-K pooling, graph U-Nets) coarsen the graph to build multi-scale receptive fields — directly analogous to multigrid in numerical linear algebra. In practice all of this is implemented in PyTorch Geometric (PyG) or DGL, and both use the same essential trick for batching: a batch of variable-size graphs is represented as *one big disconnected graph* with a batch-index vector, which is exactly what is needed to batch variable-resolution meshes — and note that "number of connected components" being the right notion of batch separation is, once again, $\beta_0$.

### 3.2 Point-cloud and mesh learning

Real geometry often arrives as a **point cloud** (an unordered set of 3D points) or a **surface mesh**. The core requirement is to respect the symmetries of 3D space.

- **PointNet** (Qi et al.): the foundational permutation-invariant architecture. It applies a shared MLP $h$ to each point independently, then aggregates with a symmetric function: $f(\{x_i\}) = \gamma\!\left( \max_i h(x_i) \right)$. The max — a symmetric function — guarantees invariance to point ordering, the same idea as the GNN aggregator. **PointNet++** adds hierarchical local neighborhoods to capture the multi-scale geometry that a single global max pooling misses.
- **Mesh networks** additionally exploit connectivity (edges and faces), so they can use *geodesic* neighborhoods and the discrete operators of Part I rather than only Euclidean $k$-nearest-neighbors. The distinction is not cosmetic: on a thin slotted plate, two points on opposite sides of the slot are Euclidean-close but geodesically far; a $k$-NN graph "short-circuits" across the gap and couples physics that the actual surface keeps separate, while the native mesh graph respects the surface.

**Definition (invariance and equivariance under $E(3)$/$SE(3)$).** Let $g$ be a rigid motion (rotation $R$ and translation $t$; $SE(3)$ excludes reflections, $E(3)$ includes them). A *scalar* prediction (e.g. a drag coefficient) should be **invariant**: $f(gX) = f(X)$. A *vector or tensor field* prediction (e.g. a velocity or stress field) should be **equivariant**: $f(gX) = g \cdot f(X)$ — rotate the input and the output rotates with it. Building these symmetries into the architecture, rather than hoping to learn them from augmented data, provably shrinks the hypothesis space to the physically meaningful functions and empirically improves sample efficiency and generalization; this is why equivariant networks are standard for molecular and physical geometry.

### 3.3 Topological deep learning

**Topological deep learning (TDL)** generalizes message passing from graphs (vertices + edges) to higher-order domains, using exactly the DEC structure of Part I as the routing of messages.

- **Domains.** A *simplicial complex* (vertices, edges, triangles, tetrahedra) is the triangulated-mesh case. A *cell complex* allows general polygonal/polyhedral cells. A *combinatorial complex* (Hajij et al.) unifies cell complexes and hypergraphs by allowing arbitrary set-valued "cells" equipped with a rank function — the most general substrate, and the one underlying the modern message-passing abstraction.
- **Higher-order message passing.** Cells of *every* rank carry features, and messages flow along the incidence relations we have already built: boundary ($B$, i.e. $\delta$-direction), coboundary ($B^\top$, i.e. $d$-direction), and the up/down adjacencies of the Hodge Laplacians. Concretely, an edge feature can be updated from the vertices that bound it, the faces it bounds, and the neighboring edges that share a vertex (down-adjacency, via $L_1^\downarrow$) or a face (up-adjacency, via $L_1^\uparrow$). This strictly subsumes graph message passing — the vertex-only, $L_0$ case — and it is precisely a learned, nonlinear generalization of the operators $d$, $\delta$, $L_k$.
- **Harmonic and persistence features.** Two families of topological features feed these networks:
  1. **Harmonic features** are the $\ker(L_k)$ components from the Hodge decomposition — global descriptors that, by the theorems of §2.8, count and localize holes.
  2. **Persistence descriptors** come from *persistent homology*: grow a scale parameter (a distance threshold on a point cloud, a level-set value on a function) and track the topological features — components, loops, voids — as they are *born* and *die* across the filtration. The multiset of (birth, death) pairs is the **persistence diagram**, a multi-scale topological signature. Its power comes from a stability theorem: perturbing the input function by $\varepsilon$ in the sup norm moves the diagram by at most $\varepsilon$ in bottleneck distance, so the signature is robust to noise by construction.

  Both families inject information that *no amount of local message passing can recover* — that is not rhetoric but the content of the 1-WL bound plus the topology theorems above, and the worked example below makes it concrete on five edges.

**The throughline to operator learning.** Topological deep learning is the framework; *topological neural operators* are the operator-learning instance: cochains of all degrees, coupled by learned versions of $d$, $\delta$, $L$, trained to approximate a discretization-invariant solution map for a PDE. The $L_0 = $ graph-Laplacian identity is what makes the inclusion precise — Fourier and graph neural operators are recovered as the rank-0 specializations — and the Hodge decomposition is what lets such an operator represent topology-constrained physics *by construction* rather than by penalty.

## 4. Worked computation: everything by hand

Theory earns its keep when you can run it on a napkin. We do the full pipeline twice: on a unit square split into two triangles (no holes), and on an annulus (one hole).

### 4.1 The two-triangle square

**The complex.** Vertices $v_1, v_2, v_3, v_4$ (say the corners of a unit square, $v_1$ bottom-left, going counterclockwise); edges $e_1 = [v_1,v_2]$, $e_2 = [v_2,v_3]$, $e_3 = [v_1,v_3]$ (the diagonal), $e_4 = [v_3,v_4]$, $e_5 = [v_1,v_4]$, each oriented from lower to higher index; faces $T_a = [v_1,v_2,v_3]$ and $T_b = [v_1,v_3,v_4]$. Sanity check with the Euler characteristic: $V - E + F = 4 - 5 + 2 = 1$, correct for a disk.

**Incidence $B_1$ (vertices × edges).** Column $e = [a,b]$ has $-1$ in row $a$ and $+1$ in row $b$:

$$
B_1 =
\begin{pmatrix}
 & e_1 & e_2 & e_3 & e_4 & e_5 \\
v_1 & -1 & 0 & -1 & 0 & -1 \\
v_2 & 1 & -1 & 0 & 0 & 0 \\
v_3 & 0 & 1 & 1 & -1 & 0 \\
v_4 & 0 & 0 & 0 & 1 & 1
\end{pmatrix}
$$

**Incidence $B_2$ (edges × faces).** Using $\partial[v_i,v_j,v_k] = [v_j,v_k] - [v_i,v_k] + [v_i,v_j]$: $\partial T_a = e_1 + e_2 - e_3$ and $\partial T_b = e_3 + e_4 - e_5$:

$$
B_2 =
\begin{pmatrix}
 & T_a & T_b \\
e_1 & 1 & 0 \\
e_2 & 1 & 0 \\
e_3 & -1 & 1 \\
e_4 & 0 & 1 \\
e_5 & 0 & -1
\end{pmatrix}
$$

**Verify $B_1 B_2 = 0$.** The $T_a$ column of the product is $B_1(e_1 + e_2 - e_3)$, computed row by row:

- $v_1:\; (-1) - 0 - (-1) = 0$
- $v_2:\; 1 - 1 - 0 = 0$
- $v_3:\; 0 + 1 - 1 = 0$
- $v_4:\; 0$

The $T_b$ column, $B_1(e_3 + e_4 - e_5)$, cancels identically the same way ($v_1: -1+0+1$, $v_3: 1-1-0$, $v_4: 0+1-1$). The boundary of each triangle, mapped down to vertices, is zero — the discrete $\operatorname{div}\operatorname{curl} = 0$, verified as arithmetic.

**Build $L_0$ and confirm it is the graph Laplacian.** With unit metric ($M_0 = M_1 = I$): $d_0 = B_1^\top$, $\delta_1 = B_1$, so

$$
L_0 = \delta_1 d_0 = B_1 B_1^{\top} =
\begin{pmatrix}
 & v_1 & v_2 & v_3 & v_4 \\
v_1 & 3 & -1 & -1 & -1 \\
v_2 & -1 & 2 & -1 & 0 \\
v_3 & -1 & -1 & 3 & -1 \\
v_4 & -1 & 0 & -1 & 2
\end{pmatrix}
= D - A.
$$

The diagonal is each vertex's degree ($v_1, v_3$ touch three edges; $v_2, v_4$ touch two), and the off-diagonal is $-1$ exactly for adjacent vertices ($v_2$ and $v_4$ are not adjacent — no edge crosses the other diagonal — hence the $0$). This is precisely $D - A$, confirming $L_0 = $ graph Laplacian on this mesh.

**Harmonic subspaces.** Degree 0: the mesh is connected, so $L_0 \mathbf{1} = 0$ for the constant vector $\mathbf{1} = (1,1,1,1)^\top$ and nothing else: $\dim \ker L_0 = 1 = \beta_0$. (Check directly: each row of $L_0$ above sums to zero.) Degree 1: the disk is simply connected, so $\dim \ker L_1 = \beta_1 = 0$ — every edge 1-cochain is purely exact + coexact, with no harmonic part.

**Dimension audit.** The Hodge decomposition of $C^1$ predicts $\dim C^1 = \operatorname{rank}(d_0) + \operatorname{rank}(d_1) + \beta_1$. Here $\operatorname{rank}(d_0) = \operatorname{rank}(B_1) = n_v - \beta_0 = 4 - 1 = 3$ (image of the gradient = exact part), $\operatorname{rank}(d_1) = \operatorname{rank}(B_2) = 2$ (the two columns of $B_2$ are visibly independent; the coexact part of $C^1$ has the same rank as $d_1$ since $\operatorname{rank}\delta_2 = \operatorname{rank} d_1$), and $\beta_1 = 0$. Total: $3 + 2 + 0 = 5 = $ number of edges. The books balance.

### 4.2 What changes when you add a hole: the annulus

Now triangulate an **annulus** — a disk with a hole punched out; concretely, the region between an outer and an inner polygon, filled with triangles. Topologically: still connected, so $\beta_0 = 1$; but now $\beta_1 = 1$ — there is one independent loop, a circuit around the hole that cannot be filled in by any combination of triangles in the complex. The structural consequences, item by item:

- $B_1 B_2 = 0$ **still holds.** It is true on every complex — the proof in §2.3 never mentioned topology. $L_1 = L_1^\uparrow + L_1^\downarrow$ is unchanged in form.
- **The harmonic subspace at degree 1 becomes one-dimensional:** $\dim \ker L_1 = \beta_1 = 1$. By the corollary of §2.6, there now exists a 1-cochain $h$ that is simultaneously *curl-free* ($d_1 h = 0$: it sums to zero around every filled triangle) and *divergence-free* ($\delta_1 h = 0$: zero net flow at every vertex), yet is **not** the gradient of any vertex potential. Concretely $h$ assigns values to edges so that flow circulates around the hole: try to build a potential for it by integrating along edges, and after one trip around the hole the "potential" has increased by the total circulation — a contradiction unless the circulation is zero. $h$ is the discrete circulation around the hole: the topological signal made flesh as a vector in $\mathbb{R}^{n_e}$.
- $L_0$ is still the graph Laplacian of the (now larger) mesh, and $\dim \ker L_0 = \beta_0 = 1$ as before.

**Why this miniature matters for learning.** The two-triangle disk has *no* harmonic edge features; the annulus has *exactly one*. A learned operator with access to a harmonic channel can represent the hole-circulating field and therefore the physics constrained by it — a conserved circulation, a flux threading the handle. A purely local message-passing operator, blind to $\ker L_1$, provably cannot: locality means each update sees a bounded combinatorial neighborhood, and (by the 1-WL argument of §3.1.1 lifted to this setting) two meshes that are locally indistinguishable but differ in $\beta_1$ receive identical local features, while the physics on them differs globally. The cleanest experimental design this suggests: ablate the harmonic channel and check that predictions change *exactly when the topology changes* and are unaffected when it does not. That is the Hodge theory of §2.8 rewritten as an experiment.

## 5. Synthesis: what to internalize

If the post is compressed to five sentences, they are these.

1. **The calculus.** $d_k = B_{k+1}^\top$ (Stokes as a matrix), with $d_0 = \nabla$, $d_1 = \operatorname{curl}$, $d_2 = \operatorname{div}$; the identity $B_k B_{k+1} = 0$ with its "head once, tail once" cancellation; the codifferential $\delta_k = M_{k-1}^{-1} B_k M_k$ (adjointness = integration by parts); the Hodge Laplacian $L_k = \delta_{k+1} d_k + d_{k-1}\delta_k$ with its up/down split; and the Hodge decomposition $C^k = \operatorname{im} d \oplus \operatorname{im}\delta \oplus \ker L$ with $\dim \ker L_k = \beta_k$.
2. **The bridge.** $L_0 = D - A$: all of spectral graph theory, GCN, and the Weisfeiler–Lehman expressivity story are the rank-0 corner of this calculus — and the higher ranks carry physics the vertex Laplacian cannot see.
3. **The topology.** Harmonic cochains are hole detectors: global, stable, computable topological signals (and persistence diagrams are their multi-scale companions), injecting information that provably no local message-passing scheme recovers — at the honest cost of a global solve.
4. **The geometry.** Permutation invariance via symmetric aggregators; $E(3)/SE(3)$ invariance for scalar predictions versus equivariance for fields; geodesic versus Euclidean neighborhoods as a real modeling decision, not a detail.
5. **The philosophy.** Conservation can be *encouraged* by a physics-informed loss term, or *guaranteed* by building on degree-typed cochains and the $d/\delta/L$ operators, where $\operatorname{div}\operatorname{curl} = 0$ is an identity of integer matrices. A soft penalty can be violated; an integer identity cannot.

## Further reading

- Keenan Crane, *Discrete Differential Geometry: An Applied Introduction* — the exterior-calculus and Laplacian chapters, with pictures worth many pages of algebra.
- Anil Hirani, *Discrete Exterior Calculus* (PhD thesis, 2003) — the rigorous definitions, including the circumcentric-dual Hodge star.
- Desbrun, Hirani, Leok, Marsden, *Discrete Exterior Calculus* (2005) — the compact statement of the theory.
- Hajij et al., *Topological Deep Learning: Going Beyond Graph Data*, and Papamarkou et al. (ICML 2024 position paper) — the TDL program and the combinatorial-complex abstraction.
- Kipf & Welling (GCN) and Xu et al. (GIN) — the GNN baselines and the Weisfeiler–Lehman expressivity lens.
- Arnold, Falk, Winther, *Finite Element Exterior Calculus* — the numerical-analysis ancestry of everything in Part I.
