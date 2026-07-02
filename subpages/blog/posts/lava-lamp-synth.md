This is the project where my research and my hobbies collide. **Dynamical Systems Orchestration** turns a lava lamp — a genuinely stochastic, analogue, thermodynamic system — into a musical instrument, by filming it with a single camera and translating what it sees, in real time, into continuously evolving sound.

The tracking pipeline runs on the same topological machinery I use in my research. Each wax blob is detected by **H0 persistent homology** (persistence as a significance filter, rather than a crude area or brightness threshold), given a stable identity through optical-flow mask advection and Wasserstein-inspired matching in the birth–death plane, and smoothed by a physics-informed Kalman filter that knows blobs decelerate at the top and bottom of the lamp. Every blob's position, speed, geometry, and hue-derived "temperature" becomes a continuous control signal; merges and splits become harmonic and dissonant events.

## A duet, not a translator

The current version (v4.0) reframes the whole thing around the *voice* rather than the blob. Sixteen polyphonic synth slots can be driven entirely by the lamp, entirely by a human at a keyboard, or in a hybrid mode where the player sets pitch and timbre while the blob keeps modulating filter, resonance, pan, and amplitude. The loop no longer closes through the lamp's own sound — it closes through the player's ears and hands.

Full write-up, videos, architecture diagrams, a parts list, and the build guides are all on the project page.

[See the project →](https://samleventhal.com/subpages/lava_lamp_synth/)
