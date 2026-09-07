# Hopfield network

The 1982 model that made "memory as a landscape you fall into" precise — and the
direct ancestor of how a read works here.

## Like you're twelve

Imagine a bedsheet pulled tight, and you press marbles into it at a few spots so
it dents. Now roll a new marble anywhere on the sheet. It does not stay where
you put it; it rolls downhill until it drops into one of the dents you made.

Whatever you roll in, one of the remembered things comes out. That is a Hopfield
network. The dents are the memories, the rolling is the remembering, and nobody
had to search a list to find the answer.

<!--figure:attractor-->

## Precisely

A Hopfield network is a set of units wired symmetrically to each other, with the
weights set so that the patterns you want to store sit at the minima of an
energy function. Update the units and the energy can only go down, so the state
slides into the nearest minimum and stops. Those minima are the
[[Attractor|attractors]], and recall is convergence rather than lookup: feed in a
fragment or a corrupted copy, and the dynamics complete it.

That is [[Associative memory]] with the mathematics attached — content
addressing, graceful degradation, and completion from partial input all falling
out of one energy landscape.

The classical version has a hard limit. A network of *N* units stores roughly
0.14*N* random patterns; push past that and the landscape develops spurious
minima — dents that correspond to nothing you stored — and recall collapses
rather than degrading. Capacity, not correctness, is what kept the model
academic for thirty years.

Two things changed that. [[Sparse Distributed Memory (SDM)]] reached the same
behaviour through a huge sparse address space instead of dense recurrent
weights, and modern Hopfield networks (2020) showed that with a different energy
function capacity becomes exponential in the dimension — and that the resulting
update rule is the attention mechanism used in transformers. Retrieval from a
memory and attention over a set turn out to be the same operation seen twice.

## Where it shows up here

HeatherDB's read is a settle in this tradition: a query is not matched against
rows, it is relaxed toward the pattern its neighbourhood supports, and
[[Fidelity|fidelity]] reports how far it travelled. What
[[Elastic Associative Memory (EAM)]] changes is the landscape itself — instead of
weights fixed by a storage rule and a capacity ceiling fixed by *N*, the
structure adapts as data arrives, which is what keeps writing cheap and
capacity from falling off a cliff.

John Hopfield shared the 2024 Nobel Prize in Physics for this line of work.

## Related

[[Attractor]] · [[Associative memory]] · [[Sparse Distributed Memory (SDM)]] · [[Pentti Kanerva]]
