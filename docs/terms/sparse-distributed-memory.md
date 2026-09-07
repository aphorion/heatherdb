# Sparse Distributed Memory (SDM)

Kanerva's model of memory as a huge address space, sparsely occupied, where
writing touches many places at once.

## Like you're twelve

Suppose everyone in a city keeps a copy of every rumour they hear from anyone
living within a mile. Tell one person something and it lands in hundreds of
heads. Ask a slightly different question later and enough of those heads answer
in agreement that the original rumour reassembles — even though no single
person kept it whole.

<!--figure:orthogonal-->

## Precisely

SDM is one of two classical answers to the same question, the other being the
[[Hopfield network]]. Where Hopfield stores patterns in dense recurrent weights
and hits a capacity ceiling proportional to the number of units, SDM buys
capacity with address space.

SDM addresses memory by content in a space far too large to enumerate — say 2¹⁰⁰⁰
possible addresses with a few million real locations scattered through it. A
write activates every location within a radius of the address and adds the
pattern to all of them; a read activates the same neighbourhood and sums what it
finds.

Two properties fall out and both carry into Heather. Storage is distributed, so
no single location holds an item and losing some degrades gracefully. And recall
is by similarity, so an approximate address retrieves the pattern anyway — which
makes SDM an early, explicit [[Cleanup memory]].

[[Elastic Associative Memory (EAM)]] is in this lineage, with the fixed sparse
scaffold replaced by structure that adapts to the data. The model is
[[Pentti Kanerva]]'s, from 1988; the project is named after him.

## Related

[[Hopfield network]] · [[Elastic Associative Memory (EAM)]] · [[High-dimensional space]] · [[Associative memory]]
