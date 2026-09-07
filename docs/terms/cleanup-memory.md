# Cleanup memory

The thing that takes a smudged vector and hands back the clean item it was
reaching for.

## Like you're twelve

Someone reads out a phone number over a bad line and you catch most of it. You
do not invent the missing digits — you look at the list of people it could be
and one of them fits. What you heard was approximate; what you dial is exact.

Cleanup is that step. Something noisy comes in, the closest known thing goes
out.

<!--figure:cleanup-->

## Precisely

Every symbolic vector operation leaves its result approximate:
[[Unbind|unbinding]] returns the filler plus interference,
[[Superposition|superposed]] terms carry each other's noise. Composition beyond
a step or two is only usable if the noise can be discharged, and discharging it
means mapping the approximate vector onto the nearest stored item.

Conventional implementations bolt on an item memory — a list of every clean
vector — and scan it by similarity. It works at demo scale and becomes a
liability at data scale, and it has to be maintained separately from wherever
the real data lives.

An [[Associative memory]] does this natively: its read path *is* cleanup, run
against the stored data rather than a side list. That is the structural reason
[[Vector symbolic architecture|VSA]] and Heather fit together.

## Related

[[Associative memory]] · [[Unbind]] · [[Attractor]]
