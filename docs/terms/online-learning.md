# Online learning

Learning from each example as it arrives, instead of from a dataset in a
training run.

## Like you're twelve

Nobody sat you down with ten thousand photos of dogs before letting you meet a
dog. You met one, then another, and got better at dogs continuously — and you
were allowed to be useful about dogs the whole time, not just at the end.

<!--figure:learning-->

## Precisely

An online learner updates its state per example, in the order examples arrive,
with no separate training phase and no requirement to see the data twice. Batch
learning is the opposite: collect data, fit, freeze, deploy, and repeat the
whole cycle when the world moves.

In HeatherDB a write *is* the update, which removes the distinction between
ingesting data and learning from it. There is no minimum corpus below which the
store is useless and no retraining window during which it is stale.

The classic hazard of learning continuously is [[Catastrophic forgetting]] —
new examples overwriting old competence — which is precisely what an
[[Elastic Associative Memory (EAM)]] is shaped to avoid.

## Related

[[Catastrophic forgetting]] · [[Elastic Associative Memory (EAM)]] · [[Attractor]]
