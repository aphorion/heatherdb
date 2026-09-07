# Catastrophic forgetting

What happens when learning something new destroys something already learned.

## Like you're twelve

Imagine that learning your new address wiped your old one, your street, and the
route home. That is not how your head works — but it is roughly how a neural
network behaves when you train it on a new task. The weights that held the old
skill get reused for the new one, and the old skill goes.

<!--figure:memory-->

## Precisely

In a network, knowledge is stored in shared weights, and gradient descent on new
data has no reason to preserve the configurations that encoded old data. Train
sequentially on task A then task B and performance on A collapses. The standard
mitigations — replay buffers, regularisation toward old weights, frozen layers —
manage the symptom at a cost, which is why most deployed models are simply
frozen after training and retrained wholesale later.

This is the practical reason model-native systems cannot keep learning in
production, and the reason a store that learns per write has to be built
differently: new patterns must settle *alongside* old ones rather than compete
for the same capacity.

## Related

[[Online learning]] · [[Elastic Associative Memory (EAM)]] · [[Superposition]]
