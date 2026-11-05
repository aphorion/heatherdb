# Catastrophic forgetting

A substantial, unwanted loss of previously learned behaviour while learning
new data or a new task. Shared representations can change in ways that impair
earlier tasks; the severity depends on the data, architecture and learning rule.
Replay, regularisation and separate capacity are among the ways to limit it.

This differs from [graceful forgetting](graceful-forgetting.md), where earlier
patterns gradually lose influence as the system adapts to a changed world.
Whether retention or adaptation is desirable depends on the application's policy.
Measure both rather than treating all change as failure or all forgetting as
useful learning.

HeatherDB's competitive mode learns shared patterns; it does not guarantee
immunity to interference. Preserve original evidence separately when needed.
See [Choose a memory mode](../how-to/choose-memory-mode.md).
