# Graceful forgetting

The gradual reduction of an earlier pattern's influence as an adaptive memory
learns changing patterns. It is useful when current knowledge should take
priority over reproducing every past observation.

Old-pattern fidelity measures retention; falling fidelity can be the intended
consequence of adaptation. It still needs to be measured alongside learning of
the new patterns. Abrupt, unwanted loss of a needed skill is a different issue:
[catastrophic forgetting](catastrophic-forgetting.md).

HeatherDB's competitive mode can exhibit this behaviour. It does not promise
a fixed forgetting schedule, expiration, or removal of all historical influence.
See [Learning and graceful forgetting](../explanation/learning-and-graceful-forgetting.md)
and [Choose a memory mode](../how-to/choose-memory-mode.md).
