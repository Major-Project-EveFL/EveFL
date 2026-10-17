"""
Pairwise-masked secure aggregation (ADR 0003, layer 2).

Step 2a (this package so far): the fixed-point codec (`fixedpoint`), the mask PRG with its no-reuse guards (`prg`) and the
injective byte encodings they share (`encoding`). The seed provider, the aggregator classes and any wiring into the federated
runner come in later steps; nothing here is connected to the FL path yet, and nothing here is a statement about the security
of a deployed system: the parties are in-process objects in tests.
"""
