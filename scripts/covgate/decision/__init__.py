"""Deriving the gate's decision rule from measured costs.

The gate compares base and head on the same sampled mutants (a paired
design) and rejects a change whose detection probability D dropped. Its
free choices — mutants per PR `n` and the nominal size `alpha` — are not
picked by hand: every candidate design is scored as a cost VECTOR, one
coordinate per stakeholder, and only the Pareto-efficient designs are
kept. Collapsing the vector to one number is a further, explicitly
labelled assumption (see `choose`).
"""
