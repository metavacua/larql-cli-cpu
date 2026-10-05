"""The replay experiment: measure what the gate's decision needs and no
history can supply — how often base and head disagree about a mutant
(discordance), how far D falls when it falls, what a mutant costs to run,
and how many past commits still build.

Past merged PRs touching the crate are replayed: the same sampled mutants
run on the PR's base and on its head (a paired design). The first run is
a pilot under a declared runner budget; the main experiment's size is
derived from the pilot by power analysis.
"""
