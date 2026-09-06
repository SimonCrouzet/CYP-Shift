"""Structure-based features from Boltz-2 cofolding of a CYP with an inhibitor.

Two heads, both fed by one cofold: the **affinity** the Boltz-2 affinity module predicts
for the complex (head A), and **pose/interaction** features read off the predicted
structure -- how close the inhibitor's coordinating atom sits to the heme iron, and the
model's own confidence (head B). See :mod:`cypshift.structure.cofold`.

Cofolding is GPU work and runs once per (enzyme, compound) into a cache; the pipeline
reads the cached features, so a sweep never re-runs Boltz.
"""
