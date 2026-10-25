# Exact generic-input rank audit

## Result

All 44 independently generated signed functional graphs passed
exact minimal-polynomial verification.

For each graph, an exact integer input vector was found whose
Krylov rank equals the predicted minimal-polynomial degree.

## Original random integer inputs

- 42/44 attained the generic rank.
- Case (4, 2): rank 2, predicted degree 3.
- Case (13, 2): rank 9, predicted degree 10.

Both exceptional cases admit exact integer input witnesses
attaining the predicted degree (3 and 10, respectively).

The discrepancies arise from nongeneric sampled input vectors,
not from incorrect polynomial degrees.

## Reproduction

Run from the repository root:

    python supplementary/S1/graph/verify_exact_genericity.py

Expected final output:

    Exact generic-rank witnesses: 44 / 44
    Nongeneric sampled inputs: 2
    PASS: independent exact degree and attainable rank verified

All matrix-rank calculations use exact SymPy arithmetic.
