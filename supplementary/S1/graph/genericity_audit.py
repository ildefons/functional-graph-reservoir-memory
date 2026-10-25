"""Exact-arithmetic audit of generic scalar-input Krylov rank.

For a signed functional-graph matrix P, the minimal-polynomial degree d(P)
is the maximal possible rank of [b, Pb, ..., P^(N-1)b]. A randomly chosen
integer input can be nongeneric and have smaller rank, but never larger.

Use with the existing supplementary/S1/graph/verify_exact.py:

    from genericity_audit import audit_matrix
    audit_matrix(P, case_id=graph_index, predicted_degree=d_formula,
                 original_b=sampled_integer_b)

All arguments except P are optional. P must contain exact integers or
rationals (typically 0, +1, -1), NOT rounded or float-normalized weights.
The audit never changes the network or the manuscript.

Run `python genericity_audit.py --self-test` for independent test cases.
"""

import argparse
import hashlib
import json
import random
from typing import Optional

import sympy as sp


def _matrix_exact(P):
    A = sp.Matrix(P)
    if A.rows != A.cols or A.rows == 0:
        raise ValueError('P must be a nonempty square matrix')
    for a in A:
        if not a.is_Rational:
            raise ValueError(f'P has a non-exact entry {a!r}; use integer graph matrix P, not scaled float W')
    return A


def _vector_exact(b, n):
    v = sp.Matrix(list(b))
    if v.shape != (n, 1):
        raise ValueError(f'Input b must have length {n}; got {v.shape}')
    if not all(x.is_Rational for x in v):
        raise ValueError('Input vector must have exact integer or rational entries')
    return v


def minimal_polynomial_degree_exact(P):
    """Find the first exact dependence of I,P,...,P^N over Q.

    It is an independent, exact method for d(P), requiring no guessed
    eigenvalues, rank thresholds, or graph formula.
    """
    A = _matrix_exact(P)
    n = A.rows
    power = sp.eye(n)
    cols = []
    for k in range(n + 1):
        columns_before = len(cols)
        cols.append(sp.Matrix(n * n, 1, lambda i, _: power[i // n, i % n]))
        rank = sp.Matrix.hstack(*cols).rank()
        if rank < columns_before + 1:
            return k
        power = power * A
    raise AssertionError('Cayley-Hamilton failure: powers remained independent through N')


def krylov_rank_exact(P, b, columns: Optional[int] = None):
    """Integer/rational arithmetic throughout, never a numerical threshold."""
    A = _matrix_exact(P)
    n = A.rows
    v = _vector_exact(b, n)
    count = n if columns is None else int(columns)
    if not 1 <= count <= n:
        raise ValueError('columns must be in [1, N]')
    vectors = []
    for _ in range(count):
        vectors.append(v)
        v = A * v
    return sp.Matrix.hstack(*vectors).rank()


def audit_matrix(P, *, case_id=None, predicted_degree=None, original_b=None,
                 max_trials=128, seed=20261025, emit=True):
    """Certify d(P) and find a reproducible, EXACT full-rank input witness.

    A witness b with rank K(P,b)=d(P) proves the predicted generic rank is
    attainable on that particular graph. Combined with rank K<=d(P) for
    every b, this establishes genericity (the lower-rank set is a proper
    algebraic variety, hence measure zero for continuous b).

    Returns a record with the witness and exact ranks. Raises on theorem-
    contradicting degrees, impossible ranks, or failure to find a witness.
    """
    A = _matrix_exact(P)
    n = A.rows
    degree = minimal_polynomial_degree_exact(A)
    expected = None if predicted_degree is None else int(predicted_degree)
    if expected is not None and degree != expected:
        raise AssertionError(
            f'{case_id}: formula degree {expected} != exact matrix degree {degree}; '
            'investigate polynomial orientation/construction before invoking genericity')

    original_rank = None
    if original_b is not None:
        original_rank = krylov_rank_exact(A, original_b, columns=degree)
        if original_rank > degree:
            raise AssertionError(f'{case_id}: exact rank {original_rank} > degree {degree}; impossible')

    # Enumerate reproducible small integer vectors, with a deterministic
    # label-specific seed. Exact verification of the witness is a certificate.
    label_bytes = str(case_id).encode('utf-8')
    label_salt = int.from_bytes(hashlib.sha256(label_bytes).digest()[:8], 'big')
    rng = random.Random(int(seed) + label_salt)
    witness = None
    witness_rank = None
    trials = 0
    for trial in range(1, max_trials + 1):
        # No dependence on NumPy floating-point rank heuristics.
        b = [rng.randint(-8, 8) for _ in range(n)]
        if not any(b):
            continue
        rank = krylov_rank_exact(A, b, columns=degree)
        trials = trial
        if rank == degree:
            witness, witness_rank = b, rank
            break
    if witness is None:
        raise AssertionError(f'{case_id}: no cyclic vector found in {max_trials} trials; '
                             'inspect graph construction and the formula')

    result = dict(case_id=case_id, N=n, exact_minpoly_degree=degree,
                  predicted_degree=expected, original_input_rank=original_rank,
                  witness_rank=witness_rank, witness=witness, witness_trial=trials,
                  interpretation=(
                      'nongeneric original input' if original_rank is not None and original_rank < degree
                      else 'original input achieves the generic rank' if original_rank == degree
                      else 'generic-rank witness found; original input not supplied'))
    if emit:
        print(json.dumps(result, sort_keys=True))
    return result


def _make_signed_functional_graph(n, rng):
    """Each row has one signed off-diagonal source, as in the paper."""
    A = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        source = rng.randrange(n - 1)
        if source >= i:
            source += 1
        A[i][source] = rng.choice((-1, +1))
    return A


def _self_test():
    cycle = [[0,1,0],[0,0,1],[1,0,0]]
    a = audit_matrix(cycle, case_id='3cycle-intentionally-nongeneric',
                     predicted_degree=3, original_b=[1,1,1], emit=False)
    assert a['original_input_rank'] == 1
    assert a['witness_rank'] == 3

    graph = [[0,1,0,0],[0,0,1,0],[1,0,0,0],[1,0,0,0]]
    a = audit_matrix(graph, case_id='3cycle-plus-tail',
                     predicted_degree=4, original_b=[1,1,1,1], emit=False)
    assert a['original_input_rank'] == 1
    assert a['witness_rank'] == 4

    rng = random.Random(20261025)
    for i in range(44):
        A = _make_signed_functional_graph(3 + i % 10, rng)
        a = audit_matrix(A, case_id=i, emit=False)
        assert a['witness_rank'] == a['exact_minpoly_degree']
    print('PASS: 2 deliberately nongeneric examples diagnosed correctly')
    print('PASS: exact generic-rank witnesses on 44 fresh signed functional graphs')
    print('NOTE: these 44 self-test graphs are NOT the 44 instances in verify_exact.py')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    else:
        parser.print_help()
