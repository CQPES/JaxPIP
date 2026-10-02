"""MSA (monomial symmetrization approach) basis generator.

Pure-Python reimplementation of the basis-generation stage of MSA 2.0.1
(Xie, Bowman; gradients: Qu; wrapper: Wang), emitting JaxPIP's
InvariantBasis directly -- no external binary, no intermediate ``.BAS``
file. Validated to reproduce MSA output exactly (orbit partition,
within-orbit order, and orbit order) for A2B/A3B/A4B degrees 3-5 and
A3/A2B2/ABC degree 4.

Ordering rules extracted from the MSA C++ source:
  * distance index: triu row-major (i < j), identical to JaxPIP
  * group: independent permutations within each same-atom-type block
  * monomial order: total degree asc, then #nonzero entries desc,
    then lexicographic asc
  * orbit members listed in that order; orbit representative = min;
    orbits themselves ordered by their max member ascending
"""

import itertools
from typing import List, Sequence

from jaxpip.types import InvariantBasis

__all__ = ["generate", "generate_pip_basis"]


def _key(mono: Sequence[int]):
    return (sum(mono), -sum(1 for x in mono if x > 0), tuple(mono))


def _bond_permutations(atom_counts: Sequence[int]) -> List[List[int]]:
    """Distance-index permutations induced by all type-preserving
    relabelings (mole_bond in MSA's molecule_simple.hh)."""
    n = sum(atom_counts)
    offsets = [sum(atom_counts[:i]) for i in range(len(atom_counts) + 1)]

    block_perms = [
        [list(p) for p in itertools.permutations(range(offsets[b], offsets[b + 1]))]
        for b in range(len(atom_counts))
    ]

    perms = []
    for combo in itertools.product(*block_perms):
        mole = [x for blk in combo for x in blk]
        pbond = []
        for i in range(n):
            for j in range(i + 1, n):
                a, b_ = min(mole[i], mole[j]), max(mole[i], mole[j])
                pbond.append(n * a - (a + 1) * a // 2 + b_ - a - 1)
        perms.append(pbond)

    # different atom relabelings can induce the same distance permutation
    seen, unique = set(), []
    for p in perms:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            unique.append(p)
    return unique


def _compositions(total: int, m: int):
    """All nonnegative integer m-vectors summing to exactly total."""
    if m == 1:
        yield (total,)
        return
    for divs in itertools.combinations(range(total + m - 1), m - 1):
        parts = []
        prev = -1
        for d in divs:
            parts.append(d - prev - 1)
            prev = d
        parts.append(total + m - 1 - prev - 1)
        yield tuple(parts)


def generate(atom_counts: Sequence[int], max_degree: int) -> InvariantBasis:
    """Generate the MSA permutationally invariant polynomial (PIP) basis.

    Arguments:
        atom_counts: Number of atoms of each element, e.g. ``(2, 1)`` for
            H2O and ``(4, 1)`` for CH4.
        max_degree: Maximum total polynomial degree.

    Returns:
        InvariantBasis: identical in content and ordering to what
            ``bas2json`` produces from the corresponding MSA ``.BAS``
            file, directly consumable by ``PolynomialDescriptor``.
    """
    n = sum(atom_counts)
    m = n * (n - 1) // 2
    pbonds = _bond_permutations(atom_counts)

    orbits = {}
    for deg in range(max_degree + 1):
        for mono in _compositions(deg, m):
            members = {tuple(mono[p] for p in pb) for pb in pbonds}
            members.add(mono)
            rep = min(members, key=_key)
            orbits.setdefault(rep, set()).update(members)

    basis = [sorted(members, key=_key) for members in orbits.values()]
    basis.sort(key=lambda orbit: _key(orbit[-1]))
    return [[list(x) for x in orbit] for orbit in basis]


# explicit alias: this generates the PIP (MSA) flavor of invariant basis
generate_pip_basis = generate
