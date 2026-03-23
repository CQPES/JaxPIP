import gzip
import json
from typing import List, Tuple

from jaxpip.types import BasisInfo, InvariantBasis


def load_basis(
    basis_file: str,
) -> InvariantBasis:
    if basis_file.endswith(".json.gz"):
        with gzip.open(basis_file, "rt") as f:
            basis_set = json.load(f)
    else:
        with open(basis_file, "r") as f:
            basis_set = json.load(f)

    return basis_set


def get_basis_info(
    basis_set: InvariantBasis,
) -> BasisInfo:
    num_poly = len(basis_set)
    num_flat_mono = sum(len(b) for b in basis_set)

    num_dist = len(basis_set[0][0])
    num_atoms = int((1 + (1 + 8 * num_dist) ** 0.5) / 2)
    max_degree = max(sum(m) for b in basis_set for m in b)

    return BasisInfo(
        num_atoms=num_atoms,
        num_flat_mono=num_flat_mono,
        num_poly=num_poly,
        max_degree=max_degree,
    )


def flatten_basis(
    basis_set: InvariantBasis,
) -> Tuple[List[int], List[int]]:
    exponents = [m for b in basis_set for m in b]
    segments = [i for i, b in enumerate(basis_set) for _ in b]

    return exponents, segments
