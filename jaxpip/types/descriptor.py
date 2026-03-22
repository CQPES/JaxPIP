from typing import List, NamedTuple

InvariantBasis = List[List[List[int]]]


class BasisInfo(NamedTuple):
    num_atoms: int
    num_flat_mono: int
    num_poly: int
    max_degree: int
