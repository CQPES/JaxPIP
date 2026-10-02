import gzip
import json

import chex
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpip.basis import get_basis_info, load_basis
from jaxpip.utils import bas2json

from conftest import WATER_BASIS


BAS_TEXT = """\
   0  0  0  0 : 0 0 0
   1  1  1  1 : 0 0 1
   1  1  2  1 : 0 1 0
   2  1  1  1 : 1 0 0
   3  2  1  1 : 0 1 1
   4  2  2  1 : 1 0 1
   4  2  2  2 : 1 1 0
   5  3  1  1 : 0 0 2
   5  3  2  1 : 0 2 0
   6  4  1  1 : 2 0 0
"""


def test_bas2json_groups_consecutive_degrees_into_orbits(tmp_path):
    bas = tmp_path / "MOL_2_1_5.BAS"
    bas.write_text(BAS_TEXT)
    out = str(tmp_path / "basis.json")
    basis = bas2json(str(bas), out, False)

    assert basis == WATER_BASIS
    assert load_basis(out) == WATER_BASIS


def test_bas2json_gz_roundtrip(tmp_path):
    bas = tmp_path / "MOL_2_1_5.BAS"
    bas.write_text(BAS_TEXT)
    out = str(tmp_path / "basis.json.gz")
    basis = bas2json(str(bas), out, True)
    assert out.endswith(".json.gz")
    with gzip.open(out, "rt") as f:
        assert json.load(f) == basis
    assert load_basis(out) == WATER_BASIS


def test_basis_info_consistency(water_desc):
    info = water_desc.basis_info
    assert info.num_atoms == 3
    assert info.num_poly == len(WATER_BASIS)
    assert info.num_flat_mono == sum(len(o) for o in WATER_BASIS)
    assert info.max_degree == 2


def test_basis_info_from_bas_matches(tmp_path):
    bas = tmp_path / "MOL_2_1_5.BAS"
    bas.write_text(BAS_TEXT)
    out = str(tmp_path / "basis.json")
    basis = bas2json(str(bas), out, False)
    info = get_basis_info(basis)
    assert (info.num_atoms, info.num_poly, info.num_flat_mono, info.max_degree) == \
        (3, 7, 10, 2)


def test_descriptor_reproducible_across_instances():
    """Two descriptor instances from the same basis agree bitwise."""
    from jaxpip.descriptor import PolynomialDescriptor
    from conftest import WATER_XYZ

    d1 = PolynomialDescriptor(basis_set=WATER_BASIS, alpha=1.0)
    d2 = PolynomialDescriptor(basis_set=WATER_BASIS, alpha=1.0)
    xyz = jnp.asarray(WATER_XYZ)
    chex.assert_trees_all_close(d1(xyz), d2(xyz))
