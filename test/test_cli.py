"""CLI smoke tests: gen / show / bas2json roundtrip."""

import json

import pytest

from jaxpip.cli.entries import main


def run_cli(capsys, argv):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("sys.argv", ["jaxpip"] + argv)
        main()
    return capsys.readouterr().out


def test_gen_and_show_roundtrip(capsys, tmp_path):
    out = str(tmp_path / "MOL_2_1_5.json")
    run_cli(capsys, ["gen", "--mol", "2_1", "--degree", "5", out])

    from jaxpip.basis.msa import generate

    with open(out) as f:
        assert json.load(f) == generate((2, 1), 5)

    out2 = run_cli(capsys, ["show", out])
    assert "num_atoms=3" in out2
    assert "num_poly=34" in out2
    assert "num_flat_mono=56" in out2


def test_gen_default_filename_and_gz(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_cli(capsys, ["gen", "--mol", "4_1", "--degree", "3", "--gz"])
    produced = tmp_path / "MOL_4_1_3.json.gz"
    assert produced.exists()

    import gzip

    with gzip.open(produced, "rt") as f:
        basis = json.load(f)
    assert len(basis) == 30
    assert sum(len(o) for o in basis) == 286
