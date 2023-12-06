# Examples

In this section, I will show you how to generate PIP basis and use `JaxPIP` to
calculate the PIP descriptor and its Jacobian matrix with respect to the
Cartesian coordinates.

## CH<sub>4</sub>: A Permutation Invariant Polynomial Tutorial

In the folder `CH4`, there are some files:

1. `CH4.xyz`
2. `prepare_basis.sh`
3. `demo.ipynb`

The `CH4.xyz` file contains Cartesian coordiantes of a methane molecule. The
`prepare_basis.sh` shell script will clone the `MSA-2.0` repo and generated
required files for `JaxPIP`. For details please refer to the comments in it.

First, run the command below to prepare the files.

```bash
$ bash prepare_basis.sh
```

During this progress, executable `msa` will be compiled, PIP basis will be
generated, then the `*.BAS` file will be converted to `json` format, which is
required by `JaxPIP.PIPDescriptor`. The `BAS2json.py` is a small utility shipped
with `JaxPIP`.

Since you already have the basis, now you can go to the `demo.ipynb` and try to
run it.
