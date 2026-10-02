import gzip
import json

from jaxpip.types import InvariantBasis


def bas2json(
    bas_file: str,
    json_file: str,
    gz: bool,
) -> InvariantBasis:
    if gz and not json_file.endswith(".gz"):
        json_file += ".gz"

    basis_set: InvariantBasis = []
    orbit_id = -1

    with open(bas_file, "r") as f:
        for line in f:
            line = line.strip()

            if not line or ":" not in line:
                continue

            label_part, basis_part = line.split(":")

            # first number is the polynomial (orbit) id, ascending per orbit
            current_id = int(label_part.split()[0])
            exponents = [int(b) for b in basis_part.split()]

            if current_id != orbit_id:
                basis_set.append([exponents])
                orbit_id = current_id
            else:
                basis_set[-1].append(exponents)

    if json_file.endswith(".json.gz"):
        with gzip.open(json_file, "wt") as f:
            json.dump(basis_set, f)
    else:
        with open(json_file, "w") as f:
            json.dump(basis_set, f)

    return basis_set
