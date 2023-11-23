import json
import re
import sys

if len(sys.argv) != 2:
    print(f"Usage: python3 {__file__} MOL_mol_degree.BAS")
    exit(-1)

inp = sys.argv[1]
out = re.sub(r"BAS", r"json", inp)

basis_list = []
degree = -1

with open(inp) as f:
    contents = f.readlines()
    for line in contents:
        label, basis = line.split(r":")
        current_degree = int(label.split()[0])
        basis = [int(b) for b in basis.split()]

        if current_degree != degree:
            basis_list.append([basis])
            degree = current_degree
        else:
            basis_list[-1].append(basis)

with open(out, "w") as f:
    json.dump(basis_list, f)


print(f"Done! Now basis of PIP in {inp} has been dumped to {out}!")
