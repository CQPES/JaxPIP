import json
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python3 {__file__} MOL_mol_degree.BAS")
        exit(-1)

    bas = sys.argv[1]
    out = bas.replace(".BAS", ".json")

    basis_list = []
    degree = -1

    with open(bas) as f:
        while line := f.readline():
            label, basis = line.split(":")
            current_degree = int(label.split()[0])
            basis = [int(b) for b in basis.split()]

            if current_degree != degree:
                basis_list.append([basis])
                degree = current_degree
            else:
                basis_list[-1].append(basis)

    with open(out, "w") as f:
        json.dump(basis_list, f)

    print(f"Done! Now basis of PIP in {bas} has been dumped to {out}!")
