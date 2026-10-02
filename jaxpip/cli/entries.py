import argparse
from importlib.metadata import PackageNotFoundError, version


def get_version() -> str:
    try:
        return version("jaxpip")
    except PackageNotFoundError:
        return "unknown-dev"


def main() -> None:
    current_version = get_version()

    parser = argparse.ArgumentParser(
        prog="jaxpip",
        description="Permutation Invariant Polynomials (PIPs) in JAX.",
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {current_version}",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Sub-commands",
    )

    bas2json_parser = subparsers.add_parser(
        "bas2json",
        help="Convert MSA .BAS to JaxPIP .json",
    )

    bas2json_parser.add_argument(
        "bas_file",
        help="Path to MSA .BAS file",
    )

    bas2json_parser.add_argument(
        "json_file",
        nargs="?",
        help="Output path (optional)",
    )

    bas2json_parser.add_argument(
        "--gz",
        action="store_true",
        help="Compress with gzip",
    )

    show_parser = subparsers.add_parser(
        "show",
        help="Show JaxPIP basis info",
    )

    show_parser.add_argument(
        "basis_file", help="Path to JaxPIP basis file, .json or .json.gz"
    )

    gen_parser = subparsers.add_parser(
        "gen",
        help="Generate an MSA PIP basis directly (no .BAS / MSA binary)",
    )

    gen_parser.add_argument(
        "--mol",
        required=True,
        metavar="COUNTS",
        help="Atoms per element, e.g. --mol 2_1 for H2O, --mol 4_1 for CH4",
    )

    gen_parser.add_argument(
        "--degree",
        required=True,
        type=int,
        help="Maximum total polynomial degree",
    )

    gen_parser.add_argument(
        "json_file",
        nargs="?",
        help="Output path (optional)",
    )

    gen_parser.add_argument(
        "--gz",
        action="store_true",
        help="Compress with gzip",
    )

    args = parser.parse_args()

    if args.command == "bas2json":
        from jaxpip.basis import get_basis_info
        from jaxpip.utils import bas2json

        target_path = args.json_file

        if not target_path:
            target_path = args.bas_file.rsplit(".", 1)[0] + ".json"
            if args.gz and not target_path.endswith(".gz"):
                target_path += ".gz"

        basis_set = bas2json(
            bas_file=args.bas_file,
            json_file=target_path,
            gz=args.gz,
        )

        print(f"Converted JaxPIP basis: {target_path}")

        basis_info = get_basis_info(basis_set)

        print(basis_info)
    elif args.command == "show":
        from jaxpip.basis import get_basis_info, load_basis

        basis_set = load_basis(args.basis_file)
        basis_info = get_basis_info(basis_set)

        print(basis_info)
    elif args.command == "gen":
        import gzip
        import json as json_lib

        from jaxpip.basis import get_basis_info
        from jaxpip.basis.msa import generate

        counts = [int(x) for x in args.mol.replace(",", "_").split("_")]
        basis_set = generate(counts, args.degree)

        target_path = args.json_file
        if not target_path:
            label = "MOL_" + "_".join(str(c) for c in counts)
            target_path = f"{label}_{args.degree}.json"
        if args.gz and not target_path.endswith(".gz"):
            target_path += ".gz"

        if target_path.endswith(".json.gz"):
            with gzip.open(target_path, "wt") as f:
                json_lib.dump(basis_set, f)
        else:
            with open(target_path, "w") as f:
                json_lib.dump(basis_set, f)

        print(f"Generated JaxPIP basis: {target_path}")

        basis_info = get_basis_info(basis_set)

        print(basis_info)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
