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
        help="Conver MSA .BAS to JaxPIP .json",
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
    else:
        parser.print_help()
