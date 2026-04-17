r"""Command-line entry point for the FreeRec package.

Provides the ``freerec`` CLI with subcommands ``skill``, ``tune``, and
``make``.
"""

import argparse


def skill(args):
    r"""Display skill-based tutorials and information for AI assistants.

    Parameters
    ----------
    args : :class:`argparse.Namespace`
        Parsed command-line arguments containing boolean flags ``make``,
        ``tune``, ``log``, and ``workflow``.
    """
    # Get skill information:
    #
    #    freerec skill --make
    #    freerec skill --tune
    #    freerec skill --log
    #    freerec skill --workflow
    #
    # Import only skills module (no heavy dependencies)
    import os

    skills_path = os.path.join(os.path.dirname(__file__), "skills.py")

    # Read and execute skills.py in isolation
    with open(skills_path, "r", encoding="utf-8") as f:
        skills_code = f.read()

    skills_ns = {}
    exec(skills_code, skills_ns)

    # Determine which skill to show
    skill_name = None
    if args.make:
        skill_name = "make"
    elif args.tune:
        skill_name = "tune"
    elif args.log:
        skill_name = "log"
    elif args.workflow:
        skill_name = "workflow"

    if skill_name is None:
        print("Available skills:")
        for name in skills_ns["list_skills"]():
            skill_info = skills_ns["get_skill"](name)
            print(f"  - {name}: {skill_info['description']}")
        print("\nUse 'freerec skill --<skill_name>' to get detailed information.")
        return

    skills_ns["print_skill"](skill_name)


def tune(args):
    r"""Run grid-search hyperparameter tuning.

    Parameters
    ----------
    args : :class:`argparse.Namespace`
        Parsed command-line arguments including ``description`` and
        ``config`` path.
    """
    # Grid search for hyper-parameters can be conducted by:
    #
    #    freerec tune NAME_OF_EXPERIMENT CONFIG.yaml
    #
    from .launcher import Adapter
    from .parser import CoreParser

    cfg = CoreParser()
    cfg.compile(args)

    tuner = Adapter()
    tuner.compile(cfg)
    tuner.fit()


def make(args):
    r"""Convert raw interaction data into a FreeRec-compatible dataset.

    Parameters
    ----------
    args : :class:`argparse.Namespace`
        Parsed command-line arguments including ``dataset``, ``root``,
        splitting strategy, and filtering options.
    """
    # Make dataset:
    #
    #    freerec make DATASET --root=../data
    #
    from .data.preprocessing.base import AtomicConverter

    converter = AtomicConverter(
        root=args.root,
        dataset=args.dataset,
        filedir=args.filedir,
        userColname=args.userColname,
        itemColname=args.itemColname,
        ratingColname=args.ratingColname,
        timestampColname=args.timestampColname,
    )

    converter.make_dataset(
        args.kcore4user,
        args.kcore4item,
        args.star4pos,
        splitting=args.splitting,
        ratios=tuple(map(int, args.ratios.split(","))),
        days=args.days,
    )


TORCHDATA_VERSION = "0.7.0"


def setup(args):
    r"""Install torchdata without overriding the existing torch installation."""
    import subprocess
    import sys

    # 1. Check torch
    try:
        import torch

        print(f"torch {torch.__version__} detected.")
    except ImportError:
        print("Error: torch is not installed.")
        print("Please install torch first: https://pytorch.org/get-started/locally/")
        sys.exit(1)

    # 2. Check torchdata
    try:
        import torchdata

        if torchdata.__version__ == TORCHDATA_VERSION:
            print(f"torchdata {TORCHDATA_VERSION} already installed, skipping.")
            return
        else:
            print(
                f"torchdata {torchdata.__version__} found, replacing with {TORCHDATA_VERSION}..."
            )
    except ImportError:
        print(f"Installing torchdata=={TORCHDATA_VERSION} (--no-deps)...")

    # 3. Install
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                f"torchdata=={TORCHDATA_VERSION}",
                "--no-deps",
            ]
        )
    except subprocess.CalledProcessError:
        print(f"Error: failed to install torchdata=={TORCHDATA_VERSION}.")
        sys.exit(1)

    print(f"torchdata=={TORCHDATA_VERSION} installed successfully.")


def main():
    r"""Parse CLI arguments and dispatch to the appropriate subcommand."""
    parser = argparse.ArgumentParser("FreeRec")
    subparsers = parser.add_subparsers()

    skill_parser = subparsers.add_parser(
        "skill",
        help="Get skill-based tutorials and information for AI assistants",
        description="Freerec Skill System - Provides detailed tutorials and guides for understanding freerec's internal mechanisms.",
        epilog="Use 'freerec skill --<skill_name>' to get detailed information about a specific skill.",
    )
    skill_parser.set_defaults(func=skill)
    skill_parser.add_argument(
        "--make",
        action="store_true",
        default=False,
        help="Dataset processing tutorial: how to convert raw data into freerec format (splitting strategies, k-core filtering, output structure)",
    )
    skill_parser.add_argument(
        "--tune",
        action="store_true",
        default=False,
        help="Hyperparameter tuning tutorial: grid search with parallel execution, config format, result interpretation",
    )
    skill_parser.add_argument(
        "--log",
        action="store_true",
        default=False,
        help="Training log analysis guide: log file structure, metrics explanation, how to read best.pkl and monitors.pkl",
    )
    skill_parser.add_argument(
        "--workflow",
        action="store_true",
        default=False,
        help="Training pipeline overview: complete architecture from data loading to model evaluation, component interactions",
    )

    tune_parser = subparsers.add_parser("tune")

    tune_parser.set_defaults(func=tune)
    tune_parser.add_argument("description", type=str, help="...")
    tune_parser.add_argument("config", type=str, help="config.yml")
    tune_parser.add_argument(
        "--exclusive",
        action="store_true",
        default=False,
        help="one by one or one for all",
    )

    tune_parser.add_argument("--root", type=str, default=None, help="data")
    tune_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="useless if no need to automatically select a dataset",
    )
    tune_parser.add_argument("--device", type=str, default=None, help="device")
    tune_parser.add_argument("--num-workers", type=int, default=None)

    tune_parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="resume the search from the recent checkpoint",
    )

    make_parser = subparsers.add_parser("make")
    make_parser.set_defaults(func=make)

    make_parser.add_argument("dataset", type=str, help="output dataset name")
    make_parser.add_argument(
        "--root", type=str, default=".", help="data, default to '.'"
    )
    make_parser.add_argument(
        "--filedir",
        type=str,
        default=None,
        help="filedir saving data. Using `dataset` instead if None (default)",
    )

    make_parser.add_argument(
        "--splitting",
        type=str,
        choices=("ROU", "RAU", "ROD", "LOU", "DOU", "DOD"),
        default="ROU",
        help="ROU: Ratio On User (default); 'RAU': Ratio and At least one on User; ROD: Ratio On Dataset; LOU: Leave-one-out On User; DOU: Day On User; DOD: Day On Dataset",
    )

    make_parser.add_argument(
        "-sp",
        "--star4pos",
        type=int,
        default=0,
        help="select interactions with `Rating >= star4pos (default: 0)'",
    )
    make_parser.add_argument(
        "-ku",
        "--kcore4user",
        type=int,
        default=10,
        help="select kcore (default: 10) interactions according to User",
    )
    make_parser.add_argument(
        "-ki",
        "--kcore4item",
        type=int,
        default=10,
        help="select kcore interactions (default: 10) according to Item",
    )
    make_parser.add_argument(
        "-rs",
        "--ratios",
        type=str,
        default="8,1,1",
        help="the ratios (default: 8,1,1) of training|validation|test set",
    )
    make_parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="the second last days (default: 7) for validation and last days (default: 7) for test",
    )

    make_parser.add_argument(
        "-uc",
        "--userColname",
        type=str,
        default="USER",
        help="the column name (default: USER) of User ID",
    )
    make_parser.add_argument(
        "-ic",
        "--itemColname",
        type=str,
        default="ITEM",
        help="the column name (default: ITEM) of Item ID",
    )
    make_parser.add_argument(
        "-rc",
        "--ratingColname",
        type=str,
        default="RATING",
        help="the column name (default: RATING) of Rating",
    )
    make_parser.add_argument(
        "-tc",
        "--timestampColname",
        type=str,
        default="TIMESTAMP",
        help="the column name (default: TIMESTAMP) of Timestamp",
    )

    setup_parser = subparsers.add_parser(
        "setup",
        help=f"Install torchdata=={TORCHDATA_VERSION} without overriding existing torch",
    )
    setup_parser.set_defaults(func=setup)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
