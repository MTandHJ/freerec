

import argparse
from .data.tags import USER, ITEM, RATING, TIMESTAMP


def tune(args):
    # Grid search for hyper-parameters can be conducted by:
    #
    #    freerec tune NAME_OF_EXPERIMENT CONFIG.yaml
    #
    from .parser import CoreParser
    from .launcher import Adapter
    cfg = CoreParser()
    cfg.compile(args)

    tuner = Adapter()
    tuner.compile(cfg)
    tuner.fit()


def make(args):
    # Make dataset:
    #
    #    freerec make DATASET --root=../data
    #
    from .data.preprocessing.base import AtomicConverter
    from .data.preprocessing import datasets
    from .data.tags import USER, SESSION, ITEM, TIMESTAMP

    converter = AtomicConverter(
        root=args.root, dataset=args.dataset, filedir=args.filedir,
        userColname=
    )

    star4pos = args.star4pos
    kcore4user = args.kcore4user
    kcore4item = args.kcore4item
    ratios = tuple(map(int, args.ratios.split(',')))
    days = args.days
    strict = not args.not_strict
    


def main():
    parser = argparse.ArgumentParser("FreeRec")
    subparsers = parser.add_subparsers()


    tune_parser = subparsers.add_parser("tune")

    tune_parser.set_defaults(func=tune)
    tune_parser.add_argument("description", type=str, help="...")
    tune_parser.add_argument("config", type=str, help="config.yml")
    tune_parser.add_argument("--exclusive", action="store_true", default=False, help="one by one or one for all")

    tune_parser.add_argument("--root", type=str, default=None, help="data")
    tune_parser.add_argument("--dataset", type=str, default=None, help="useless if no need to automatically select a dataset")
    tune_parser.add_argument("--device", type=str, default=None, help="device")
    tune_parser.add_argument("--num-workers", type=int, default=None)

    tune_parser.add_argument("--resume", action="store_true", default=False, help="resume the search from the recent checkpoint")


    make_parser = subparsers.add_parser("make")
    make_parser.set_defaults(func=make)

    make_parser.add_argument("dataset", type=str, help="output dataset name")
    make_parser.add_argument("--root", type=str, default=".", help="data")
    make_parser.add_argument(
        "--filedir", type=str, default=None, 
        help="filedir saving data. Using `dataset` instead if None"
    )

    make_parser.add_argument(
        "--splitting", type=str, choices=('ROU', 'ROD', 'LOU', 'DOU', 'DOD'), default='ROU',
        help="ROU: Ratio On User; ROD: Ratio On Dataset; LOU: Leave-One-Out; DOU: Day on User; DOD: Day on Dataset"
    )

    make_parser.add_argument("-sp", "--star4pos", type=int, default=0, help="select interactions with `Rating >= star4pos'")
    make_parser.add_argument("-ku", "--kcore4user", type=int, default=10, help="select kcore interactions according to User")
    make_parser.add_argument("-ki", "--kcore4item", type=int, default=10, help="select kcore interactions according to Item")
    make_parser.add_argument("-rs", "--ratios", type=str, default="8,1,1", help="the ratios of training|validation|test set")
    make_parser.add_argument("--days", type=int, default=7, help="the second last days for validation and last days for test")

    make_parser.add_argument("-uc", "--userColname", type=str, default=USER.name, help="the column name of User ID")
    make_parser.add_argument("-ic", "--itemColname", type=str, default=ITEM.name, help="the column name of Item ID")
    make_parser.add_argument("-rc", "--ratingColname", type=str, default=RATING.name, help="the column name of Rating")
    make_parser.add_argument("-tc", "--timestampColname", type=str, default=TIMESTAMP.name, help="the column name of Timestamp")

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()