

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

    converter = AtomicConverter(
        root=args.root, dataset=args.dataset, filedir=args.filedir,
        userColname=args.userColname, itemColname=args.itemColname,
        ratingColname=args.ratingColname, timestampColname=args.timestampColname
    )

    converter.make_dataset(
        args.kcore4user, args.kcore4item, args.star4pos,
        splitting=args.splitting, 
        ratios=tuple(map(int, args.ratios.split(','))),
        days=args.days
    )

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
    make_parser.add_argument(
        "--root", type=str, default=".", 
        help="data, default to '.'"
    )
    make_parser.add_argument(
        "--filedir", type=str, default=None, 
        help="filedir saving data. Using `dataset` instead if None (default)"
    )

    make_parser.add_argument(
        "--splitting", type=str, choices=('ROU', 'ROD', 'LOU', 'DOU', 'DOD'), default='ROU',
        help="ROU: Ratio On User (default); ROD: Ratio On Dataset; LOU: Leave-one-out On User; DOU: Day On User; DOD: Day On Dataset"
    )

    make_parser.add_argument("-sp", "--star4pos", type=int, default=0, help="select interactions with `Rating >= star4pos (default: 0)'")
    make_parser.add_argument("-ku", "--kcore4user", type=int, default=10, help="select kcore (default: 10) interactions according to User")
    make_parser.add_argument("-ki", "--kcore4item", type=int, default=10, help="select kcore interactions (default: 10) according to Item")
    make_parser.add_argument("-rs", "--ratios", type=str, default="8,1,1", help="the ratios (default: 8,1,1) of training|validation|test set")
    make_parser.add_argument("--days", type=int, default=7, help="the second last days (default: 7) for validation and last days (default: 7) for test")

    make_parser.add_argument("-uc", "--userColname", type=str, default=USER.name, help=f"the column name (default: {USER.name}) of User ID")
    make_parser.add_argument("-ic", "--itemColname", type=str, default=ITEM.name, help=f"the column name (default: {ITEM.name}) of Item ID")
    make_parser.add_argument("-rc", "--ratingColname", type=str, default=RATING.name, help=f"the column name (default: {RATING.name}) of Rating")
    make_parser.add_argument("-tc", "--timestampColname", type=str, default=TIMESTAMP.name, help=f"the column name (default: {TIMESTAMP.name}) of Timestamp")

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()