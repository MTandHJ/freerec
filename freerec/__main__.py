

import argparse


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
    from .data.preprocessing import datasets
    from .data.tags import USER, ITEM, TIMESTAMP

    converter = getattr(datasets, args.dataset)(
        root=args.root,
        filename=args.filename
    )

    star4pos = args.star4pos
    kcore4user = args.kcore4user
    kcore4item = args.kcore4item
    ratios = tuple(map(int, args.ratios.split(',')))
    
    if args.datatype == 'gen' and args.by == 'ratio':
        fields = None if args.all else (USER.name, ITEM.name)
        converter.make_general_dataset(
            star4pos=star4pos,
            kcore4user=kcore4user,
            kcore4item=kcore4item,
            ratios=ratios,
            fields=fields
        )
    elif args.datatype == 'seq' and args.by == 'last-two':
        fields = None if args.all else (USER.name, ITEM.name, TIMESTAMP.name)
        converter.make_sequential_dataset(
            star4pos=star4pos,
            kcore4user=kcore4user,
            kcore4item=kcore4item,
            fields=fields
        )
    elif args.datatype == 'seq' and args.by == 'ratio':
        fields = None if args.all else (USER.name, ITEM.name, TIMESTAMP.name)
        converter.make_sequential_dataset_by_ratio(
            star4pos=star4pos,
            kcore4user=kcore4user,
            kcore4item=kcore4item,
            fields=fields
        )
    else:
        raise ValueError(f"`{args.datatype}' type dataset cannot be made by `{args.by}'")

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

    tune_parser.add_argument("--eval-freq", type=int, default=None, help="the evaluation frequency")
    tune_parser.add_argument("--num-workers", type=int, default=None)

    tune_parser.add_argument("--resume", action="store_true", default=False, help="resume the search from the recent checkpoint")


    make_parser = subparsers.add_parser("make")
    make_parser.set_defaults(func=make)

    make_parser.add_argument("dataset", type=str, help="dataset name")
    make_parser.add_argument("--root", type=str, default=".", help="data")
    make_parser.add_argument("--filename", type=str, default=None, help="filename of Atomic files")

    make_parser.add_argument("--datatype", type=str, choices=('gen', 'seq'), default='gen')
    make_parser.add_argument("--by", type=str, choices=('ratio', 'last-two'), default='ratio')

    make_parser.add_argument("--star4pos", type=int, default=0)
    make_parser.add_argument("--kcore4user", type=int, default=10)
    make_parser.add_argument("--kcore4item", type=int, default=10)
    make_parser.add_argument("--ratios", type=str, default="8,1,1")

    make_parser.add_argument("--all", action="store_true", default=False)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()