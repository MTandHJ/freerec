

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
    from .data.preprocessing.base import AtomicConverter
    from .data.preprocessing import datasets
    from .data.tags import USER, SESSION, ITEM, TIMESTAMP

    converter: AtomicConverter = getattr(datasets, args.dataset)(
        root=args.root,
        filename=args.filename
    )

    star4pos = args.star4pos
    kcore4user = args.kcore4user
    kcore4item = args.kcore4item
    ratios = tuple(map(int, args.ratios.split(',')))
    days = args.days
    strict = not args.not_strict
    
    if args.datatype == 'gen' and args.by == 'ratio':
        fields = None if args.all else (USER.name, ITEM.name)
        converter.make_general_dataset(
            star4pos=star4pos,
            kcore4user=kcore4user,
            kcore4item=kcore4item,
            strict=strict,
            ratios=ratios,
            fields=fields
        )
    elif args.datatype == 'seq' and args.by == 'leave-one-out':
        fields = None if args.all else (USER.name, ITEM.name, TIMESTAMP.name)
        converter.make_sequential_dataset(
            star4pos=star4pos,
            kcore4user=kcore4user,
            kcore4item=kcore4item,
            strict=strict,
            fields=fields
        )
    elif args.datatype == 'seq' and args.by == 'ratio':
        fields = None if args.all else (USER.name, ITEM.name, TIMESTAMP.name)
        converter.make_sequential_dataset_by_ratio(
            star4pos=star4pos,
            kcore4user=kcore4user,
            kcore4item=kcore4item,
            strict=strict,
            ratios=ratios,
            fields=fields
        )
    elif args.datatype == 'sess' and args.by == 'day':
        fields = None if args.all else (SESSION.name, ITEM.name, TIMESTAMP.name)
        converter.make_session_dataset_by_day(
            star4pos=star4pos,
            kcore4user=kcore4user,
            kcore4item=kcore4item,
            strict=strict,
            days=days,
            fields=fields
        )
    elif args.datatype == 'sess' and args.by == 'ratio':
        fields = None if args.all else (SESSION.name, ITEM.name, TIMESTAMP.name)
        converter.make_session_dataset_by_ratio(
            star4pos=star4pos,
            kcore4user=kcore4user,
            kcore4item=kcore4item,
            strict=strict,
            ratios=ratios,
            fields=fields
        )
    elif args.datatype =='ctxt' and args.by == 'ratio':
        fields = None if args.all else (USER.name, ITEM.name, TIMESTAMP.name)
        converter.make_context_dataset_by_ratio(
            star4pos=star4pos,
            kcore4user=kcore4user,
            kcore4item=kcore4item,
            strict=strict,
            ratios=ratios,
            fields=fields
        )
    elif args.datatype =='ctxt' and args.by == 'leave-one-out':
        fields = None if args.all else (USER.name, ITEM.name, TIMESTAMP.name)
        converter.make_context_dataset_by_last_two(
            star4pos=star4pos,
            kcore4user=kcore4user,
            kcore4item=kcore4item,
            strict=strict,
            fields=fields
        )
    elif args.datatype =='ctxt' and args.by == 'day':
        fields = None if args.all else (SESSION.name, ITEM.name, TIMESTAMP.name)
        converter.make_context_dataset_by_day(
            star4pos=star4pos,
            kcore4user=kcore4user,
            kcore4item=kcore4item,
            strict=strict,
            days=days,
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
    tune_parser.add_argument("--num-workers", type=int, default=None)

    tune_parser.add_argument("--resume", action="store_true", default=False, help="resume the search from the recent checkpoint")


    make_parser = subparsers.add_parser("make")
    make_parser.set_defaults(func=make)

    make_parser.add_argument("dataset", type=str, help="dataset name")
    make_parser.add_argument("--root", type=str, default=".", help="data")
    make_parser.add_argument("--filename", type=str, default=None, help="filename of Atomic files")

    make_parser.add_argument(
        "--datatype", type=str, choices=('gen', 'seq', 'sess', 'ctxt'), default='gen', 
        help="gen: general; seq: sequential; sess: session; ctxt:context"
    )
    make_parser.add_argument(
        "--by", type=str, choices=('ratio', 'leave-one-out', 'day'), default='ratio', 
        help="gen: ratio; seq: leave-one-out; ratio; sess: ratio, day"
    )

    make_parser.add_argument("--star4pos", type=int, default=0, help="select interactions with `Rating >= star4pos'")
    make_parser.add_argument("--kcore4user", type=int, default=10, help="select kcore interactions according to User")
    make_parser.add_argument("--kcore4item", type=int, default=10, help="select kcore interactions according to Item")
    make_parser.add_argument("--not-strict", action="store_true", default=False, help="filter by kcore once if True")
    make_parser.add_argument("--ratios", type=str, default="8,1,1", help="the ratios of training|validation|test set")
    make_parser.add_argument("--days", type=int, default=7, help="the second last days for validation and last days for test")

    make_parser.add_argument("--all", action="store_true", default=False, help="reserve all fields if True")

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()