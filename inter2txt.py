

import argparse

from freerec.data.preprocessing import GenInter2Txt, SeqInter2Txt

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
parser.add_argument("dataset", type=str)
parser.add_argument("--root", type=str, default="../data")
parser.add_argument("--datatype", type=str, choices=('gen', 'seq'), default="gen")
parser.add_argument("--kcore4user", type=int, default=10, help="select the user interacted >=k items")
parser.add_argument("--kcore4item", type=int, default=10, help="select the item interacted >=k users")
parser.add_argument("--star4pos", type=int, default=4, help="select pairs with star >= k")
parser.add_argument("--ratios", type=str, default='8,1,1', help="train:valid:text for gen")
args = parser.parse_args()

args.ratios = list(map(int, args.ratios.split(',')))

if args.datatype == 'gen':
    print(">>> Make General dataset")
    args.datatype = GenInter2Txt
elif args.datatype == 'seq':
    print(">>> Make Sequential dataset")
    args.datatype = SeqInter2Txt


processor = args.datatype(
    root=args.root, filename=args.filename, dataset=args.dataset,
    kcore4user=args.kcore4user, kcore4item=args.kcore4item,
    star4pos=args.star4pos, ratios=args.ratios
)

processor.compile()
