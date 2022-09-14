

from freerec.parser import CoreParser
from freerec.launcher import Adapter


def main():

    cfg = CoreParser()
    cfg.compile()

    tuner = Adapter()
    tuner.compile(cfg)

    tuner.grid_search()


if __name__ == "__main__":
    main()



