



class Tag:
    __slot__ = ('name')

    def __init__(self, name) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name

SPARSE = Tag('Sparse')
DENSE = Tag('Dense')
ID = Tag('ID')
USER = Tag('User')
ITEM = Tag('Item')
FEATURE = Tag('Feature')
TARGET = Tag('Target')
