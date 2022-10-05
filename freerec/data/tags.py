

from enum import Enum


class FieldTags(Enum):
    Sparse = 'Sparse'
    Dense = 'Dense'
    ID = 'ID'
    User = 'User'
    Item = 'Item'
    Feature = 'Feature'
    Target = 'Target'


SPARSE = FieldTags('Sparse')
DENSE = FieldTags('Dense')
ID = FieldTags('ID')
USER = FieldTags('User')
ITEM = FieldTags('Item')
FEATURE = FieldTags('Feature')
TARGET = FieldTags('Target')