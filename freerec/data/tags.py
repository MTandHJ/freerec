

from enum import Enum


class FieldTags(Enum):
    Sparse = 'Sparse'
    Dense = 'Dense'
    User = 'User'
    Item = 'Item'
    Interaction = 'Interaction'
    Timestamp = 'Timestamp'
    Rating = 'Rating'
    ID = 'ID'
    Feature = 'Feature'
    Target = 'Target'
    Positive = 'Positive'
    Negative = 'Negative'
    Seen = 'Seen'
    Unseen = 'Unseen'


SPARSE = FieldTags('Sparse')
DENSE = FieldTags('Dense')
USER = FieldTags('User')
ITEM = FieldTags('Item')
INTERACTION = FieldTags('Interaction')
TIMESTAMP = FieldTags('Timestamp')
RATING = FieldTags('Rating')
ID = FieldTags('ID')
FEATURE = FieldTags('Feature')
TARGET = FieldTags('Target')
POSITIVE = FieldTags('Positive')
NEGATIVE = FieldTags('Negative')
SEEN = FieldTags('Seen')
UNSEEN = FieldTags('Unseen')