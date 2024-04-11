

from enum import Enum


class FieldTags(Enum):
    Sparse = 'Sparse'
    Dense = 'Dense'
    Affiliate = 'Affiliate'

    ID = 'ID'
    User = 'User'
    Item = 'Item'
    Rating = 'Rating'
    Feature = 'Feature'
    Timestamp = 'Timestamp'

    Matching = 'Matching'
    NextItem = 'NextItem'

    Seen = 'Seen'
    Unseen = 'Unseen'
    Positive = 'Positive'
    Negative = 'Negative'




SPARSE = FieldTags('Sparse')
DENSE = FieldTags('Dense')
AFFILIATE = FieldTags('Affiliate')

ID = FieldTags('ID')
USER = FieldTags('User')
ITEM = FieldTags('Item')
RATING = FieldTags('Rating')
FEATURE = FieldTags('Feature')
TIMESTAMP = FieldTags('Timestamp')

MATCHING = FieldTags('Matching')
NEXTITEM = FieldTags('NEXTITEM')

SEEN = FieldTags('Seen')
UNSEEN = FieldTags('Unseen')
POSITIVE = FieldTags('Positive')
NEGATIVE = FieldTags('Negative')