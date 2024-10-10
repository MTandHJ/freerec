

from enum import Enum


class FieldTags(Enum):
    # description
    ID = 'ID'
    USER = 'USER'
    ITEM = 'ITEM'
    LABEL = 'LABEL'
    RATING = 'RATING'
    FEATURE = 'FEATURE'
    TIMESTAMP = 'TIMESTAMP'

    # functional
    SIZE = 'SIZE'
    SEEN = 'SEEN'
    UNSEEN = 'UNSEEN'
    SEQUENCE = 'SEQUENCE'
    POSITIVE = 'POSITIVE'
    NEGATIVE = 'NEGATIVE'

    # embedding
    EMBED = 'EMBED'

class TaskTags(Enum):
    MATCHING = 'MATCHING'
    NEXTITEM = 'NEXTITEM'
    PREDICTION = 'PREDICTION'

ID = FieldTags('ID')
USER = FieldTags('USER')
ITEM = FieldTags('ITEM')
LABEL = FieldTags('LABEL')
RATING = FieldTags('RATING')
FEATURE = FieldTags('FEATURE')
TIMESTAMP = FieldTags('TIMESTAMP')

SIZE = FieldTags('SIZE')
SEEN = FieldTags('SEEN')
UNSEEN = FieldTags('UNSEEN')
SEQUENCE = FieldTags('SEQUENCE')
POSITIVE = FieldTags('POSITIVE')
NEGATIVE = FieldTags('NEGATIVE')

EMBED = FieldTags('EMBED')

MATCHING = TaskTags('MATCHING')
NEXTITEM = TaskTags('NEXTITEM')
PREDICTION = TaskTags('PREDICTION')