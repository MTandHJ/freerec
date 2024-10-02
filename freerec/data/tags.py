

from enum import Enum


class FieldTags(Enum):
    ID = 'ID'
    USER = 'USER'
    ITEM = 'ITEM'
    LABEL = 'LABEL'
    RATING = 'RATING'
    FEATURE = 'FEATURE'
    TIMESTAMP = 'TIMESTAMP'

    SEEN = 'SEEN'
    UNSEEN = 'UNSEEN'
    SEQUENCE = 'SEQUENCE'
    POSITIVE = 'POSITIVE'
    NEGATIVE = 'NEGATIVE'

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

SEEN = FieldTags('SEEN')
UNSEEN = FieldTags('UNSEEN')
SEQUENCE = FieldTags('SEQUENCE')
POSITIVE = FieldTags('POSITIVE')
NEGATIVE = FieldTags('NEGATIVE')

MATCHING = TaskTags('MATCHING')
NEXTITEM = TaskTags('NEXTITEM')
PREDICTION = TaskTags('PREDICTION')