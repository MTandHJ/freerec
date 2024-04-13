

from enum import Enum


class FieldTags(Enum):
    STR = 'STR'
    INT = 'INT'
    FLOAT = 'FLOAT'
    TOKEN = 'TOKEN'

    ID = 'ID'
    USER = 'USER'
    ITEM = 'ITEM'
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


STR = FieldTags('STR')
INT = FieldTags('INT')
FLOAT = FieldTags('FLOAT')
TOKEN = FieldTags('TOKEN')

ID = FieldTags('ID')
USER = FieldTags('USER')
ITEM = FieldTags('ITEM')
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