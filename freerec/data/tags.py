from enum import Enum


class FieldTags(Enum):
    r"""Enumeration of tags used to describe and annotate fields.

    Members fall into three categories:

    **Descriptive tags** identify the semantic role of a field:
    ``ID``, ``USER``, ``ITEM``, ``LABEL``, ``RATING``, ``FEATURE``,
    ``TIMESTAMP``.

    **Functional tags** mark how a field is used during training or
    evaluation: ``SIZE``, ``SEEN``, ``UNSEEN``, ``SEQUENCE``,
    ``POSITIVE``, ``NEGATIVE``.

    **Embedding tag** indicates the field carries learnable embeddings:
    ``EMBED``.

    **Reserved tags** are placeholders for future use: ``X``, ``XX``,
    ``XXX``.
    """

    # description
    ID = "ID"
    USER = "USER"
    ITEM = "ITEM"
    LABEL = "LABEL"
    RATING = "RATING"
    FEATURE = "FEATURE"
    TIMESTAMP = "TIMESTAMP"

    # functional
    SIZE = "SIZE"
    SEEN = "SEEN"
    UNSEEN = "UNSEEN"
    SEQUENCE = "SEQUENCE"
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"

    # embedding
    EMBED = "EMBED"

    # reserved Tags
    X = "X"
    XX = "XX"
    XXX = "XXX"


class TaskTags(Enum):
    r"""Enumeration of tags used to identify recommendation task types.

    Members
    -------
    MATCHING
        User-item matching (e.g., collaborative filtering).
    NEXTITEM
        Next-item prediction (e.g., sequential recommendation).
    PREDICTION
        General prediction tasks (e.g., rating prediction).
    """

    MATCHING = "MATCHING"
    NEXTITEM = "NEXTITEM"
    PREDICTION = "PREDICTION"


ID = FieldTags("ID")
USER = FieldTags("USER")
ITEM = FieldTags("ITEM")
LABEL = FieldTags("LABEL")
RATING = FieldTags("RATING")
FEATURE = FieldTags("FEATURE")
TIMESTAMP = FieldTags("TIMESTAMP")

SIZE = FieldTags("SIZE")
SEEN = FieldTags("SEEN")
UNSEEN = FieldTags("UNSEEN")
SEQUENCE = FieldTags("SEQUENCE")
POSITIVE = FieldTags("POSITIVE")
NEGATIVE = FieldTags("NEGATIVE")

EMBED = FieldTags("EMBED")

X = FieldTags("X")
XX = FieldTags("XX")
XXX = FieldTags("XXX")


MATCHING = TaskTags("MATCHING")
NEXTITEM = TaskTags("NEXTITEM")
PREDICTION = TaskTags("PREDICTION")
