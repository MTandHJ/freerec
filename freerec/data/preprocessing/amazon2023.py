


from typing import List, Dict, Optional

import gzip, json, os
import pandas as pd

from ..tags import USER, ITEM, TIMESTAMP, RATING


__all__ = ["extract_from_amazon2023"]


INTER_FIELDS = {
    'user_id': USER.name,
    'asin': ITEM.name,
    'rating': RATING.name,
    'timestamp': TIMESTAMP.name,
    'parent_asin': 'parent_asin'
}

ITEM_FIELDS = {
    'parent_asin': 'parent_asin',
    'title': 'title',
    'categories': 'categories',
    # 'features': 'features',
    # 'description': 'description',
    'images': 'image_urls'
}

def open_and_read_json_gz(root, file) -> List:
    r"""Read a gzipped JSON-lines file and return all records as a list.

    Parameters
    ----------
    root : str
        Directory containing the file.
    file : str
        Filename of the ``.json.gz`` file.

    Returns
    -------
    list
        List of parsed JSON objects (one per line).
    """
    data = []
    for line in gzip.open(os.path.join(root, file)):
        data.append(json.loads(line.strip()))
    return data

def extract_from_amazon2023(
    root: str,
    review_file: Optional[str] = None,
    meta_file: Optional[str] = None,
    inter_fields: Dict = INTER_FIELDS,
    item_fields: Dict = ITEM_FIELDS
):
    r"""Extract interaction and item dataframes from Amazon Reviews 2023.

    Parameters
    ----------
    root : str
        Root directory containing the gzipped review and meta files.
    review_file : str or None, optional
        Filename of the review gzip file. If ``None``, the file is
        detected automatically.
    meta_file : str or None, optional
        Filename of the meta gzip file. If ``None``, the file is
        detected automatically.
    inter_fields : dict, optional
        Mapping from raw field names to canonical names for the
        interaction dataframe. Defaults to ``INTER_FIELDS``.
    item_fields : dict, optional
        Mapping from raw field names to canonical names for the item
        dataframe. Defaults to ``ITEM_FIELDS``.

    Returns
    -------
    inter_df : :class:`pandas.DataFrame`
        Interaction dataframe with columns defined by *inter_fields*.
    item_df : :class:`pandas.DataFrame`
        Item metadata dataframe with columns defined by *item_fields*
        plus a ``brand`` column.

    Examples
    --------
    >>> from freerec.data.preprocessing.amazon2023 import extract_from_amazon2023
    >>> inter_df, item_df = extract_from_amazon2023(
    ...     "../RecSets/Amazon2023/Baby",
    ... )
    """

    # find review/meta data
    if meta_file is None:
        meta_file = next(filter(lambda file: file.endswith('gz') and file.startswith('meta'), os.listdir(root)))
    if review_file is None:
        review_file = next(filter(lambda file: file.endswith('gz') and not file.startswith('meta'), os.listdir(root)))

    # fields
    inter_fields.update(INTER_FIELDS)
    item_fields.update(ITEM_FIELDS)

    # interaction data
    ego_cols, cur_cols = list(inter_fields.keys()), list(inter_fields.values())
    inter_df = pd.DataFrame(
        [[row[key] for key in ego_cols] for row in open_and_read_json_gz(root, review_file)],
        columns=cur_cols
    )

    # item meta data
    ego_cols, cur_cols = list(item_fields.keys()), list(item_fields.values())
    raw_meta = {
        row['parent_asin']: [row[key] for key in ego_cols] + [row['details'].get('Brand', '')]
        for row in open_and_read_json_gz(root, meta_file)
    }
    uniques = inter_df.groupby(ITEM.name).head(1)
    items, parents = uniques[ITEM.name], uniques['parent_asin']
    raw_item = [
        [item_id] + raw_meta[parent_id] for (item_id, parent_id) in zip(items, parents)
    ]
    item_df = pd.DataFrame(
        raw_item,
        columns=[ITEM.name] + cur_cols + ['brand']
    )

    return inter_df, item_df
