

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
    r"""
    Extract interaction and item dataframe from amazon2023.

    Parameters:
    -----------
    root: root path
    review_file: optional[str]
        review gzip file
        - `None`: Find it automatically.
    meta_file: optional[str]
        meta gzip file
        - `None`: Find it automatically.
    inter_fields: Dict
        fields to be retained and renamed for interaction.
        - default: 'user_id', 'asin', 'rating', 'timestamp', 'parent_asin'
    item_fields:
        - default: 'parent_asin', 'title', 'features', 'description'
    
    Examples:
    ---------
    >>> from freerec.data.preprocessing.amazon2023 import extract_from_amazon2023
    >>> inter_df, item_df = extract_from_amazon2023(
        "../RecSets/Amazon2023/Baby",
    )
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