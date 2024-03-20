

from typing import Callable, Iterable

import torch, os
import torchdata.datapipes as dp
from .base import TripletWithMeta
from ...tags import ITEM, ID
from ...utils import download_from_url


__all__ = [
    'Amazon2023',
    'Amazon2023_All_Beauty_550_Chron',
    'Amazon2023_Beauty_550_Chron', 'Amazon2023_Beauty_10100_Chron',
    'Amazon2023_Toys_550_Chron', 'Amazon2023_Toys_10100_Chron',
    'Amazon2023_Office_550_Chron',
    'Amazon2023_CDs_550_Chron', 'Amazon2023_CDs_10100_Chron',
]


class Amazon2023(TripletWithMeta):
    r"""
    Amazon datasets (2023).
    See [[here](https://amazon-reviews-2023.github.io/)] for more details.
    """

    text_converter = {
        'Title': lambda x: x,
        'Features': lambda x: ' '.join(eval(x)),
        'Description': lambda x: ' '.join(eval(x)),
        'Details': lambda x: ' '.join([f"{key}: {val}" for key, val in eval(x).items()])
    }

    def download_images(
        self, 
        image_folder: str = "item_images",
        image_size: str = 'thumb'
    ) -> None:
        r"""
        Download images from urls.

        Parameters:
        -----------
        image_folder: str,
            The image folder to save images.
        image_size: str, 'thumb', 'large' or 'hi_res'
        """
        import tqdm
        from concurrent.futures import ThreadPoolExecutor
        assert image_size in ('thumb', 'large', 'hi_res'), f"`size` should be 'thumb', 'large', 'hi_res' ..."
        urls = self.fields['Images'].data
        with ThreadPoolExecutor() as executor:
            for id_, url in tqdm.tqdm(enumerate(urls), desc="Download images: "):
                try:
                    url = eval(url)[0][image_size]
                    executor.submit(
                        download_from_url,
                        url=url,
                        root=os.path.join(self.path, image_folder, image_size),
                        filename=f"{id_}.jpg",
                        log=False
                    )
                except (KeyError, IndexError):
                    continue
        return

    def encode_visual_modality(
        self,
        model: str, model_dir: str, 
        image_folder: str, image_size: str = 'thumb',
        saved_file: str = "visual_modality.pkl",
        pool: bool = True, num_workers: int = 4, batch_size: int = 128,
    ) -> torch.Tensor:
        r"""
        Visual modality encoding via `transformers`.

        Parameters:
        -----------
        model: str
            The model name. Refer to [[here](https://huggingface.co/models?pipeline_tag=image-feature-extraction&sort=trending)]
        model_dir: str
            The cache dir for model.
        image_folder: str
            The cache folder for saving images to be encoded. 
            Note that `self.root/image/folder` will be used as the final path.
        image_size: str, 'thumb', 'large' or 'hi_res'
        saved_file: str
            The filename for saving visual modality features.
        pool: bool, default to `True`
            Pooled hidden states as visual modality features if `True`.
        num_workers: int
        batch_size: int
        """
        import tqdm
        from transformers import AutoImageProcessor, AutoModel
        from PIL import Image
        from freeplot.utils import export_pickle

        assert image_size in ('thumb', 'large', 'hi_res'), f"`size` should be 'thumb', 'large', 'hi_res' ..."
        Item = self.fields[ITEM, ID]
        images = []
        processor = AutoImageProcessor.from_pretrained(
            model, cache_dir=model_dir
        )

        def _process(idx: int):
            try:
                image = Image.open(
                    os.path.join(
                        self.path, image_folder, image_size, f"{idx}.jpg"
                    )
                ).convert('RGB')
            except FileNotFoundError:
                image = Image.new('RGB', (224, 224))
            return idx, processor(images=image, return_tensors='pt')['pixel_values'][0]

        datapipe = dp.iter.IterableWrapper(
            range(Item.count)
        ).sharding_filter().map(
            _process
        )
        dataloader = torch.utils.data.DataLoader(
            datapipe, 
            num_workers=num_workers, batch_size=batch_size,
            shuffle=False
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = AutoModel.from_pretrained(
            model, cache_dir=model_dir
        ).to(device)

        vIndices = []
        vFeats = []
        with torch.no_grad():
            encoder.eval()
            for (indices, images) in tqdm.tqdm(dataloader, desc="Visual batches: "):
                vIndices.append(indices)
                if pool:
                    vFeats.append(
                        encoder(pixel_values=images.to(device)).pooler_output.cpu()
                    )
                else:
                    vFeats.append(
                        encoder(pixel_values=images.to(device)).last_hidden_state.cpu()
                    )
        vIndices = torch.cat(vIndices, dim=0)
        vFeats = torch.cat(vFeats, dim=0).flatten(1) # (N, D)
        vFeats = vFeats[vIndices.argsort()] # reindex
        assert vFeats.size(0) == Item.count, f"Unknown errors happen ..."

        export_pickle(
            vFeats, os.path.join(
                self.path, saved_file
            )
        )
        return vFeats

    def encode_textual_modality(
        self,
        model: str, model_dir: str, 
        field_names: Iterable[str] = ('Title', 'Features', 'Description', 'Details'),
        saved_file: str = "textual_modality.pkl",
        batch_size: int = 128,
    ) -> torch.Tensor:
        r"""
        Textual modality encoding via `sentence_transformers`.

        Parameters:
        -----------
        model: str
            The model name. Refer to [[here](https://www.sbert.net/docs/pretrained_models.html)]
        model_dir: str
            The cache dir for model.
        field_names: Iterable[str]
            The fields to be encoded.
        saved_file: str
            The filename for saving visual modality features.
        batch_size: int
        """
        import tqdm
        from sentence_transformers import SentenceTransformer
        from freeplot.utils import export_pickle

        Item = self.fields[ITEM, ID]

        tfields = [self.fields[name] for name in field_names]
        sentences = []
        for i in tqdm.tqdm(range(Item.count), desc="Make sentences: "):
            sentence = ' '.join(
                self.text_converter[field.name](
                    field.data[i]
                )
                for field in tfields
            )
            sentences.append(sentence)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = SentenceTransformer(
            model, 
            cache_folder=model_dir,
            device=device
        )

        tFeats = encoder.encode(
            sentences, 
            convert_to_tensor=True,
            batch_size=batch_size, show_progress_bar=True
        ).cpu()
        assert tFeats.size(0) == Item.count, f"Unknown errors happen ..."

        export_pickle(
            tFeats, os.path.join(
                self.path, saved_file
            )
        )
        return tFeats


class Amazon2023_All_Beauty_550_Chron(Amazon2023):
    r"""
    Chronologically-ordered Amazon All Beauty dataset with meta data.

    Config:
    -------
    filename: All_Beauty
    dataset: Amazon2023_All_Beauty
    by: leave-one-out
    kcore4user: 5
    kcore4item: 5
    star4pos: 0

    +--------+--------+-------------------+---------------+--------+--------+-------+----------------------+
    | #Users | #Items |      Avg.Len      | #Interactions | #Train | #Valid | #Test |       Density        |
    +--------+--------+-------------------+---------------+--------+--------+-------+----------------------+
    |  346   |  466   | 9.184971098265896 |      3178     |  2486  |  346   |  346  | 0.019710238408295912 |
    +--------+--------+-------------------+---------------+--------+--------+-------+----------------------+
    """

class Amazon2023_Beauty_550_Chron(Amazon2023):
    r"""
    Chronologically-ordered Amazon Beauty dataset with meta data.

    Config:
    -------
    filename: Beauty_and_Personal_Care
    dataset: Amazon2023_Beauty
    by: leave-one-out
    kcore4user: 5
    kcore4item: 5
    star4pos: 0

    +--------+--------+---------------+---------+--------+--------+-----------------------+
    | #User  | #Item  | #Interactions |  #Train | #Valid | #Test  |        Density        |
    +--------+--------+---------------+---------+--------+--------+-----------------------+
    | 697966 | 253928 |    6340876    | 4944944 | 697966 | 697966 | 3.577703953833393e-05 |
    +--------+--------+---------------+---------+--------+--------+-----------------------+
    """


class Amazon2023_Beauty_10100_Chron(Amazon2023):
    r"""
    Chronologically-ordered Amazon Beauty dataset with meta data.

    Config:
    -------
    filename: Beauty_and_Personal_Care
    dataset: Amazon2023_Beauty
    by: leave-one-out
    kcore4user: 10
    kcore4item: 10
    star4pos: 0

    +-------+-------+---------------+---------+--------+-------+-----------------------+
    | #User | #Item | #Interactions |  #Train | #Valid | #Test |        Density        |
    +-------+-------+---------------+---------+--------+-------+-----------------------+
    | 73234 | 59208 |    1396703    | 1250235 | 73234  | 73234 | 0.0003221149776682619 |
    +-------+-------+---------------+---------+--------+-------+-----------------------+
    """


class Amazon2023_Toys_550_Chron(Amazon2023):
    r"""
    Chronologically-ordered Amazon Toys dataset with meta data.

    Config:
    -------
    filename: Toys_and_Games
    dataset: Amazon2023_Toys
    by: leave-one-out
    kcore4user: 5
    kcore4item: 5
    star4pos: 0

    +--------+--------+---------------+---------+--------+--------+-----------------------+
    | #User  | #Item  | #Interactions |  #Train | #Valid | #Test  |        Density        |
    +--------+--------+---------------+---------+--------+--------+-----------------------+
    | 427176 | 175207 |    3830356    | 2976004 | 427176 | 427176 | 5.117770914043441e-05 |
    +--------+--------+---------------+---------+--------+--------+-----------------------+
    """


class Amazon2023_Toys_10100_Chron(Amazon2023):
    r"""
    Chronologically-ordered Amazon Toys dataset with meta data.

    Config:
    -------
    filename: Toys_and_Games
    dataset: Amazon2023_Toys
    by: leave-one-out
    kcore4user: 10
    kcore4item: 10
    star4pos: 0

    +-------+-------+---------------+--------+--------+-------+-----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |        Density        |
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    | 35690 | 28529 |     611908    | 540528 | 35690  | 35690 | 0.0006009703339130787 |
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    """


class Amazon2023_Office_550_Chron(Amazon2023):
    r"""
    Chronologically-ordered Amazon Office dataset with meta data.

    Config:
    -------
    filename: Office_Products
    dataset: Amazon2023_Office
    by: leave-one-out
    kcore4user: 5
    kcore4item: 5
    star4pos: 0

    +--------+-------+---------------+---------+--------+--------+----------------------+
    | #User  | #Item | #Interactions |  #Train | #Valid | #Test  |       Density        |
    +--------+-------+---------------+---------+--------+--------+----------------------+
    | 204681 | 85028 |    1648181    | 1238819 | 204681 | 204681 | 9.47033647237546e-05 |
    +--------+-------+---------------+---------+--------+--------+----------------------+
    """


class Amazon2023_CDs_550_Chron(Amazon2023):
    r"""
    Chronologically-ordered Amazon CDs dataset with meta data.

    Config:
    -------
    filename: CDs_and_Vinyl
    dataset: Amazon2023_CDs
    by: leave-one-out
    kcore4user: 5
    kcore4item: 5
    star4pos: 0

    +--------+-------+---------------+---------+--------+--------+-----------------------+
    | #User  | #Item | #Interactions |  #Train | #Valid | #Test  |        Density        |
    +--------+-------+---------------+---------+--------+--------+-----------------------+
    | 126767 | 91509 |    1594801    | 1341267 | 126767 | 126767 | 0.0001374790356746118 |
    +--------+-------+---------------+---------+--------+--------+-----------------------+
    """


class Amazon2023_CDs_10100_Chron(Amazon2023):
    r"""
    Chronologically-ordered Amazon CDs dataset with meta data.

    Config:
    -------
    filename: CDs_and_Vinyl
    dataset: Amazon2023_CDs
    by: leave-one-out
    kcore4user: 10
    kcore4item: 10
    star4pos: 0

    +-------+-------+---------------+--------+--------+-------+-----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |        Density        |
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    | 23059 | 20762 |     540118    | 494000 | 23059  | 23059 | 0.0011281815544690774 |
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    """