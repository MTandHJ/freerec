

from typing import TypeVar, Callable, Optional, Union, List, Tuple

import numpy as np
import os, requests, warnings, hashlib, tqdm

from ..utils import infoLogger


T = TypeVar('T')
class DataSetLoadingError(Exception): ...

def safe_cast(val: T, dest_type: Callable[[T], T], default: T) -> T:
    r"""
    Cast the value to the specified type.

    Parameters:
    -----------
    val : T
        The value to be casted to the specified type.
    dest_type : Callable[[T], T]
        A function to cast `val` to the specified type.
    default : T
        The default value to use if `val` is None or an empty string.

    Returns:
    --------
    value: T
        The value of `val` casted to the specified type.

    Raises:
    -------
    ValueError
        If `val` or `default` cannot be casted to the specified type `dest_type`.

    Notes:
    ------
    This function casts `val` to the specified type using the `dest_type` function.
    If `val` is None or an empty string, the function will use the `default` value.
    If the `default` value is None, a ValueError will be raised.
    """
    try:
        if val not in (None, ''):
            return dest_type(val)
        else: # fill_na
            if default is None:
                raise ValueError
            return dest_type(default)
    except (ValueError, TypeError):
        raise ValueError(
            f"Using '{dest_type.__name__}' to convert '{val}' where the default value is '{default}' ..." \
            f"This happens when the value (or the default value: '{default}') to be cast is not of the type '{dest_type.__name__}'.",
        )


def download_from_url(
    url: str, root: str = '.', filename: Optional[str] = None, 
    overwrite: bool = False, retries=5, chunk_size: int = 1024 * 1024,
    sha1_hash: Optional[str] = None, verify_ssl: bool = True, 
    log: bool = True
):
    r"""
    Download a file from a given URL.

    Codes borrowed from dgl.data.utils

    Parameters:
    -----------
    url : str
        The URL to download the file from.
    root : str, optional
        The root directory where the downloaded file will be saved. Default is current directory ('.').
    filename: str, optional
        The filename of the downloaded file. If None, the filename will be inferred from the URL.
    overwrite : bool, optional
        Whether to overwrite the destination file if it already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    retries : int, optional
        The number of times to attempt downloading in case of failure or non 200 return codes. Default is 5.
    chunk_size: int, default to 1024 * 1024 (=1MB)
    verify_ssl : bool, optional
        Verify SSL certificates. Default is True.
    log : bool, optional
        Whether to print the progress of download. Default is True.

    Returns:
    --------
    str
        The file path of the downloaded file.
    """
    if filename is None:
        filename = url.split('/')[-1]
        # Empty filenames are invalid
        assert filename, 'Can\'t construct file-name from this URL. ' \
            'Please set the `filename` option manually.'
    file_ = os.path.join(root, filename)

    assert retries >= 0, "Number of retries should be at least 0"

    if not verify_ssl:
        warnings.warn(
            'Unverified HTTPS request is being made (verify_ssl=False). '
            'Adding certificate verification is strongly advised.')

    if overwrite or not os.path.exists(file_) or (sha1_hash and not check_sha1(file_, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(file_)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        while retries+1 > 0:
            # Disable pyling too broad Exception
            # pylint: disable=W0703
            try:
                if log:
                    infoLogger('[DataSet] >>> Downloading %s from %s...' % (file_, url))
                r = requests.get(url, stream=True, verify=verify_ssl)
                filesize = int(r.headers.get("Content-Length", 0))
                if r.status_code != 200:
                    raise RuntimeError("Failed downloading url %s" % url)
                with open(file_, 'wb') as f:
                    if log:
                        progress_bar = tqdm.tqdm(total=filesize, unit="B", unit_scale=True, desc="վ'ᴗ' ի-")
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if chunk:  # filter out keep-alive new chunks
                                progress_bar.update(len(chunk))
                                f.write(chunk)
                        progress_bar.close()
                    else:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if chunk:  # filter out keep-alive new chunks
                                f.write(chunk)
                if sha1_hash and not check_sha1(file_, sha1_hash):
                    raise DataSetLoadingError(
                        'File {} is downloaded but the content hash does not match.'
                        ' The repo may be outdated or download may be incomplete. '
                        'If the "repo_url" is overridden, consider switching to '
                        'the default repo.'.format(file_)
                    )
                break
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    raise e
                else:
                    if log:
                        infoLogger("[DataSet] >>> Download failed, retrying, {} attempt{} left"
                              .format(retries, 's' if retries > 1 else ''))

    return file_


def extract_archive(file_, target_dir, overwrite=False):
    r"""
    Extract files from an archive.

    Codes borrowed from dgl/data/utils.py

    Parameters:
    -----------
    file_ : str
        The path to the archive file to extract.
    target_dir : str
        The directory to extract the archive contents to.
    overwrite : bool, optional
        Whether to overwrite existing files in the target directory. Defaults to False.

    Raises:
    -------
    DataSetLoadingError
        If an unsupported archive file type is encountered.
    """

    if os.path.exists(target_dir) and not overwrite:
        return
    infoLogger('[DataSet] >>> Extracting file to {}'.format(target_dir))
    if file_.endswith('.tar.gz') or file_.endswith('.tar') or file_.endswith('.tgz'):
        import tarfile
        with tarfile.open(file_, 'r') as archive:
            archive.extractall(path=target_dir)
    elif file_.endswith('.gz'):
        import gzip
        import shutil
        with gzip.open(file_, 'rb') as f_in:
            target_file = os.path.join(target_dir, os.path.basename(file_)[:-3])
            with open(target_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    elif file_.endswith('.zip'):
        import zipfile
        with zipfile.ZipFile(file_, 'r') as archive:
            archive.extractall(path=target_dir)
    else:
        raise DataSetLoadingError('Unrecognized file type: ' + file_)


def check_sha1(filename, sha1_hash):
    r"""
    Check if the SHA1 hash of a file matches the expected hash.

    Parameters:
    ----------
    filename : str
        The path to the file to check the hash of.
    sha1_hash : str
        The expected SHA1 hash value.

    Returns:
    --------
    bool
        True if the file's hash matches the expected value, False otherwise.
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash


def negsamp_vectorized_bsearch(
    positives: Union[Tuple, List], n_items: int, 
    size: Union[int, List[int], Tuple[int]] = 1,
    replacement: bool = True
) -> List:
    r"""
    Uniformly sampling negatives according to a list of ordered positives
    See [here](https://tech.hbc.com/2018-03-23-negative-sampling-in-numpy.html) for more details.

    Parameters:
    -----------
    positives: Union[Tuple, List], 1-D
        The positive indices should be in ordered.
    n_items: int
        The number of all items.
    size: int or List[int] or Tuple[int]
        The size of required negatives.
    replacement: bool, default to `True`
        - True: sampling with replacement
        - False: sampling without replacement
    
    Raises:
    -------
    AssertionError:
        Given `positives` is not 1-D.
    ValueError:
        Too much negatives required.
    """
    positives = np.array(positives, copy=False)
    assert positives.ndim == 1, f"positives should be 1-D array but {positives.ndim}-D received ..."
    try:
        if replacement:
            raw_samp = np.random.randint(0, n_items - len(positives), size=size)
        else:
            raw_samp = np.random.choice(n_items - len(positives), size=size, replace=replacement)
    except ValueError:
        raise ValueError(
            "The number of required negatives is larger than that of candidates, but replacement is False ..."
        )
    pos_inds_adj = positives - np.arange(len(positives))
    ss = np.searchsorted(pos_inds_adj, raw_samp, side='right')
    neg_inds = raw_samp + ss
    return neg_inds.tolist()