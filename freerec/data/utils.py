


from typing import TypeVar, Callable, Optional, Union, List, Tuple

import numpy as np
import os, requests, warnings, hashlib, tqdm

from ..utils import infoLogger


T = TypeVar('T')
class DataSetLoadingError(Exception):
    r"""Exception raised when a dataset fails to load."""
    ...

def safe_cast(val: T, dest_type: Callable[[T], T], default: T) -> T:
    r"""Cast a value to the specified type with a fallback default.

    If ``val`` is ``None`` or an empty string, ``default`` is used instead.
    Both ``val`` and ``default`` are cast through ``dest_type``.

    Parameters
    ----------
    val : T
        The value to cast.
    dest_type : callable
        A callable that performs the type conversion.
    default : T
        The fallback value when ``val`` is ``None`` or ``''``.

    Returns
    -------
    T
        The value cast to the specified type.

    Raises
    ------
    ValueError
        If neither ``val`` nor ``default`` can be cast by ``dest_type``.

    Notes
    -----
    This function casts ``val`` to the specified type using ``dest_type``.
    If ``val`` is ``None`` or an empty string, the function uses ``default``.
    If ``default`` is also ``None``, a ``ValueError`` is raised.
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
    r"""Download a file from a URL to a local directory.

    Codes borrowed from dgl.data.utils.

    Parameters
    ----------
    url : str
        The URL to download the file from.
    root : str, optional
        The directory where the file will be saved. Default is ``'.'``.
    filename : str, optional
        The name for the downloaded file. If ``None``, the filename is
        inferred from the URL.
    overwrite : bool, optional
        Whether to overwrite an existing file. Default is ``False``.
    retries : int, optional
        Number of retry attempts on failure. Default is ``5``.
    chunk_size : int, optional
        Download chunk size in bytes. Default is ``1048576`` (1 MB).
    sha1_hash : str, optional
        Expected SHA-1 hash in hexadecimal. If specified and the existing
        file does not match, the file is re-downloaded.
    verify_ssl : bool, optional
        Whether to verify SSL certificates. Default is ``True``.
    log : bool, optional
        Whether to log download progress. Default is ``True``.

    Returns
    -------
    str
        The file path of the downloaded file.

    Raises
    ------
    RuntimeError
        If the server returns a non-200 status code.
    :class:`DataSetLoadingError`
        If the downloaded file's SHA-1 hash does not match ``sha1_hash``.
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

    if overwrite or not os.path.exists(file_) or (sha1_hash and sha1_hash != check_sha1(file_)):
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
                return file_
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    raise e
                else:
                    if log:
                        infoLogger("[DataSet] >>> Download failed, retrying, {} attempt{} left"
                              .format(retries, 's' if retries > 1 else ''))
        infoLogger("Failed downloading url %s" % url)
    return file_


def extract_archive(file_, target_dir, overwrite=False):
    r"""Extract an archive file to a target directory.

    Supports ``.tar.gz``, ``.tar``, ``.tgz``, ``.gz``, and ``.zip``
    formats. Codes borrowed from dgl/data/utils.py.

    Parameters
    ----------
    file_ : str
        The path to the archive file.
    target_dir : str
        The directory to extract the archive contents into.
    overwrite : bool, optional
        Whether to overwrite existing files. Default is ``False``.

    Raises
    ------
    :class:`DataSetLoadingError`
        If the archive format is not recognized.
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

def check_sha1(data: Union[str, bytes]) -> str:
    r"""Compute the SHA-1 hash of a file or raw bytes.

    Parameters
    ----------
    data : str or bytes
        If str, treated as a file path whose contents are hashed.
        If bytes, the raw data is hashed directly.

    Returns
    -------
    str
        The hexadecimal SHA-1 digest.
    """
    sha1 = hashlib.sha1()
    if isinstance(data, str):
        with open(data, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
    else:
        sha1.update(data)
    return sha1.hexdigest()

def is_empty_dir(path: str) -> bool:
    r"""Check whether a directory is empty or does not exist.

    Parameters
    ----------
    path : str
        The directory path to check.

    Returns
    -------
    bool
        ``True`` if the directory does not exist or contains no entries.
    """
    return not os.path.exists(path) or not any(True for _ in os.scandir(path))

def negsamp_vectorized_bsearch(
    positives: Union[Tuple, List], n_items: int,
    size: Union[int, List[int], Tuple[int]] = 1,
    replacement: bool = True
) -> List:
    r"""Sample negative indices uniformly, excluding given positives.

    Uses a binary-search-based algorithm for efficient negative sampling.
    See `here <https://tech.hbc.com/2018-03-23-negative-sampling-in-numpy.html>`_
    for more details.

    Parameters
    ----------
    positives : list or tuple
        Sorted 1-D sequence of positive indices to exclude.
    n_items : int
        Total number of items (index range is ``[0, n_items)``).
    size : int or list of int or tuple of int, optional
        Shape of the output sample. Default is ``1``.
    replacement : bool, optional
        If ``True``, sample with replacement. Default is ``True``.

    Returns
    -------
    list
        The sampled negative indices.

    Raises
    ------
    AssertionError
        If ``positives`` is not 1-D.
    ValueError
        If ``replacement`` is ``False`` and the requested sample size
        exceeds the number of available negatives.
    """
    positives = np.asarray(positives)
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
