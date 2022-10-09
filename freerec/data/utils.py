

from typing import TypeVar, Callable, Dict, List, Optional

import numpy as np
import os, requests, warnings, hashlib, tqdm

from ..utils import errorLogger, infoLogger

T = TypeVar('T')

def safe_cast(val: T, dest_type: Callable[[T], T], default: T) -> T:
    try:
        if val:
            return dest_type(val)
        else: # fill_na
            return default
    except ValueError:
        errorLogger(
            f"Using '{dest_type.__name__}' to convert '{val}' where the default value is {default}", 
            ValueError
        )


def collate_dict(batch: List[Dict]):
    elem = batch[0]
    return {key: np.array([d[key] for d in batch]) for key in elem}


def download_from_url(
    url: str, root: str = '.', filename: Optional[str] = None, 
    overwrite: bool = False, retries=5, 
    sha1_hash: Optional[str] = None, verify_ssl: bool = True, 
    log: bool = True
):
    """Download a given URL.

    Codes borrowed from dgl.data.utils

    Parameters
    ---

    url : str
        URL to download.
    root : str, default '.'
        Destination path to store downloaded file.
    filename: str, optional
        Filename of the downloaded file.
    overwrite : bool, optional
        Whether to overwrite the destination file if it already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    retries : integer, default 5
        The number of times to attempt downloading in case of failure or non 200 return codes.
    verify_ssl : bool, default True
        Verify SSL certificates.
    log : bool, default True
        Whether to print the progress for download

    Returns
    ---

    str
        The file path of the downloaded file.
    """
    if filename is None:
        filename = url.split('/')[-1]
        # Empty filenames are invalid
        assert filename, 'Can\'t construct file-name from this URL. ' \
            'Please set the `path` option manually.'
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
                    infoLogger('Downloading %s from %s...' % (file_, url))
                r = requests.get(url, stream=True, verify=verify_ssl)
                if r.status_code != 200:
                    errorLogger("Failed downloading url %s" % url, RuntimeError)
                with open(file_, 'wb') as f:
                    for chunk in tqdm.tqdm(r.iter_content(chunk_size=1024), leave=False, desc="վ'ᴗ' ի-"):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                if sha1_hash and not check_sha1(file_, sha1_hash):
                    errorLogger(
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
                        infoLogger("download failed, retrying, {} attempt{} left"
                              .format(retries, 's' if retries > 1 else ''))

    return file_


def extract_archive(file_, target_dir, overwrite=False):
    """Extract archive file.

    Codes borrowed from dgl/data/utils.py

    Parameters
    ---

    file_ : str
        Absolute path of the archive file.
    target_dir : str
        Target directory of the archive to be uncompressed.
    overwrite : bool, default True
        Whether to overwrite the contents inside the directory.
        By default always overwrites.
    """
    if os.path.exists(target_dir) and not overwrite:
        return
    infoLogger('Extracting file to {}'.format(target_dir))
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
        errorLogger('Unrecognized file type: ' + file_)


def check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.

    Codes borrowed from dgl/data/utils.py

    Parameters
    ---

    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.

    Returns
    ---

    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash

