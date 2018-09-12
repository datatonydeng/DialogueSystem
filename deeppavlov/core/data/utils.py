"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
from hashlib import md5
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Union
import requests
from tqdm import tqdm
import tarfile
import gzip
import zipfile
import re
import shutil
import secrets

import requests
from tqdm import tqdm
import numpy as np

from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)

_MARK_DONE = '.done'

tqdm.monitor_interval = 0


def get_download_token():
    token_file = Path.home() / '.deeppavlov'
    if not token_file.exists():
        token_file.write_text(secrets.token_urlsafe(32), encoding='utf8')

    return token_file.read_text(encoding='utf8').strip()


def simple_download(url: str, destination: [Path, str]):
    CHUNK = 32 * 1024

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    headers = {'dp-token': get_download_token()}
    r = requests.get(url, stream=True, headers=headers)
    total_length = int(r.headers.get('content-length', 0))

    log.info('Downloading from {} to {}'.format(url, destination))
    with destination.open('wb') as f, tqdm(total=total_length, unit='B', unit_scale=True) as pbar:
        done = False
        downloaded = 0
        while not done:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if chunk:  # filter out keep-alive new chunks
                    downloaded += len(chunk)
                    pbar.update(len(chunk))
                    f.write(chunk)
            if downloaded < total_length:
                log.warn(f'Download stopped abruptly, trying to resume from {downloaded} to reach {total_length}')
                headers['Range'] = f'bytes={downloaded}-'
                r = requests.get(url, headers=headers, stream=True)
                if total_length - downloaded != int(r.headers['content-length']):
                    raise RuntimeError('It looks like the server does not support resuming downloads')
            else:
                done = True


def download(dest_file_path: [List[Union[str, Path]]], source_url: str, force_download=True):
    """Download a file from URL to one or several target locations

    Args:
        dest_file_path: path or list of paths to the file destination files (including file name)
        source_url: the source URL
        force_download: download file if it already exists, or not

    """

    if isinstance(dest_file_path, list):
        dest_file_paths = [Path(path) for path in dest_file_path]
    else:
        dest_file_paths = [Path(dest_file_path).absolute()]

    if not force_download:
        to_check = list(dest_file_paths)
        dest_file_paths = []
        for p in to_check:
            if p.exists():
                log.info(f'File already exists in {p}')
            else:
                dest_file_paths.append(p)

    if dest_file_paths:
        cache_dir = os.getenv('DP_CACHE_DIR')
        cached_exists = False
        if cache_dir:
            first_dest_path = Path(cache_dir) / md5(source_url.encode('utf8')).hexdigest()[:15]
            cached_exists = first_dest_path.exists()
        else:
            first_dest_path = dest_file_paths.pop()

        if not cached_exists:
            first_dest_path.parent.mkdir(parents=True, exist_ok=True)

            simple_download(source_url, first_dest_path)
        else:
            log.info(f'Found cached {source_url} in {first_dest_path}')

        for dest_path in dest_file_paths:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(first_dest_path), str(dest_path))


def untar(file_path, extract_folder=None):
    """Simple tar archive extractor

    Args:
        file_path: path to the tar file to be extracted
        extract_folder: folder to which the files will be extracted

    """
    file_path = Path(file_path)
    if extract_folder is None:
        extract_folder = file_path.parent
    extract_folder = Path(extract_folder)
    tar = tarfile.open(file_path)
    tar.extractall(extract_folder)
    tar.close()


def ungzip(file_path, extract_path: Path=None):
    """Simple .gz archive extractor

        Args:
            file_path: path to the gzip file to be extracted
            extract_folder: folder to which the files will be extracted

        """
    CHUNK = 16 * 1024
    file_path = Path(file_path)
    extract_path = extract_path or file_path.with_suffix('')

    with gzip.open(file_path, 'rb') as fin, extract_path.open('wb') as fout:
        while True:
            block = fin.read(CHUNK)
            if not block:
                break
            fout.write(block)


def download_decompress(url: str, download_path: [Path, str], extract_paths=None):
    """Download and extract .tar.gz or .gz file to one or several target locations.
    The archive is deleted if extraction was successful.

    Args:
        url: URL for file downloading
        download_path: path to the directory where downloaded file will be stored
        until the end of extraction
        extract_paths: path or list of paths where contents of archive will be extracted
    """
    file_name = Path(urlparse(url).path).name
    download_path = Path(download_path)

    if extract_paths is None:
        extract_paths = [download_path]
    elif isinstance(extract_paths, list):
        extract_paths = [Path(path) for path in extract_paths]
    else:
        extract_paths = [Path(extract_paths)]

    cache_dir = os.getenv('DP_CACHE_DIR')
    extracted = False
    if cache_dir:
        cache_dir = Path(cache_dir)
        url_hash = md5(url.encode('utf8')).hexdigest()[:15]
        arch_file_path = cache_dir / url_hash
        extracted_path = cache_dir / (url_hash + '_extracted')
        extracted = extracted_path.exists()
        if not extracted and not arch_file_path.exists():
            simple_download(url, arch_file_path)
    else:
        arch_file_path = download_path / file_name
        simple_download(url, arch_file_path)
        extracted_path = extract_paths.pop()

    if not extracted:
        log.info('Extracting {} archive into {}'.format(arch_file_path, extracted_path))
        extracted_path.mkdir(parents=True, exist_ok=True)

        if file_name.endswith('.tar.gz'):
            untar(arch_file_path, extracted_path)
        elif file_name.endswith('.gz'):
            ungzip(arch_file_path, extracted_path / Path(file_name).with_suffix('').name)
        elif file_name.endswith('.zip'):
            with zipfile.ZipFile(arch_file_path, 'r') as zip_ref:
                zip_ref.extractall(extracted_path)
        else:
            raise RuntimeError(f'Trying to extract an unknown type of archive {file_name}')

        if not cache_dir:
            arch_file_path.unlink()

    for extract_path in extract_paths:
        for src in extracted_path.iterdir():
            dest = extract_path / src.name
            if src.is_dir():
                copytree(src, dest)
            else:
                extract_path.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(src), str(dest))


def copytree(src: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        f_dest = dest / f.name
        if f.is_dir():
            copytree(f, f_dest)
        else:
            shutil.copy(str(f), str(f_dest))


def load_vocab(vocab_path):
    vocab_path = Path(vocab_path)
    with vocab_path.open(encoding='utf8') as f:
        return f.read().split()


def mark_done(path):
    mark = Path(path) / _MARK_DONE
    mark.touch(exist_ok=True)


def is_done(path):
    mark = Path(path) / _MARK_DONE
    return mark.is_file()


def tokenize_reg(s):
    pattern = "[\w]+|[‑–—“”€№…’\"#$%&\'()+,-./:;<>?]"
    return re.findall(re.compile(pattern), s)


def zero_pad(batch, dtype=np.float32):
    if len(batch) == 1 and len(batch[0]) == 0:
        return np.array([], dtype=dtype)
    batch_size = len(batch)
    max_len = max(len(utterance) for utterance in batch)
    if isinstance(batch[0][0], (int, np.int)):
        padded_batch = np.zeros([batch_size, max_len], dtype=np.int32)
        for n, utterance in enumerate(batch):
            padded_batch[n, :len(utterance)] = utterance
    else:
        n_features = len(batch[0][0])
        padded_batch = np.zeros([batch_size, max_len, n_features], dtype=dtype)
        for n, utterance in enumerate(batch):
            for k, token_features in enumerate(utterance):
                padded_batch[n, k] = token_features
    return padded_batch


def zero_pad_char(batch, dtype=np.float32):
    if len(batch) == 1 and len(batch[0]) == 0:
        return np.array([], dtype=dtype)
    batch_size = len(batch)
    max_len = max(len(utterance) for utterance in batch)
    max_token_len = max(len(ch) for token in batch for ch in token)
    if isinstance(batch[0][0][0], (int, np.int)):
        padded_batch = np.zeros([batch_size, max_len, max_token_len], dtype=np.int32)
        for n, utterance in enumerate(batch):
            for k, token in enumerate(utterance):
                padded_batch[n, k, :len(token)] = token
    else:
        n_features = len(batch[0][0][0])
        padded_batch = np.zeros([batch_size, max_len, max_token_len, n_features], dtype=dtype)
        for n, utterance in enumerate(batch):
            for k, token in enumerate(utterance):
                for q, char_features in enumerate(token):
                    padded_batch[n, k, q] = char_features
    return padded_batch


def get_all_elems_from_json(search_json, search_key):
    result = []
    if isinstance(search_json, dict):
        for key in search_json:
            if key == search_key:
                result.append(search_json[key])
            else:
                result.extend(get_all_elems_from_json(search_json[key], search_key))
    elif isinstance(search_json, list):
        for item in search_json:
            result.extend(get_all_elems_from_json(item, search_key))

    return result


def check_nested_dict_keys(check_dict: dict, keys: list):
    if isinstance(keys, list) and len(keys) > 0:
        element = check_dict
        for key in keys:
            if isinstance(element, dict) and key in element.keys():
                element = element[key]
            else:
                return False
        return True
    else:
        return False


def jsonify_data(input):
    if isinstance(input, list):
        result = [jsonify_data(item) for item in input]
    elif isinstance(input, tuple):
        result = [jsonify_data(item) for item in input]
    elif isinstance(input, dict):
        result = {}
        for key in input.keys():
            result[key] = jsonify_data(input[key])
    elif isinstance(input, np.ndarray):
        result = input.tolist()
    elif isinstance(input, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                            np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        result = int(input)
    elif isinstance(input, (np.float_, np.float16, np.float32, np.float64)):
        result = float(input)
    else:
        result = input
    return result
