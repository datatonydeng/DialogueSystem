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

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import deeppavlov
from deeppavlov.core.commands.utils import get_deeppavlov_root, set_deeppavlov_root, expand_path
from deeppavlov.core.common.file import read_json
from deeppavlov.core.data.utils import download, download_decompress, get_all_elems_from_json
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument('--config', '-c', help="path to a pipeline json config", type=str,
                    default=None)
parser.add_argument('-all', action='store_true',
                    help="Download everything. Warning! There should be at least 10 GB space"
                         " available on disk.")


def get_config_downloads(config_path):
    dp_root_back = get_deeppavlov_root()
    config = read_json(config_path)
    set_deeppavlov_root(config)

    downloads = set()
    if 'metadata' in config and 'download' in config['metadata']:
        for resource in config['metadata']['download']:
            if isinstance(resource, str):
                resource = {
                    'url': resource
                }

            url = resource['url']
            dest = expand_path(resource.get('subdir', ''))

            downloads.add((url, dest))

    config_references = [expand_path(config_ref) for config_ref in get_all_elems_from_json(config, 'config_path')]

    downloads |= {(url, dest) for config in config_references for url, dest in get_config_downloads(config)}

    set_deeppavlov_root({'deeppavlov_root': dp_root_back})

    return downloads


def get_configs_downloads(config_path=None):
    all_downloads = defaultdict(set)

    if config_path:
        configs = [config_path]
    else:
        configs = list(Path(deeppavlov.__path__[0], 'configs').glob('**/*.json'))

    for config_path in configs:
        for url, dest in get_config_downloads(config_path):
            all_downloads[url].add(dest)

    return all_downloads


def download_resource(url, dest_paths):
    dest_paths = list(dest_paths)

    if url.endswith(('.tar.gz', '.gz', '.zip')):
        download_path = dest_paths[0].parent
        download_decompress(url, download_path, dest_paths)
    else:
        file_name = url.split('/')[-1]
        dest_files = [dest_path / file_name for dest_path in dest_paths]
        download(dest_files, url)


def download_resources(args):
    if not args.all and not args.config:
        log.error('You should provide either skill config path or -all flag')
        sys.exit(1)
    elif args.all:
        downloads = get_configs_downloads()
    else:
        config_path = Path(args.config).resolve()
        downloads = get_configs_downloads(config_path=config_path)

    for url, dest_paths in downloads.items():
        download_resource(url, dest_paths)


def deep_download(args: [str, Path, list]=None):
    if isinstance(args, (str, Path)):
        args = ['-c', str(args)]  # if args is a path to config
    args = parser.parse_args(args)
    log.info("Downloading...")
    download_resources(args)
    log.info("\nDownload successful!")


if __name__ == "__main__":
    deep_download()
