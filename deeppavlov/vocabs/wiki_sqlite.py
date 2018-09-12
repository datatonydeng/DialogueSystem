# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Any, Optional, Union

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.dataset_iterators.sqlite_iterator import SQLiteDataIterator

logger = get_logger(__name__)


@register('wiki_sqlite_vocab')
class WikiSQLiteVocab(SQLiteDataIterator):
    """Get content from SQLite database by document ids.

    Args:
        data_url: an URL where to download a DB from
        data_dir:  a directory where to save downloaded DB to
        join_docs: whether to join extracted docs with ' ' or not
        shuffle: whether to shuffle data or not

    Attributes:
        join_docs: whether to join extracted docs with ' ' or not

    """

    def __init__(self, data_url: str, data_dir: str = '', join_docs: bool=True, shuffle: bool=False, **kwargs):
        super().__init__(data_dir=data_dir, data_url=data_url, shuffle=shuffle)
        self.join_docs = join_docs

    def __call__(self, doc_ids: Optional[List[List[Any]]] = None, *args, **kwargs) -> List[Union[str, List[str]]]:
        """Get the contents of files, stacked by space or as they are.

        Args:
            doc_ids: a batch of lists of ids to get contents for

        Returns:
            a list of contents / list of lists of contents
        """
        all_contents = []
        if not doc_ids:
            logger.warn('No doc_ids are provided in WikiSqliteVocab, return all docs')
            doc_ids = [self.get_doc_ids()]

        for ids in doc_ids:
            contents = [self.get_doc_content(doc_id) for doc_id in ids]
            if self.join_docs:
                contents = ' '.join(contents)
            all_contents.append(contents)

        return all_contents
