from deeppavlov.core.data.dataset_reader import DatasetReader
from pathlib import Path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import download_decompress, mark_done, is_done
from deeppavlov.core.commands.utils import get_deeppavlov_root, expand_path
from typing import Dict, List, Union


@register('insurance_reader')
class InsuranceReader(DatasetReader):
    
    def read(self, data_path: str, **kwargs) -> Dict[str, List[Dict[str, Union[int, List[int]]]]]:
        """Read the InsuranceQA data from files and forms the dataset.

        Args:
            data_path: A path to a folder where dataset files are stored.
            **kwargs: Other parameters.

        Returns:
            A dictionary containing training, validation and test parts of the dataset obtainable via
            ``train``, ``valid`` and ``test`` keys.
        """

        data_path = expand_path(data_path)
        self._download_data(data_path)
        dataset = {'train': None, 'valid': None, 'test': None}
        train_fname = Path(data_path) / 'insuranceQA-master/V1/question.train.token_idx.label'
        valid_fname = Path(data_path) / 'insuranceQA-master/V1/question.dev.label.token_idx.pool'
        test_fname = Path(data_path) / 'insuranceQA-master/V1/question.test1.label.token_idx.pool'
        self.idxs2cont_vocab = self._build_context2toks_vocabulary(train_fname, valid_fname, test_fname)
        dataset["valid"] = self._preprocess_data_valid_test(valid_fname)
        dataset["train"] = self._preprocess_data_train(train_fname)
        dataset["test"] = self._preprocess_data_valid_test(test_fname)

        return dataset
    
    def _download_data(self, data_path):
        """Download archive with the InsuranceQA dataset files and decompress if there is no dataset files in `data_path`.

        Args:
            data_path: A path to a folder where dataset files are stored.
        """
        if not is_done(Path(data_path)):
            download_decompress(url="http://files.deeppavlov.ai/datasets/insuranceQA-master.zip",
                                download_path=data_path)
            mark_done(data_path)

    def _build_context2toks_vocabulary(self, train_f, val_f, test_f):
        contexts = []
        with open(train_f, 'r') as f:
            data = f.readlines()
        for eli in data:
            eli = eli[:-1]
            c, _ = eli.split('\t')
            contexts.append(c)
        with open(val_f, 'r') as f:
            data = f.readlines()
        for eli in data:
            eli = eli[:-1]
            _, c, _ = eli.split('\t')
            contexts.append(c)
        with open(test_f, 'r') as f:
            data = f.readlines()
        for eli in data:
            eli = eli[:-1]
            _, c, _ = eli.split('\t')
            contexts.append(c)
        idxs2cont_vocab = {el[1]: el[0] for el in enumerate(contexts)}
        return idxs2cont_vocab

    def _preprocess_data_train(self, fname):
        positive_responses_pool = []
        contexts = []
        responses = []
        with open(fname, 'r') as f:
            data = f.readlines()
        for eli in data:
            eli = eli[:-1]
            q, pa = eli.split('\t')
            pa_list = [int(el) - 1 for el in pa.split(' ')]
            for elj in pa_list:
                contexts.append(self.idxs2cont_vocab[q])
                responses.append(elj)
                positive_responses_pool.append(pa_list)
        train_data = [{"context": el[0], "response": el[1],
                       "pos_pool": el[2], "neg_pool": None}
                      for el in zip(contexts, responses, positive_responses_pool)]
        return train_data
    
    def _preprocess_data_valid_test(self, fname):
        pos_responses_pool = []
        neg_responses_pool = []
        contexts = []
        pos_responses = []
        with open(fname, 'r') as f:
            data = f.readlines()
        for eli in data:
            eli = eli[:-1]
            pa, q, na = eli.split('\t')
            pa_list = [int(el) - 1 for el in pa.split(' ')]
            for elj in pa_list:
                contexts.append(self.idxs2cont_vocab[q])
                pos_responses.append(elj)
                pos_responses_pool.append(pa_list)
                nas = [int(el) - 1 for el in na.split(' ')]
                nas = [el for el in nas if el not in pa_list]
                neg_responses_pool.append(nas)
        data = [{"context": el[0], "response": el[1], "pos_pool": el[2], "neg_pool": el[3]}
                for el in zip(contexts, pos_responses, pos_responses_pool, neg_responses_pool)]
        return data
