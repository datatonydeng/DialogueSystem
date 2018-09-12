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

from overrides import overrides
from typing import List, Union, Iterator
import tensorflow as tf

import numpy as np
import tensorflow_hub as hub

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.data.utils import zero_pad

log = get_logger(__name__)


@register('elmo')
class ELMoEmbedder(Component):
    """
    ``ELMo`` (Embeddings from Language Models) representations are pre-trained contextual representations from
    large-scale bidirectional language models. See a paper `Deep contextualized word representations
    <https://arxiv.org/abs/1802.05365>`__ for more information about the algorithm and a detailed analysis.

    Parameters:
        spec: A ``ModuleSpec`` defining the Module to instantiate or a path where to load a ``ModuleSpec`` from via
            ``tenserflow_hub.load_module_spec`` by using `TensorFlow Hub <https://www.tensorflow.org/hub/overview>`__.
        dim: Dimensionality of output token embeddings of ELMo model.
        pad_zero: Whether to use pad samples or not.
        mean: Whether to return a mean ELMo embedding of tokens per sample.

    Examples:
        You can use ELMo models from DeepPavlov as usual `TensorFlow Hub Module
        <https://www.tensorflow.org/hub/modules/google/elmo/2>`_.

        >>> import tensorflow_hub as hub
        >>> elmo = hub.Module("http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz",
        trainable=True)
        >>> embeddings = elmo(["это предложение", "word"], signature="default", as_dict=True)["elmo"]

        You can also embed tokenized sentences.

        >>> tokens_input = [["мама", "мыла", "раму"], ["рама", "", ""]]
        >>> tokens_length = [3, 1]
        >>> embeddings = elmo(
                inputs={
                        "tokens": tokens_input,
                        "sequence_len": tokens_length
                        },
                signature="tokens",
                as_dict=True)["elmo"]

        You can also get ``hub.text_embedding_column`` like described `here
        <https://www.tensorflow.org/hub/tutorials/text_classification_with_tf_hub#feature_columns>`_.


    """
    def __init__(self, spec: str, dim: int = 1024, pad_zero: bool = False, mean: bool = False,
                **kwargs) -> None:

        self.spec = spec if '://' in spec else str(expand_path(spec))
        self.dim = dim
        self.pad_zero = pad_zero
        self.mean = mean
        self.elmo_outputs, self.sess,  self.tokens_ph,  self.tokens_length_ph = self._load()

    def _load(self):
        """
        Load a ELMo TensorFlow Hub Module from a self.spec.

        Returns:
            ELMo pre-trained model wrapped in TenserFlow Hub Module.
        """
        elmo_module = hub.Module(self.spec, trainable=False)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess = tf.Session(config=sess_config)

        tokens_ph = tf.placeholder(shape=(None, None), dtype=tf.string, name='tokens')
        tokens_length_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name='tokens_length')

        elmo_outputs = elmo_module(
                                    inputs={
                                        "tokens": tokens_ph,
                                        "sequence_len": tokens_length_ph
                                    },
                                    signature="tokens",
                                    as_dict=True
                                   )

        sess.run(tf.global_variables_initializer())

        return elmo_outputs, sess, tokens_ph, tokens_length_ph

    @overrides
    def __call__(self, batch: List[List[str]],
                 *args, **kwargs) -> Union[List[np.ndarray], np.ndarray]:
        """
        Embed sentences from a batch.

        Args:
            batch: A list of tokenized text samples.

        Returns:
            A batch of ELMo embeddings.
        """
        if not batch:
            empty_vec = np.zeros(self.dim, dtype=np.float32)
            return [empty_vec] if self.mean else [[empty_vec]]

        filled_batch = []
        for batch_line in batch:
            batch_line = batch_line if batch_line else ['']
            filled_batch.append(batch_line)

        batch = filled_batch

        tokens_length = [len(batch_line) for batch_line in batch]
        tokens_length_max = max(tokens_length)
        batch = [batch_line + ['']*(tokens_length_max - len(batch_line)) for batch_line in batch]

        elmo_outputs = self.sess.run(
                                    self.elmo_outputs,
                                    feed_dict=
                                    {
                                        self.tokens_ph: batch,
                                        self.tokens_length_ph: tokens_length,
                                    }
                                    )

        if self.mean:
            batch = elmo_outputs['default']

            dim0, dim1 = batch.shape

            if self.dim != dim1:
                batch = np.resize(batch, (dim0,self.dim))
        else:
            batch = elmo_outputs['elmo']

            dim0, dim1, dim2 = batch.shape

            if self.dim != dim2:
                batch = np.resize(batch, (dim0, dim1, self.dim))

            batch = [batch_line[:length_line] for length_line, batch_line in zip(tokens_length, batch)]

            if self.pad_zero:
                batch = zero_pad(batch)

        return batch

    def __iter__(self) -> Iterator:
        """
        Iterate over all words from a ELMo model vocabulary.
        The ELMo model vocabulary consists of ``['<S>', '</S>', '<UNK>']``.

        Returns:
            An iterator of three elements ``['<S>', '</S>', '<UNK>']``.
        """

        yield from ['<S>', '</S>', '<UNK>']
