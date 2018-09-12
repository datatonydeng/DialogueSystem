# -*- coding: utf-8 -*-

"""
Goal-Oriented Bot(gobot) Demo Example using DeepPavlov
Author: Tony Deng
Date: Sep 2018
"""

import json
import os
import deeppavlov
from  deeppavlov.download import deep_download
from deeppavlov.core.commands.train import train_evaluate_model_from_config
from deeppavlov.core.commands.infer import build_model_from_config

if not os.path.isdir("gobot"):
    os.mkdir("gobot")

"""
Step 1: Vocabulary Configurations
- dataset_reader: configuration of dataset reader component (is responsible for data download and saving to disk);
- dataset_iterator: configuration of dataset iterator component (is responsible for making batches (sequences) of data that will be further fed to pipe components);
- metadata: extra info (urls for data download and telegram configuration), a list of data which should be downloaded in order for config to work;
- train: training process configuration (size of batches, number of training epochs, etc.);
- chainer: specifies data flow (which components are run and in what order);
"""

vocab_config = {}

"dataset_reader"
dstc2_reader_comp_config = {
    'name': 'dstc2_reader',
    'data_path': 'dstc2'
}
vocab_config['dataset_reader'] = dstc2_reader_comp_config

"dataset_iterator"
dialog_iterator_comp_config = {
    'name': 'dialog_iterator'
}
vocab_config['dataset_iterator'] = dialog_iterator_comp_config

"metadata"
dstc2_download_config = {
    'url': 'http://files.deeppavlov.ai/datasets/dstc2_v2.tar.gz',
    'subdir': 'dstc2'
}
vocab_config['metadata'] = {}
vocab_config['metadata']['download'] = [dstc2_download_config]

"training, nothing to train here, just to build a dictionary"
vocab_config['train'] = {}

"chainer"
vocab_config['chainer'] = {}
vocab_config['chainer']['in'] = ['utterance']  # utterance as input
vocab_config['chainer']['in_y'] = []
vocab_config['chainer']['out'] = []

"""
Step 2: Component Configurations
 - name: registered name of a component
 - save_path: path to save the component
 - load_path: path to load the component
 - fit_on: a list of data fields to fit on 
 - level: on which level to operate ('token' level or 'char')
 - tokenizer: if input is a string, then it will be tokenized by the tokenizer, optional parameter
"""

vocab_comp_config = {
    'name': 'default_vocab',
    'save_path': 'vocabs/token.dict',
    'load_path': 'vocabs/token.dict',
    'fit_on': ['utterance'],
    'level': 'token',
    'tokenizer': {'name': 'split_tokenizer'},
    'main': True
}
# chainer.pipe: a list of consequently run components
vocab_config['chainer']['pipe'] = [vocab_comp_config]

json.dump(vocab_config, open("gobot/vocab_config.json", 'wt'))

""" Download "dstc2_v2" dataset, need to do only once """
deep_download(['--config', 'gobot/vocab_config.json'])
dstc2_path = deeppavlov.__path__[0] + '/../download/dstc2' # Data was downloaded to dstc2_path


"""
Step 3: Vocabulary Building
"""

train_evaluate_model_from_config("gobot/vocab_config.json")

vocabs_path = deeppavlov.__path__[0] + '/../download/vocabs'
vocab_comp_config['in'] = ['utterance']
vocab_comp_config['out'] = ['utterance_token_indices']

vocab_config['chainer']['pipe'] = [vocab_comp_config]
vocab_config['chainer']['out'] = ['utterance_token_indices']

# model = build_model_from_config(vocab_config)
# model(['hi'])

"""
Step 4: Gobot Configurations
"""

db_config = {}

"""dataset_reader, dataset_iterator and metadata will be the same as for vocabulary only"""
db_config['dataset_reader'] = dstc2_reader_comp_config
db_config['dataset_iterator'] = dialog_iterator_comp_config
db_config['metadata'] = {}
db_config['metadata']['download'] = [dstc2_download_config]

"""
X here is a dict 'x' containing context 'text', 'intents', db_result', 'prev_resp_act'
Y here is a dict 'y' containing response 'act' and 'text'
Prediction 'y_predicted' here will be only 'text'
"""
db_config['chainer'] = {}
db_config['chainer']['in'] = ['x']
db_config['chainer']['in_y'] = ['y']
db_config['chainer']['out'] = ['y_predicted']

"""
The bot consists (pipe section) of two components:
- default_vocab (or DefaultVocabulary): component that remembers all tokens from user utterances.
- id: reference name for the component
"""
vocab_comp_config = {
    'name': 'default_vocab',
    'id': 'token_vocab',
    'load_path': 'vocabs/token.dict',
    'save_path': 'vocabs/token.dict',
    'fit_on': ['x'],
    'level': 'token',
    'tokenizer': {'name': 'split_tokenizer'}
}

"""Adding vocabulary to chainer:"""
db_config['chainer']['pipe'] = []
db_config['chainer']['pipe'].append(vocab_comp_config)


"""
go_bot (or GoalOrientedBot) components:
 - slot_filler: user utterance outputs mentioned slots
 - tracker: update dialogue state
 - tokenizer: converts user utterance in string format (x) to tokens
 - bow_embedder: embeds the tokens with bag-of-words
 - embedder: embeds the utterance as a mean of embeddings of utterance tokens
 - concatenates embeddings and passes it as an input to a recurrent neural network (RNN)
 - trains RNN (with LongShortTermMemory (LSTM) as a core graph) that outputs an action label
 - loads templates (mapping from labels to string) using template_path and template_type and converts action label to string
"""

bot_with_db_comp_config = {
    'name': 'go_bot',
    'in': ['x'],
    'in_y': ['y'],
    'out': ['y_predicted'],
    'word_vocab': None,
    'bow_embedder': {"name": "bow"},
    'embedder': None,
    'slot_filler': None,
    'template_path': 'dstc2/dstc2-templates.txt',
    'template_type': 'DualTemplate',
    'database': None,
    'api_call_action': 'api_call',
    'network_parameters': {
        'load_path': 'gobot_dstc2_db/model',
        'save_path': 'gobot_dstc2_db/model',
        'dense_size': 64,
        'hidden_size': 128,
        'learning_rate': 0.002,
        'attention_mechanism': None
    },
    'tokenizer': {'name': 'stream_spacy_tokenizer',
                  'lowercase': False},
    'tracker': {'name': 'featurized_tracker',
                'slot_names': ['pricerange', 'this', 'area', 'food', 'name']},
    'main': True,
    'debug': False
}

""" use vocabulary by reference """
bot_with_db_comp_config['word_vocab'] = '#token_vocab'

""" Announcing slot filler component. We assume that slot filler is already trained, and use it by referencing it's config """
slot_filler_comp_config = {
    'config_path': deeppavlov.__path__[0] + '/../deeppavlov/configs/ner/slotfill_dstc2.json'
}

""" Adding slot filler to bot component """
bot_with_db_comp_config['slot_filler'] = slot_filler_comp_config

""" Adding bot_comp_config to pipe """
db_config['chainer']['pipe'].append(bot_with_db_comp_config)

""" Creating database component config """
db_comp_config = {
    'name': 'sqlite_database',
    'id': 'restaurant_database',
    'save_path': 'dstc2/resto.sqlite',
    'primary_keys': ['name'],
    'table_name': 'mytable'
}

""" Adding vocab and database components to pipe """
db_config['chainer']['pipe'] = []
db_config['chainer']['pipe'].append(vocab_comp_config)
db_config['chainer']['pipe'].append(db_comp_config)

""" Adding database to bot component config """
bot_with_db_comp_config['database'] = '#restaurant_database'

""" Adding bot component to pipe """
db_config['chainer']['pipe'].append(bot_with_db_comp_config)
json.dump(db_config, open("gobot/db_config.json", 'wt'))

"""
Model training and building
"""

""" Train gobot_dstc2_db model """
# train_evaluate_model_from_config("gobot/db_config.json")

""" build model """
model = build_model_from_config(db_config)

"""
Dialogue system
"""

" starting new dialog, if the cell is running, please do not run other cells in parallel -- there is a possibility of a hangup"
model.reset()
utterance = ""

while utterance != 'exit':
    print(">> " + model([utterance])[0])
    utterance = input(':: ')