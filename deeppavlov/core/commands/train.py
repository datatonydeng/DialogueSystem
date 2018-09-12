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

import datetime
import importlib
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import List, Callable, Tuple, Dict, Union

from deeppavlov.core.commands.utils import expand_path, set_deeppavlov_root, import_packages
from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.registry import get_model
from deeppavlov.core.common.metrics_registry import get_metrics_by_names
from deeppavlov.core.common.params import from_params
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.core.data.data_fitting_iterator import DataFittingIterator
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.estimator import Estimator
from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


def prettify_metrics(metrics: dict, precision: int = 4) -> OrderedDict:
    """Prettifies the dictionary of metrics."""
    prettified_metrics = OrderedDict()
    for key, value in metrics:
        value = round(value, precision)
        prettified_metrics[key] = value
    return prettified_metrics


def _fit(model: Estimator, iterator: DataLearningIterator, train_config) -> Estimator:
    x, y = iterator.get_instances('train')
    model.fit(x, y)
    model.save()
    return model


def _fit_batches(model: Estimator, iterator: DataFittingIterator, train_config) -> Estimator:
    model.fit_batches(iterator, batch_size=train_config['batch_size'])
    model.save()
    return model


def fit_chainer(config: dict, iterator: Union[DataLearningIterator, DataFittingIterator]) -> Chainer:
    """Fit and return the chainer described in corresponding configuration dictionary."""
    chainer_config: dict = config['chainer']
    chainer = Chainer(chainer_config['in'], chainer_config['out'], chainer_config.get('in_y'))
    for component_config in chainer_config['pipe']:
        component = from_params(component_config, mode='train')
        if 'fit_on' in component_config:
            component: Estimator

            preprocessed = chainer(*iterator.get_instances('train'), to_return=component_config['fit_on'])
            if len(component_config['fit_on']) == 1:
                preprocessed = [preprocessed]
            else:
                preprocessed = zip(*preprocessed)
            component.fit(*preprocessed)
            component.save()

        if 'fit_on_batch' in component_config:
            component: Estimator
            component.fit_batches(iterator, config['train']['batch_size'])
            component.save()

        if 'in' in component_config:
            c_in = component_config['in']
            c_out = component_config['out']
            in_y = component_config.get('in_y', None)
            main = component_config.get('main', False)
            chainer.append(component, c_in, c_out, in_y, main)
    return chainer


def train_evaluate_model_from_config(config: [str, Path, dict], to_train: bool = True, to_validate: bool = True) -> None:
    """Make training and evaluation of the model described in corresponding configuration file."""
    if isinstance(config, (str, Path)):
        config = read_json(config)
    set_deeppavlov_root(config)

    import_packages(config.get('metadata', {}).get('imports', []))

    dataset_config = config.get('dataset', None)

    if dataset_config:
        config.pop('dataset')
        ds_type = dataset_config['type']
        if ds_type == 'classification':
            reader = {'name': 'basic_classification_reader'}
            iterator = {'name': 'basic_classification_iterator'}
            config['dataset_reader'] = {**dataset_config, **reader}
            config['dataset_iterator'] = {**dataset_config, **iterator}
        else:
            raise Exception("Unsupported dataset type: {}".format(ds_type))

    data = []
    reader_config = config.get('dataset_reader', None)

    if reader_config:
        reader_config = config['dataset_reader']
        if 'class' in reader_config:
            c = reader_config.pop('class')
            try:
                module_name, cls_name = c.split(':')
                reader = getattr(importlib.import_module(module_name), cls_name)()
            except ValueError:
                e = ConfigError('Expected class description in a `module.submodules:ClassName` form, but got `{}`'
                                .format(c))
                log.exception(e)
                raise e
        else:
            reader = get_model(reader_config.pop('name'))()
        data_path = reader_config.pop('data_path', '')
        if isinstance(data_path, list):
            data_path = [expand_path(x) for x in data_path]
        else:
            data_path = expand_path(data_path)
        data = reader.read(data_path, **reader_config)
    else:
        log.warning("No dataset reader is provided in the JSON config.")

    iterator_config = config['dataset_iterator']
    iterator: Union[DataLearningIterator, DataFittingIterator] = from_params(iterator_config,
                                                                             data=data)

    train_config = {
        'metrics': ['accuracy'],
        'validate_best': to_validate,
        'test_best': True,
        'show_examples': False
    }

    try:
        train_config.update(config['train'])
    except KeyError:
        log.warning('Train config is missing. Populating with default values')

    metrics_functions = list(zip(train_config['metrics'], get_metrics_by_names(train_config['metrics'])))

    if to_train:
        model = fit_chainer(config, iterator)

        if callable(getattr(model, 'train_on_batch', None)):
            _train_batches(model, iterator, train_config, metrics_functions)
        elif callable(getattr(model, 'fit_batches', None)):
            _fit_batches(model, iterator, train_config)
        elif callable(getattr(model, 'fit', None)):
            _fit(model, iterator, train_config)
        elif not isinstance(model, Chainer):
            log.warning('Nothing to train')

    if train_config['validate_best'] or train_config['test_best']:
        # try:
        #     model_config['load_path'] = model_config['save_path']
        # except KeyError:
        #     log.warning('No "save_path" parameter for the model, so "load_path" will not be renewed')
        model = build_model_from_config(config, load_trained=True)
        log.info('Testing the best saved model')

        if train_config['validate_best']:
            report = {
                'valid': _test_model(model, metrics_functions, iterator,
                                     train_config.get('batch_size', -1), 'valid',
                                     show_examples=train_config['show_examples'])
            }

            print(json.dumps(report, ensure_ascii=False))

        if train_config['test_best']:
            report = {
                'test': _test_model(model, metrics_functions, iterator,
                                    train_config.get('batch_size', -1), 'test',
                                    show_examples=train_config['show_examples'])
            }

            print(json.dumps(report, ensure_ascii=False))


def _test_model(model: Component, metrics_functions: List[Tuple[str, Callable]],
                iterator: DataLearningIterator, batch_size=-1, data_type='valid',
                start_time: float=None, show_examples=False) -> Dict[str, Union[int, OrderedDict, str]]:
    if start_time is None:
        start_time = time.time()

    val_y_true = []
    val_y_predicted = []
    for x, y_true in iterator.gen_batches(batch_size, data_type, shuffle=False):
        y_predicted = list(model(list(x)))
        val_y_true += y_true
        val_y_predicted += y_predicted

    metrics = [(s, f(val_y_true, val_y_predicted)) for s, f in metrics_functions]

    report = {
        'eval_examples_count': len(val_y_true),
        'metrics': prettify_metrics(metrics),
        'time_spent': str(datetime.timedelta(seconds=round(time.time() - start_time + 0.5)))
    }

    if show_examples:
        try:
            report['examples'] = [{
                'x': x_item,
                'y_predicted': y_predicted_item,
                'y_true': y_true_item
            } for x_item, y_predicted_item, y_true_item in zip(x, y_predicted, y_true)]
        except NameError:
            log.warning(f'Could not log examples for {data_type}, assuming it\'s empty')

    return report


def _train_batches(model: NNModel, iterator: DataLearningIterator, train_config: dict,
                   metrics_functions: List[Tuple[str, Callable]]) -> NNModel:

    default_train_config = {
        'epochs': 0,
        'max_batches': 0,
        'batch_size': 1,

        'metric_optimization': 'maximize',

        'validation_patience': 5,
        'val_every_n_epochs': 0,

        'log_every_n_batches': 0,
        'log_every_n_epochs': 0,

        'validate_best': True,
        'test_best': True,
        'tensorboard_log_dir': None,
    }

    train_config = dict(default_train_config, **train_config)

    if 'train_metrics' in train_config:
        train_metrics_functions = list(zip(train_config['train_metrics'],
                                           get_metrics_by_names(train_config['train_metrics'])))
    else:
        train_metrics_functions = metrics_functions

    if train_config['metric_optimization'] == 'maximize':
        def improved(score, best):
            return score > best
        best = float('-inf')
    elif train_config['metric_optimization'] == 'minimize':
        def improved(score, best):
            return score < best
        best = float('inf')
    else:
        raise ConfigError('metric_optimization has to be one of {}'.format(['maximize', 'minimize']))

    i = 0
    epochs = 0
    examples = 0
    saved = False
    patience = 0
    log_on = train_config['log_every_n_batches'] > 0 or train_config['log_every_n_epochs'] > 0
    train_y_true = []
    train_y_predicted = []
    losses = []
    start_time = time.time()
    break_flag = False

    if train_config['tensorboard_log_dir'] is not None:
        import tensorflow as tf
        tb_log_dir = expand_path(train_config['tensorboard_log_dir'])

        tb_train_writer = tf.summary.FileWriter(str(tb_log_dir / 'train_log'))
        tb_valid_writer = tf.summary.FileWriter(str(tb_log_dir / 'valid_log'))

    try:
        while True:
            for x, y_true in iterator.gen_batches(train_config['batch_size']):
                if log_on:
                    y_predicted = list(model(list(x)))
                    train_y_true += y_true
                    train_y_predicted += y_predicted
                loss = model.train_on_batch(x, y_true)
                if loss is not None:
                    losses.append(loss)
                i += 1
                examples += len(x)

                if train_config['log_every_n_batches'] > 0 and i % train_config['log_every_n_batches'] == 0:
                    metrics = [(s, f(train_y_true, train_y_predicted)) for s, f in train_metrics_functions]
                    report = {
                        'epochs_done': epochs,
                        'batches_seen': i,
                        'examples_seen': examples,
                        'metrics': prettify_metrics(metrics),
                        'time_spent': str(datetime.timedelta(seconds=round(time.time() - start_time + 0.5)))
                    }

                    if train_config['show_examples']:
                        try:
                            report['examples'] = [{
                                'x': x_item,
                                'y_predicted': y_predicted_item,
                                'y_true': y_true_item
                            } for x_item, y_predicted_item, y_true_item in zip(x, y_predicted, y_true)]
                        except NameError:
                            log.warning('Could not log examples as y_predicted is not defined')

                    if losses:
                        report['loss'] = sum(losses)/len(losses)
                        losses = []
                    report = {'train': report}

                    if train_config['tensorboard_log_dir'] is not None:
                        for name, score in metrics:
                            metric_sum = tf.Summary(value=[tf.Summary.Value(tag='every_n_batches/' + name,
                                                                            simple_value=score), ])
                            tb_train_writer.add_summary(metric_sum, i)

                        if losses:
                            loss_sum = tf.Summary(value=[tf.Summary.Value(tag='every_n_batches/' + 'loss',
                                                                            simple_value=report['loss']), ])
                            tb_train_writer.add_summary(loss_sum, i)

                    print(json.dumps(report, ensure_ascii=False))
                    train_y_true.clear()
                    train_y_predicted.clear()

                if i >= train_config['max_batches'] > 0:
                    break_flag = True
                    break

                report = {
                    'epochs_done': epochs,
                    'batches_seen': i,
                    'train_examples_seen': examples,
                    'time_spent': str(datetime.timedelta(seconds=round(time.time() - start_time + 0.5)))
                }
                model.process_event(event_name='after_batch', data=report)
            if break_flag:
                break

            epochs += 1

            report = {
                'epochs_done': epochs,
                'batches_seen': i,
                'train_examples_seen': examples,
                'time_spent': str(datetime.timedelta(seconds=round(time.time() - start_time + 0.5)))
            }
            model.process_event(event_name='after_epoch', data=report)

            if train_config['log_every_n_epochs'] > 0 and epochs % train_config['log_every_n_epochs'] == 0\
                    and train_y_true:
                metrics = [(s, f(train_y_true, train_y_predicted)) for s, f in train_metrics_functions]
                report = {
                    'epochs_done': epochs,
                    'batches_seen': i,
                    'train_examples_seen': examples,
                    'metrics': prettify_metrics(metrics),
                    'time_spent': str(datetime.timedelta(seconds=round(time.time() - start_time + 0.5)))
                }

                if train_config['show_examples']:
                    try:
                        report['examples'] = [{
                            'x': x_item,
                            'y_predicted': y_predicted_item,
                            'y_true': y_true_item
                        } for x_item, y_predicted_item, y_true_item in zip(x, y_predicted, y_true)]
                    except NameError:
                        log.warning('Could not log examples')

                if losses:
                    report['loss'] = sum(losses)/len(losses)
                    losses = []

                if train_config['tensorboard_log_dir'] is not None:
                    for name, score in metrics:
                        metric_sum = tf.Summary(value=[tf.Summary.Value(tag='every_n_epochs/' + name,
                                                                        simple_value=score), ])
                        tb_train_writer.add_summary(metric_sum, epochs)

                    if losses:
                        loss_sum = tf.Summary(value=[tf.Summary.Value(tag='every_n_epochs/' + 'loss',
                                                                        simple_value=report['loss']), ])
                        tb_train_writer.add_summary(loss_sum, epochs)

                model.process_event(event_name='after_train_log', data=report)
                report = {'train': report}
                print(json.dumps(report, ensure_ascii=False))
                train_y_true.clear()
                train_y_predicted.clear()

            if train_config['val_every_n_epochs'] > 0 and epochs % train_config['val_every_n_epochs'] == 0:
                report = _test_model(model, metrics_functions, iterator,
                                     train_config['batch_size'], 'valid', start_time, train_config['show_examples'])
                report['epochs_done'] = epochs
                report['batches_seen'] = i
                report['train_examples_seen'] = examples

                metrics = list(report['metrics'].items())

                if train_config['tensorboard_log_dir'] is not None:
                    for name, score in metrics:
                        metric_sum = tf.Summary(value=[tf.Summary.Value(tag='every_n_epochs/' + name,
                                                                        simple_value=score), ])
                        tb_valid_writer.add_summary(metric_sum, epochs)

                m_name, score = metrics[0]
                if improved(score, best):
                    patience = 0
                    log.info('New best {} of {}'.format(m_name, score))
                    best = score
                    log.info('Saving model')
                    model.save()
                    saved = True
                else:
                    patience += 1
                    log.info('Did not improve on the {} of {}'.format(m_name, best))

                report['impatience'] = patience
                if train_config['validation_patience'] > 0:
                    report['patience_limit'] = train_config['validation_patience']

                model.process_event(event_name='after_validation', data=report)
                report = {'valid': report}
                print(json.dumps(report, ensure_ascii=False))

                if patience >= train_config['validation_patience'] > 0:
                    log.info('Ran out of patience')
                    break

            if epochs >= train_config['epochs'] > 0:
                break
    except KeyboardInterrupt:
        log.info('Stopped training')

    if not saved:
        log.info('Saving model')
        model.save()

    return model
