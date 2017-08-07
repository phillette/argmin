import datetime
import itertools

import matplotlib.pyplot as plt
import numpy as np

import batching
import mongoi
import stats
from argmin import util


def batch_size_info(history):
    return '%s (%s%%)' \
           % (history['batch_size'],
              round(
                  100 * history['batch_size']
                  / stats.COLLECTION_SIZE[history['db']][history['collection']],
                  2))


def compare(ids=None, param_to_compare='model_name', query=None):
    if not ids and not query:
        histories = get_all()
    elif ids:
        histories = get_many(ids)
    elif query:
        histories = list(mongoi.HistoryDb().all.find(query))
    # loss
    lines_loss = []
    plt.subplot(2, 2, 1)
    for i in range(len(histories)):
        line_i_loss = plot(
            history=histories[i],
            value_to_compare='loss',
            param_to_compare=param_to_compare,
            iter_key='iter'
        )
        lines_loss.append(line_i_loss)
    plt.legend(handles=lines_loss, loc=1)
    plt.xlabel(comparison_x_label(param_to_compare, 'loss'))
    plt.ylabel('standardized loss')
    # accuracy
    lines_accuracy = []
    plt.subplot(2, 2, 2)
    for i in range(len(histories)):
        line_i_accuracy = plot(
            history=histories[i],
            value_to_compare='accuracy',
            param_to_compare=param_to_compare,
            iter_key='iter'
        )
        lines_accuracy.append(line_i_accuracy)
    plt.legend(handles=lines_accuracy, loc=4)
    plt.xlabel(comparison_x_label(param_to_compare, 'accuracy'))
    plt.ylabel('training set accuracy')
    # tuning_accuracy
    lines_tune = []
    plt.subplot(2, 2, 3)
    for i in range(len(histories)):
        line_i_tune = plot(
            history=histories[i],
            value_to_compare='tuning_accuracy',
            param_to_compare=param_to_compare,
            iter_key='tuning_iter'
        )
        lines_tune.append(line_i_tune)
    plt.legend(handles=lines_tune, loc=4)
    plt.xlabel(comparison_x_label(param_to_compare, 'tuning_accuracy'))
    plt.ylabel('tuning set accuracy')
    plt.show()


def comparison_label(history, param_to_compare):
    if param_to_compare == 'batch_size':
        return batch_size_info(history)
    else:
        if param_to_compare in global_keys():
            return '%s' % history[param_to_compare]
        else:
            return '%s' % history['config'][param_to_compare]


def comparison_x(history, param_to_compare, value_to_compare, iter_key):
    if iter_key == 'tuning_iter':
        return np.array(history['tuning_iter'])
    else:
        return np.array(scale_iters_to_epochs(history, iter_key))


def comparison_x_label(param_to_compare, value_to_compare):
    return 'epoch'


def comparison_y(history, value_to_compare):
    if value_to_compare == 'loss':
        return scaled_loss(history)
    else:
        return np.array(history[value_to_compare])


def delete(ids):
    db = mongoi.HistoryDb()
    if isinstance(ids, int):
        db.db.all.delete_many({'id': ids})
    elif isinstance(ids, list):
        for id in ids:
            db.db.all.delete_many({'id': id})


def export(id):
    history = get_one(id)
    util.save_pickle(history, 'histories/%s.pkl' % id)


def get_all():
    db = mongoi.HistoryDb()
    return list(db.all.find_all())


def get_and_project(id, attr):
    db = mongoi.HistoryDb()
    try:
        history_projected = next(db.all.find({'id': id}, {attr: 1}))
    except:
        raise Exception('Could not find history with id %s' % id)
    return history_projected


def get_config_keys(docs):
    config_dicts = [d['config'] for d in docs]
    config_keys = \
        set(
            list(
                itertools.chain.from_iterable(
                    [c.keys() for c in config_dicts])))
    return config_keys


def get_doc_dicts(docs, config_keys):
    dicts = []
    for doc in docs:
        dict = {}
        for key in global_keys():
            dict[key] = doc[key]
        for key in config_keys:
            dict[key] = doc['config'][key]
        dicts.append(dict)
    return dicts


def iters_per_epoch(history):
    return batching.num_iters(
        db=history['db'],
        collection=history['collection'],
        batch_size=history['batch_size'],
        subset_size=history['subset_size']
    )


def get_many(ids):
    docs = []
    db = mongoi.HistoryDb()
    for id in ids:
        id = int(id)  # in case they come as numpy ints
        try:
            docs.append(db.all.get(id))
        except:
            raise Exception('History with id %s not found' % id)
    return docs


def get_one(id):
    id = int(id)  # in case it comes as a numpy int
    db = mongoi.HistoryDb()
    history = db.all.get(id)
    if not history:
        raise Exception('No history with id %s found' % id)
    return history


def global_keys():
    return ['id',
            'model_name',
            'db',
            'collection',
            'date_time',
            'batch_size',
            'subset_size',
            'epochs']


def import_one(id):
    history = util.load_pickle('histories/%s.pkl' % id)
    history['id'] = new_id()
    save(history)


def new_history(model_name,
                db,
                collection,
                batch_size,
                subset_size,
                config):
    history = {
        'id': new_id(),
        'model_name': model_name,
        'date_time': datetime.datetime.now(),
        'db': db,
        'collection': collection,
        'batch_size': batch_size,
        'subset_size': subset_size,
        'config': config,
        'epochs': 0,
        'iter': [],
        'loss': [],
        'accuracy': [],
        'epoch_change_loss': [],
        'epoch_change_accuracy': [],
        'tuning_iter': [],
        'tuning_accuracy': []
    }
    db = mongoi.HistoryDb()
    db.all.insert_one(history)
    return history['id']


def new_id():
    db = mongoi.HistoryDb()
    query = db.all.find_all()
    docs = query
    ids = [doc['id'] for doc in docs]
    return max(ids) + 1 if len(ids) > 0 else 1


def plot(history, param_to_compare, value_to_compare, iter_key):
    line, = plt.plot(
        comparison_x(
            history=history,
            param_to_compare=param_to_compare,
            value_to_compare=value_to_compare,
            iter_key=iter_key),
        comparison_y(
            history=history,
            value_to_compare=value_to_compare),
        label=comparison_label(history, param_to_compare))
    return line


def print_config(docs):
    print('----\t----\t----\t\t----\t\t----\t\t----\t\t----')
    print('id\talpha\thidden_size\tp_keep_ff\tp_keep_rnn\tgrad_clip_norm\tlambda')
    print('----\t----\t----\t\t----\t\t----\t\t----\t\t----')
    for doc in docs:
        print('%s\t%s\t%s\t\t%s\t\t%s\t\t%s\t\t%s' %
              (doc['id'],
               doc['config']['learning_rate'],
               doc['config']['hidden_size'],
               doc['config']['p_keep_ff'] if 'p_keep_ff' in doc['config'].keys() else doc['config']['p_keep'],
               doc['config']['p_keep_rnn']
                   if 'p_keep_rnn' in doc['config'].keys()
                   else 'na',
               doc['config']['grad_clip_norm'],
               doc['config']['lambda'] if 'lambda' in doc['config'].keys() else doc['config']['lamda']))


def print_core(docs):
    print('----\t----\t\t----\t----\t\t----')
    print('id\tmodel\t\tdb\tcollection\tdate_time')
    print('----\t----\t\t----\t----\t\t----')
    for doc in docs:
        print('%s\t%s\t%s\t%s\t\t%s' %
              (doc['id'],
               doc['model_name'],
               doc['db'],
               doc['collection'],
               doc['date_time']))


def print_globals(docs):
    print('----\t----\t----\t----\t----\t\t----\t\t----')
    print('id\tbatch\tsubset\tepochs'
          '\tmin_loss\tmax_train_acc\tmax_tune_acc')
    print('----\t----\t----\t----\t----\t\t----\t\t----')
    for doc in docs:
        print('%s\t%s\t%s\t%s\t%6.4f\t\t%5.3f%%\t\t%5.3f%%' %
              (doc['id'],
               doc['batch_size'],
               doc['subset_size'],
               doc['epochs'],
               min(doc['loss'])
                   if len(doc['loss']) > 0
                   else 0.0,
               max(doc['accuracy'])
                   if len(doc['accuracy']) > 0
                   else 0.0,
               max(doc['tuning_accuracy'])
                   if len(doc['tuning_accuracy']) > 0
                   else 0.0
               ))


def report_batch(id, iter, average_loss, average_accuracy):
    update_list(id, int(iter), 'iter')
    update_list(id, float(average_loss), 'loss')
    update_list(id, float(average_accuracy), 'accuracy')


def report_epoch(id, epoch, change_average_loss, change_average_accuracy):
    update_attr(id, 'epochs', int(epoch))
    update_list(id, float(change_average_loss), 'epoch_change_loss')
    update_list(id, float(change_average_accuracy), 'epoch_change_accuracy')


def report_tuning(id, iter, accuracy):
    update_list(id, int(iter), 'tuning_iter')
    update_list(id, float(accuracy), 'tuning_accuracy')


def runs(query={}):
    print('Training histories:')
    db = mongoi.HistoryDb()
    docs = list(db.all.find(query))
    if len(docs) == 0:
        print('No runs found.')
        return
    print_core(docs)
    print_globals(docs)
    print_config(docs)


def save(history):
    db = mongoi.HistoryDb()
    db.all.update(history)


def scale_epochs_to_iters(history, iters_key):
    return list(i * iters_per_epoch(history) for i in history[iters_key])


def scale_iters_to_epochs(history, iters_key):
    return list(i / iters_per_epoch(history) for i in history[iters_key])


def scaled_loss(history):
    return np.log(np.array(history['loss']) / history['batch_size'])


def summary_keys():
    return [
        'min_loss',
        'max_train_acc',
        'max_tune_acc'
    ]


def update_attr(id, attr, value):
    db = mongoi.HistoryDb()
    history = get_and_project(int(id), '_id')
    db.all.update_one(history['_id'], {attr: value})


def update_list(id, new_item, list_key):
    history = get_and_project(int(id), list_key)
    _list = history[list_key]
    _list.append(new_item)
    update_attr(id, list_key, _list)


def visualize(id):
    history = get_one(id)
    plt.subplot(1, 2, 1)
    loss, = plt.plot(np.array(history['iter']),
                     scaled_loss(history),
                     label='loss')
    accuracy, = plt.plot(np.array(history['iter']),
                         history['accuracy'],
                         label='accuracy')
    tuning, = plt.plot(
        np.array(scale_epochs_to_iters(history, 'tuning_iter')),
        np.array(history['tuning_accuracy']),
        label='tuning accuracy')
    plt.legend(handles=[loss, accuracy, tuning], loc=3)
    plt.subplot(1, 2, 2)
    epoch_loss, = plt.plot(
        np.arange(history['epochs']),
        np.array(history['epoch_change_loss']),
        label='change in loss')
    epoch_accuracy, = plt.plot(
        np.arange(history['epochs']),
        np.array(history['epoch_change_accuracy']),
        label='change in accuracy')
    plt.legend(handles=[epoch_loss, epoch_accuracy], loc=2)
    plt.show()


def fix_tuning_acc(id):
    history = get_one(id)
    if 'fixed_tuning' in history.keys():
        raise Exception('Already fixed tuning for this record')
    accumulated_accuracy = 0.0
    for i in range(len(history['tuning_accuracy'])):
        accuracy = history['accuracy'][i]
        history['accuracy'][i] = (accumulated_accuracy + accuracy) / (i + 1)
    history['fixed_tuning'] = 1
    save(history)
