import numpy as np
import matplotlib.pyplot as plt
import mongoi
import datetime
import stats
import itertools


def batch_size_info(history):
    return '%s (%s%%)' \
           % (history['batch_size'],
              round(
                  100 * history['batch_size']
                  / stats.COLLECTION_SIZE[history['db']][history['collection']],
                  2))


def compare(ids, param_to_compare):
    histories = get_many(ids)
    # loss
    lines_loss = []
    plt.subplot(2, 2, 1)
    for i in range(len(histories)):
        line_i_loss, = plt.plot(
            np.array(histories[i]['iter']),
            comparison_x(histories[i], 'loss'),
            label=comparison_label(histories[i], param_to_compare))
        lines_loss.append(line_i_loss)
    plt.legend(handles=lines_loss, loc=1)
    plt.xlabel('iteration')
    plt.ylabel('standardized loss')
    # accuracy
    lines_accuracy = []
    plt.subplot(2, 2, 2)
    for i in range(len(histories)):
        line_i_accuracy, = plt.plot(
            np.array(histories[i]['iter']),
            comparison_x(histories[i], 'accuracy'),
            label=comparison_label(histories[i], param_to_compare))
        lines_accuracy.append(line_i_accuracy)
    plt.legend(handles=lines_accuracy, loc=2)
    plt.xlabel('iteration')
    plt.ylabel('training set accuracy')
    # tuning_accuracy
    lines_tune = []
    plt.subplot(2, 2, 3)
    for i in range(len(histories)):
        line_i_tune, = plt.plot(
            np.array(histories[i]['tuning_iter']),
            comparison_x(histories[i], 'tuning_accuracy'),
            label=comparison_label(histories[i], param_to_compare))
        lines_tune.append(line_i_tune)
    plt.legend(handles=lines_tune, loc=2)
    plt.xlabel('iteration')
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


def comparison_x(history, value_to_compare):
    if value_to_compare == 'loss':
        return scaled_loss(history)
    elif value_to_compare == 'accuracy':
        return np.array(history['accuracy'])
    elif value_to_compare == 'tuning_accuracy':
        return np.array(history['tuning_accuracy'])
    else:
        raise Exception('Unexpected value: %s' % value_to_compare)


def delete(id):
    db = mongoi.HistoryDb()
    db.db.all.delete_many({'id': id})


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


def get_many(ids):
    docs = []
    db = mongoi.HistoryDb()
    for id in ids:
        try:
            docs.append(next(db.all.get(id)))
        except:
            print('History with id %s not found' % id)
    return docs


def get_one(id):
    db = mongoi.HistoryDb()
    try:
        history = next(db.all.get(id))
        return history
    except:
        print('No history with id %s found' % id)


def global_keys():
    return ['id',
            'model_name',
            'db',
            'collection',
            'date_time',
            'batch_size',
            'subset_size',
            'epochs']


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
    return history


def new_id():
    db = mongoi.HistoryDb()
    query = db.all.find_all()
    docs = query
    ids = [doc['id'] for doc in docs]
    return max(ids) + 1 if len(ids) > 0 else 1


def print_config(docs):
    print('----\t----\t----\t\t----\t----\t\t----')
    print('id\talpha\thidden_size\tp_keep\tgrad_clip_norm\tlambda')
    print('----\t----\t----\t\t----\t----\t\t----')
    for doc in docs:
        print('%s\t%s\t%s\t\t%s\t%s\t\t%s' %
              (doc['id'],
               doc['config']['learning_rate'],
               doc['config']['hidden_size'],
               doc['config']['p_keep'],
               doc['config']['grad_clip_norm'],
               doc['config']['lambda']
               ))


def print_core(docs):
    print('----\t----\t\t----\t----\t\t----')
    print('id\tmodel\t\tdb\tcollection\tdate_time')
    print('----\t----\t\t----\t----\t\t----')
    for doc in docs:
        print('%s\t%s\t%s\t%s\t\t%s' %
              (doc['id'], doc['model_name'], doc['db'],
               doc['collection'], doc['date_time']))


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


def report_batch(history, iter, average_loss, average_accuracy):
    history['iter'].append(int(iter))
    history['loss'].append(float(average_loss))
    history['accuracy'].append(float(average_accuracy))


def report_epoch(history,
                 change_average_loss,
                 change_average_accuracy):
    history['epoch_change_loss'].append(float(change_average_loss))
    history['epoch_change_accuracy'].append(float(change_average_accuracy))


def report_tuning(history, iter, accuracy):
    history['tuning_iter'].append(int(iter))
    history['tuning_accuracy'].append(float(accuracy))


def save(history):  # try this just with "update"
    db = mongoi.HistoryDb()
    db.all.update(history)


def scaled_loss(history):
    return np.log(np.array(history['loss']) / history['batch_size'])


def summary_keys():
    return [
        'min_loss',
        'max_train_acc',
        'max_tune_acc'
    ]


def view():
    print('Training histories:')
    db = mongoi.HistoryDb()
    docs = list(db.all.find_all())
    if len(docs) == 0:
        print('No runs found.')
        return
    print_core(docs)
    print_globals(docs)
    print_config(docs)


def visualize(id):
    history = get_one(id)
    plt.subplot(1, 2, 1)
    loss, = plt.plot(np.array(history['iter']),
                     scaled_loss(history),
                     label='loss')
    accuracy, = plt.plot(np.array(history['iter']),
                         history['accuracy'],
                         label='accuracy')
    tuning, = plt.plot(np.array(history['tuning_iter']),
                       np.array(history['tuning_accuracy']),
                       label='tuning accuracy')
    plt.legend(handles=[loss, accuracy, tuning], loc=1)
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


if __name__ == '__main__':
    view_runs()
