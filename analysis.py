"""Analyzing the results of model predictions."""
import stats
import pandas as pd
import numpy as np
import batching
import util
import mongoi


"""
What kinds of things do I want to do here?
1) Be able to examine random or particular predictions one by one in detail.
2) Be able to add annotations like what I have for MNLI or custom ones.
3) Be able to count statistics along the lines of the annotations.
4) Need a "difference" calculation between two result sets:
    - which results moved from wrong to right and vice versa.

* We start by getting the results into a dictionary {id: [values]}
* We can make that a dataframe - set up a function. For queries and stats.
"""


def get_results(model, db, collection, sess, to_pandafy=True):
    """Run prediction ops and obtain results.

    The model passed is already initialized and the
    required checkpoint is already loaded.

    Args:
      model: a pre-initialized, checkpoint loaded model.
      db: the name of the db to analyze.
      collection: the name of the collection to analyze.
      sess: tf.Session() object.
      to_pandafy: if True converts results to a pandas DataFrame.
    Returns:
      Dictionary of structure: {id: [predicted_label, confidence, correct]}
      if pandify is False, otherwise a DataFrame with id at the index.
    """
    ids = []
    values = []
    batch_gen = batching.get_batch_gen(db, collection, 32)
    num_iters = batching.num_iters(db, collection, 32)
    for _ in num_iters:
        batch = next(batch_gen)
        predicted_labels, confidences, correct_predictions = \
            sess.run([model.predicted_labels,
                      model.confidences,
                      model.correct_predictions],
                     feed_dict=util.feed_dict(model, batch, False))
        ids += batch.ids
        values += list(zip(
            predicted_labels,
            confidences,
            correct_predictions))
    results = dict(zip(ids, values))
    if to_pandafy:
        results = pandafy(results, db, collection)
    return results


# GET THE SENTENCES IN AT PREVIOUS STEP OR IN PANDAFY?


def pandafy(results, db, collection):
    """Create a pandas DataFrame from prediction results dictionary.

    Args:
      results: Dictionary of structure
        {id: [predicted_label, confidence, correct]}.
      db: the name of the db the results come from.
      collection: the name of the collection the results come from.
    Returns:
      Pandas dictionary with id as index, and three columns:
        [label, confidence, correct].
    """
    return pd.DataFrame(
        data=list(results.values()),
        index=results.keys(),
        columns=['label', 'confidence', 'correct'])
