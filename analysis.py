"""Analyzing the results of model predictions."""
import pandas as pd
import batching
import util
import errors
import mongoi
import spacy


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


def determine_max_linear_adj_recursion_depth(db, collection):
    """Determine max linear adjective recursion depth in a collection.

    The results will be saved as a pickle for later reference.
    See Returns for the structure of the returned dictionary.
    The pickle will be saved as:
      pickles/lin_adj_rec_depth_db_collection.pkl

    Running this has yielded the following results:
    db    collection      premise  hypothesis
    ----  ----            ----     ----
    MNLI  train           3        2
          dev_matched     2        0
          dev_mismatched  0        2
    SNLI  train           2        3
          dev             1        2
          test            2        0

    Percentage of samples with linear adjective recursion:
    db    collection      percentage
    ----  ----            ----
    MNLI  train           10 %
          dev_matched     11 %
          dev_mismatched  8  %
    SNLI  train           8  %
          dev_matched     9  %
          dev_mismatched  9  %

    Args:
      db: the name of the db to query.
      collection: the name of the collection to query.

    Returns:
      intger, integer, dictionary, dictionary:
        the integers are: max_depth_premises, max_depth_hypothesis;
        the dictionaries take the form {pairID: max_depth}, and
        relate to premises and hypotheses, respectively.
    """
    repository = mongoi.get_repository(db, collection)
    nlp = spacy.load('en')
    count = 0
    premise_max_depth = 0
    hypothesis_max_depth = 0
    premise_depths = {}
    hypothesis_depths = {}

    def update_max_depth(previous_max, current_max, sentence_type):
        new_max = max(previous_max, current_max)
        if new_max > previous_max:
            print('New max for %s: %s' % (sentence_type, new_max))
        return new_max

    for sample in repository.all():
        count += 1
        premise = nlp(sample['sentence1'])
        hypothesis = nlp(sample['sentence2'])
        has_recursion, max_depth = has_linear_adj_recursion(premise)
        if has_recursion:
            poo


def has_linear_adj_recursion(doc):
    """Determine presence and depth of linear adjective recursion.

    Args:
      doc: SpaCy doc of the sentence to evaluate.

    Returns:
      boolean, integer: whether the linear adjective recursion is
        present in the sentence, and to what maximum depth. If the
        sentence has recursion in different places, it will return
        the maximum depth over the whole sentence.

    Raises:
      UnexpectedTypeError: if the doc argument is not a SpaCy doc.
    """
    if not isinstance(doc, spacy.tokens.doc.Doc):
        raise errors.UnexpectedTypeError(spacy.tokens.doc.Doc, type(doc))
    found = False
    last_token_is_adj = False
    current_recursion_depth = 0
    max_depth = 0
    for token in doc:
        if token.tag_ == 'JJ':
            if last_token_is_adj:
                found = True
                current_recursion_depth += 1
            else:
                last_token_is_adj = True
        else:
            last_token_is_adj = False
            max_depth = max(max_depth, current_recursion_depth)
            current_recursion_depth = 0
    return found, max_depth


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


def _update_global_max_depth(previous_max, current_max, sentence_type):
    new_max = max(previous_max, current_max)
    if new_max > previous_max:
        print('New max for %s: %s' % (sentence_type, new_max))
    return new_max


if __name__ == '__main__':
    import spacy
    nlp = spacy.load('en')
    import mongoi
    db = mongoi.SNLIDb()
    results = []
    count = 0
    premise_max_depth = 0
    hypothesis_max_depth = 0
    for sample in db.test.all():
        count += 1
        premise = nlp(sample['sentence1'])
        hypothesis = nlp(sample['sentence2'])
        has_recursion, max_depth = has_linear_adj_recursion(premise)
        if has_recursion:
            results.append(premise)
            premise_max_depth = _update_global_max_depth(
                premise_max_depth, max_depth, 'premise')
        has_recursion, max_depth = has_linear_adj_recursion(hypothesis)
        if has_recursion:
            results.append(premise)
            premise_max_depth = _update_global_max_depth(
                premise_max_depth, max_depth, 'hypothesis')
    print('%s / %s = %s%%' % (len(results), count, len(results) / count))
