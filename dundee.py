import mongoi
import process_data


"""
Look again at papers that report results on this data set for baselines
and how to do train-test split, etc.

For my purposes, I may need to process it into sentence pairs with labels.
That may involve creating a bunch of "None" pairs that aren't in the json.

Nice to see the "concatenated" file with all the pure text.
It could be used for word vector training potentially.
But I would want to remove the urls it would seem.

* Moens 2007 (found from 'Argumentation Mining') says 10-fold CV.
  They also added 1,072 non-argumentative sentences from the same sources
  to balance out the corpus.
* My methodology is a little different - I don't need to add the non-arg
  sentences.
* 10-fold CV might be time consuming...
"""


def blah():
    pass
