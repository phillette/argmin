"""For pre-processing the MNLI data set."""
import mongoi


"""
Steps for MNLI preprocessing as follows.
1) mongoimport MNLI data
2) run process_data.pre_process('mnli')
3) run import_snli_samples() below
"""


def import_snli_samples():
    """Import 77,350 SNLI data samples into MNLI."""
    print('Importing 77,350 SNLI samples into MNLI...')

    db = mongoi.MNLIDb()
    all_snli = mongoi.get_repository('snli', 'train').all()
    for _ in range(77350):
        doc = next(all_snli)
        doc['genre'] = 'snli'
        db.train.insert_one(doc)

    print('Completed successfully.')
