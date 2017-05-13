"""For pre-processing the MNLI data set."""
import mongoi


# With MNLI already established, run the following commands in mongo:
# > use mnli
# > then all we need to do is copy 77,350 samples from SNLI
# With SNLI already processed, this means just iterate and add,
# remembering to add the genre attribute.

def import_snli():
    """Import 77,350 SNLI data samples into MNLI."""
    print('Importing 77,350 SNLI samples into MNLI...')

    db = mongoi.MNLIDb()
    all_snli = mongoi.get_repository('snli', 'train').all()
    for _ in range(77350):
        doc = next(all_snli)
        doc['genre'] = 'snli'
        db.train.insert_one(doc)

    print('Completed successfully.')
