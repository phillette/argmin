import mongoi


# With MNLI already established, run the following commands in mongo:
# > use mnli
# > db.copyDatabase("mnli", "xnli", "127.0.0.1")
# > then all we need to do is copy 77,350 samples from SNLI
# With SNLI already processed, this means just iterate and add.

def import_snli():
    print('Importing 77,350 SNLI samples into XNLI...')
    db = mongoi.XNLIDb()
    all_snli = mongoi.get_repository('snli', 'train').find_all()
    for _ in range(77350):
        doc = next(all_snli)
        doc['genre'] = 'snli'
        db.train.insert_one(doc)
    print('Completed successfully.')


def fill_empty_genres():
    pass


if __name__ == '__main__':
    import_snli()
