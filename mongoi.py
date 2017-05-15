"""MongoDB interface."""
from pymongo import MongoClient
from bson.binary import Binary
import _pickle as cPickle
import numpy as np
import errors


# list of collections in each db
COLLECTIONS = {
    'snli': ['train', 'dev', 'test'],
    'mnli': ['train', 'dev_matched', 'dev_mismatched'],
    'carstens': ['all', 'train', 'test'],
    'history': ['all'],
    'node': ['debate_train', 'debate_test', 'wiki_train', 'wiki_test']
}


# got to be a better way to do this
def get_db(db_name):
    if db_name == 'snli':
        return SNLIDb()
    elif db_name == 'carstens':
        return CarstensDb()
    elif db_name == 'mnli':
        return MNLIDb()
    elif db_name == 'history':
        return HistoryDb()
    elif db_name == 'node':
        return NodeDb()
    else:
        raise Exception('Unexpected db_name: %s' % db_name)


def get_repository(db_name, collection_name):
    db = get_db(db_name)
    return db.repository(collection_name)


def array_to_string(array):
    """
    Converts a numpy array to a binarized string so it can be saved in MongoDB.
    :param array: the array to save
    :return: a binary string
    """
    return Binary(cPickle.dumps(array, protocol=2))


def string_to_array(array_string):
    """
    Takes a saved string in a MongoDB document that represents an array, and returns the array.
    :param array_string: the saved string to convert
    :return: a numpy array
    """
    return cPickle.loads(array_string)


class RepositoryFacade:
    """
    Facade to provide a convenient single point of access for the db connection
    and repository classes for accessing the collections.
    """
    def __init__(self,
                 server='localhost',
                 port=27017,
                 db_name=None):
        """
        :param server: MongoDB server address
        :param port: MongoDB server port
        :param db: the name of the database to access
        :param collections: a list of collections for which repositories will be initialized
        """
        self.connection = MongoClient(server, port)
        exec('self.db = self.connection.%s' % db_name)

    def repository(self, collection_name):
        return getattr(self, collection_name)


class AharoniDb(RepositoryFacade):
    def __init__(self):
        RepositoryFacade.__init__(self, db_name='aharoni')
        # ?


class CarstensDb(RepositoryFacade):
    """Repository Facade for the Carstens and Toni (2015) data set.

    Access to specific collections in this db are provided by
    Repository class attributes.

    http://www.doc.ic.ac.uk/~lc1310/

    Attributes:
      all: Repository for the collection containing all the data samples.
      train: Repository containing randomly selected 3,500 samples
        designated as training data.
      test: Repository containing randomly selected 558 samples
        designated as test data.
    """
    def __init__(self):
        RepositoryFacade.__init__(self, db_name='carstens')
        self.all = Repository('carstens', self.db.all)
        self.train = Repository('carstens', self.db.train)
        self.test = Repository('carstens', self.db.test)


class DundeeDb(RepositoryFacade):
    def __init__(self):
        RepositoryFacade.__init__(self, db_name='dundee')
        # ?


class MNLIDb(RepositoryFacade):
    """ Repository Facade for the MNLI 0.9 data set.

    Access to specific collections in this db are provided by
    Repository class attributes.

    https://www.nyu.edu/projects/bowman/multinli/

    Attributes:
      train: Repository for the train collection.
      dev_matched: Repository for the dev collection that includes
        only genres in the training set.
      dev_mismatched: Repository for teh dev collection that only
        includes genres not in the training set.
    """
    def __init__(self):
        RepositoryFacade.__init__(self, db_name='mnli')
        self.train = Repository('mnli', self.db.train)
        self.dev_matched = Repository('mnli', self.db.dev_matched)
        self.dev_mismatched = Repository('mnli', self.db.dev_mismatched)
        #self.test = Repository('mnli', self.db.test)


class NodeDb(RepositoryFacade):
    """ Repository Facade for the NoDe argument mining data set.

    Access to specific collections in this db are provided by
    Repository class attributes.

    http://www-sop.inria.fr/NoDE/

    Attributes:
      debate_train: Repository for the DebatePedia train collection.
      debate_test: Repository for the DebatePedia test collection.
      wiki_train: Repository for the Wikipedia train collection.
      wiki_test: Repository for the Wikipedia test collection
    """
    def __init__(self):
        RepositoryFacade.__init__(self, db_name='node')
        self.debate_train = Repository('node', self.db.debate_train)
        self.debate_test = Repository('node', self.db.debate_test)
        self.wiki_train = Repository('node', self.db.wiki_train)
        self.wiki_test = Repository('node', self.db.wiki_test)


class SNLIDb(RepositoryFacade):
    """ Repository Facade for the SNLI 1.0 data set.

    Access to specific collections in this db are provided by
    Repository class attributes.

    https://nlp.stanford.edu/projects/snli/

    Attributes:
      train: Repository for the train collection.
      dev: Repository for the dev collection.
      test: Repository for the test collection.
    """
    def __init__(self):
        RepositoryFacade.__init__(self, db_name='snli')
        self.train = Repository('snli', self.db.train)
        self.dev = Repository('snli', self.db.dev)
        self.test = Repository('snli', self.db.test)


class HistoryDb(RepositoryFacade):
    def __init__(self):
        RepositoryFacade.__init__(self, db_name='history')
        self.all = Repository('history', self.db.all)


class Repository:
    """Repository pattern interface for a MongoDB collection.

    The functions of this class abstract away pymongo syntax
    and provide a custom interface designed for convenience.

    Attributes:
      db_name: the name of the database the collection is in.
      collection: the Pymongo collection that can be queried against.
    """

    def __init__(self, db_name, collection_name):
        """Create a new Repository."""
        self.db_name = db_name
        self.collection = collection_name

    def add(self, doc):
        """Add a document to the collection.

        Args:
          doc: the doc to add
        """
        self.collection.insert_one(doc)

    def all(self):
        """Return all docs in the collection.

        Returns:
          PyMongo cursor (a generator) which yields
            all docs in the collection.
        """
        return self.collection.find()

    def batch(self):
        """
        Gets all documents for a batch - projecting just _id, premises, hypotheses, and label.
        :return: generator
        """
        return self.collection.find({}, {'id': 1,
                                         'premise': 1,
                                         'hypothesis': 1,
                                         'label_encoding': 1})

    def count(self):
        """
        Gets the document count in the collection.
        :return: integer
        """
        return self.collection.count()

    def delete(self, doc):
        """Deletes the doc from the collection.

        Args:
          doc: the doc to delete.
        Returns:
          Result object.
        Raises:
          NotDeletedError: if the item could not be deleted for
            whatever reason.
        """
        result = self.collection.delete_many({'_id': doc['_id']})
        if result.deleted_count != 1:
            raise errors.NotDeleteError(self.db_name,
                                        self.collection.name,
                                        doc['_id'])

    def delete_one(self, id):
        """
        Removes the document of the given id.
        Raises an exception if the delete operation fails.
        :param id: the _id value for the document
        :return: Result object
        """
        result = self.collection.delete_many({'_id': id})
        if result.deleted_count != 1:
            raise Exception('Item with _id "%s" not deleted' % id)
        return result

    def find(self, attr_dict, projection=None):
        """
        Performs a search for records based on the criteria in the attribute dictionary.
        :param attr_dict: a dictionary of search attributes and values
        :return: generator with results
        """
        return self.collection.find(attr_dict, projection)

    def find_all(self):
        # note: deprecated - all() is a nicer name, more succinct.
        """
        Get a cursor for all documents in the collection
        :return: cursor for all documents in the collection
        """
        return self.collection.find()

    def find_in(self, attr, value_list):
        """
        Queries the collection for attributes with values in the given list.
        :param attr: the attribute to search on
        :param value_list: the list of values to match
        :return: cursor for matched documents
        """
        return self.collection.find({attr: {'$in': value_list}})

    def find_one(self, attr, value):
        """
        Find a document from the attribute and value specified.
        Returns None if not found.
        :param attr: the attribute to search on
        :param value: the value of the attribute to search on
        :return: a dictionary representing the object, if found - else None
        """
        return self.collection.find_one({attr: value})

    def get(self, id):
        """
        Get a document by id.
        :param id: the friendly id for the record.
        :return: a doc if found, else None.
        """
        return self.collection.find({'id': id})

    def insert_one(self, document):
        """
        Inserts the document into the collection.
        :param document: the document to be inserted
        :return: None
        """
        self.collection.insert_one(document)

    def random_sample(self, size):
        return self.collection.aggregate([{'$sample': {'size': size}}])

    def update(self, doc):
        """Update the doc, saving attibute states into the db.

        Args:
          doc: the document to update.
        """
        self.collection.save(doc)

    def update_one(self, _id, attr_value_dict):
        """
        Update the given attribute of the document with the given id to the value given.
        Will create the attribute if it does not already exist.
        :param _id: the _id of the record to update
        :param attr_value_dict: dictionary with attribute and value names to update
        :return: None
        """
        self.collection.update_one({'_id': _id}, {'$set': attr_value_dict})


def test_array_insert_and_fetch():
    """
    Test both array_to_string() and string_to_array() at once,
    performing a save and retrieve on the test database.
    """
    print('*** Testing array_to_string() and string_to_array() ***')
    connection = MongoClient('localhost', 27017)
    db = connection.test
    collection = db.arr
    collection.remove()  # empty the collection of past test data
    A = np.array([[1, 2, 3], [4, 5, 6]])
    collection.insert({'_id': 't1', 'array': array_to_string(A)})
    doc = next(collection.find({'_id': 't1'}))
    B = string_to_array(doc['array'])
    result = np.array_equal(A, B)
    print('Test passed: %s' % result)
    assert result
