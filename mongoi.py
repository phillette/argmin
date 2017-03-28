from pymongo import MongoClient
from bson.binary import Binary
import _pickle as cPickle
import numpy as np


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
    def __init__(self, server='localhost', port=27017, db=None, collections=[]):
        """
        :param server: MongoDB server address
        :param port: MongoDB server port
        :param db: the name of the database to access
        :param collections: a list of collections for which repositories will be initialized
        """
        self.connection = MongoClient(server, port)
        exec('self.db = self.connection.%s' % db)
        for collection in collections:
            exec('self.%s = Repository(self.db.%s)' % (collection, collection))

    def repository(self, collection):
        return getattr(self, collection)


class Carstens(RepositoryFacade):
    """Repository Facade for the Carstens and Toni (2015) data set. """
    def __init__(self):
        RepositoryFacade.__init__(self, 'localhost', 27017, 'carstens', ['all', 'train', 'test'])


class SNLIDb(RepositoryFacade):
    """ Repository Facade for the SNLI 1.0 data set. """
    def __init__(self):
        RepositoryFacade.__init__(self, 'localhost', 27017, 'snlidb', ['train', 'dev', 'test'])


class Repository:
    """
    Interface for a MongoDB collection.
    The functions of this class abstract away pymongo syntax and provide a custom interface.
    The pymongo interface is still available on the Facade class in the db variable.
    """
    def __init__(self, collection):
        self.collection = collection

    def find_all(self):
        """
        Get a cursor for all documents in the collection
        :return: cursor for all documents in the collection
        """
        return self.collection.find()

    def find_one(self, attr, value):
        """
        Find a document from the attribute and value specified.
        Returns None if not found.
        :param attr: the attribute to search on
        :param value: the value of the attribute to search on
        :return: a dictionary representing the object, if found - else None
        """
        return self.collection.find_one({attr: value})

    def insert_one(self, document):
        """
        Inserts the document into the collection.
        :param document: the document to be inserted
        :return: None
        """
        self.collection.insert_one(document)

    def random_sample(self, size):
        return self.collection.aggregate([{'$sample': {'size': size}}])

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


if __name__ == '__main__':
    test_array_insert_and_fetch()
