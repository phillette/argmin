class Error(Exception):
    """Base error class for this module."""
    pass


class DbNotFoundError(Error):
    """Database name not found in mongoi.COLLECTIONS.keys().

    Attributes:
      db_name: the name of the db attempted to be accessed.
    """

    def __init__(self, db_name):
        self.db_name = db_name


class FriendlyIdNotFoundError(Error):
    """Friendly id not in keys on mongo document.

    Attributes:
      db_name: the name of the db the document belongs to.
      collection_name: the name of the collection the document belongs to.
    """

    def __init__(self, db_name, collection_name):
        self.db_name = db_name
        self.collection_name = collection_name


class LabelNotFoundError(Error):
    """A label was not found in the dictionary.

    Attributes:
      label: the text of the unfound label.
    """

    def __init__(self, label):
        self.label = label


class NotDeleteError(Error):
    """An item could not be deleted from mongo.

    Attributes:
      db_name: the name of the database.
      collection_name: the name of the collection.
      _id: the mongo _id attribute value.
    """

    def __init__(self, db_name, collection_name, _id):
        self.db_name = db_name
        self.collection_name = collection_name
        self._id = _id


class PickleNotFoundError(Error):
    """The pickle was not found.

    Attributes:
      file_name: the name of the pickle file that could not be found.
    """

    def __init__(self, file_name):
        self.file_name = file_name


class UnexpectedValueError(Error):
    """The value of an argument is unexpected.

    Attributes:
      arg_name: the name of the argument.
      value: the value of the argument.
    """

    def __init__(self, arg_name, value):
        self.arg_name = arg_name
        self.value = value
