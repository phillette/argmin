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
