import unittest
from coldnet.testing import training_tests

# > python -m unittest discover


test_cases = [
    training_tests.BalancedDataLoaderTests,

]


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for test_case in test_cases:
        tests = loader.loadTestsFromTestCase(test_case)
        suite.addTests(tests)
    return suite
