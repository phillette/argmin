import unittest

from argmin.testing import prep_essays_tests

# > python -m unittest discover


test_cases = [
    prep_essays_tests.PrepEssaysTests
]


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for test_case in test_cases:
        tests = loader.loadTestsFromTestCase(test_case)
        suite.addTests(tests)
    return suite
