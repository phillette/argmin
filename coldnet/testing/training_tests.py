"""Test cases for coldnet/training.py."""
import unittest
from coldnet import training


class BalancedDataLoaderTests(unittest.TestCase):
    def test_init(self):
        data = [['a', 'b', 'c', 'd'], [1, 2, 3]]
        loader = training.BalancedDataLoader(data, 2)
        self.assertEqual(2, loader.n_streams)
        self.assertEqual(2, loader.batch_size)
        self.assertEqual(3, loader.min_length)
        self.assertEqual(6, loader.size)
        self.assertEqual(3, loader.batches_per_epoch)

    def test_epoch_composition(self):
        data = [['a', 'b', 'c', 'd'], [1, 2, 3]]
        loader = training.BalancedDataLoader(data, 2)
        self.assertEqual(3, len([d for d in loader.data if isinstance(d, str)]))
        self.assertEqual(3, len([d for d in loader.data if isinstance(d, int)]))

    def test_epoch_change_logic(self):
        data = [['a', 'b', 'c', 'd'], [1, 2, 3]]
        loader = training.BalancedDataLoader(data, 2)
        # one epoch's worth of data has 6 samples. That's three batches / epoch.
        for i in range(3):
            _ = loader.next_batch()
            if i < 2:
                self.assertEqual(i + 1, loader.batch_number)
            else:
                self.assertEqual(0, loader.batch_number)
