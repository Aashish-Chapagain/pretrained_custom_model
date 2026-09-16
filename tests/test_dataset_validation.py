import os
import tempfile
import unittest

import numpy as np

from datasetclass import LLMDataset


class DatasetValidationTests(unittest.TestCase):
    def test_rejects_tokens_outside_vocab_range(self):
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as handle:
            np.save(handle.name, np.array([0, 1, 2, 3, 4, 5, 6, 7, 5000], dtype=np.int64))
            path = handle.name

        try:
            with self.assertRaisesRegex(ValueError, "out of range"):
                LLMDataset(path, sequence_length=4, stride=2)
        finally:
            os.remove(path)

    def test_rejects_too_short_corpus(self):
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as handle:
            np.save(handle.name, np.array([1, 2, 3], dtype=np.int64))
            path = handle.name

        try:
            with self.assertRaisesRegex(ValueError, "too short"):
                LLMDataset(path, sequence_length=8, stride=2)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
