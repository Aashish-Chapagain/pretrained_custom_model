import unittest

import torch

from generate import accuracy_from_logits


class InferenceValidationTests(unittest.TestCase):
    def test_accuracy_from_logits_counts_correct_predictions(self):
        logits = torch.tensor(
            [
                [2.0, 1.0, 0.2],
                [0.1, 3.0, 0.4],
                [0.2, 0.1, 2.5],
            ]
        )
        targets = torch.tensor([0, 1, 2])

        accuracy = accuracy_from_logits(logits, targets)

        self.assertEqual(accuracy, 1.0)

    def test_accuracy_from_logits_handles_partial_accuracy(self):
        logits = torch.tensor(
            [
                [0.1, 2.0, 0.3],
                [0.2, 0.1, 3.0],
            ]
        )
        targets = torch.tensor([0, 2])

        accuracy = accuracy_from_logits(logits, targets)

        self.assertEqual(accuracy, 0.5)


if __name__ == "__main__":
    unittest.main()
