import os
import shutil
import tempfile
import unittest
import numpy as np
import torch
import torch.nn as nn

from config.settings import MODEL
from finetune import SFTDataset, load_base_model
from model import MiniLLM


class TestFinetune(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_sft_")
        self.npz_path = os.path.join(self.temp_dir, "test_sft.npz")

        # Create dummy SFT dataset
        num_samples = 10
        seq_len = MODEL["max_seq_len"]

        # Prompt length: 20 tokens, Response length: 30 tokens, remainder pad
        input_ids = np.zeros((num_samples, seq_len), dtype=np.int32)
        labels = np.full((num_samples, seq_len), -100, dtype=np.int32)

        for i in range(num_samples):
            input_ids[i, :50] = np.random.randint(10, MODEL["vocab_size"], size=50)
            # Mask first 20 tokens (prompt)
            labels[i, 20:50] = input_ids[i, 20:50]

        np.savez_compressed(self.npz_path, input_ids=input_ids, labels=labels)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_sft_dataset_loading(self):
        ds = SFTDataset(self.npz_path)
        self.assertEqual(len(ds), 10)
        x, y = ds[0]
        self.assertEqual(x.shape, (MODEL["max_seq_len"],))
        self.assertEqual(y.shape, (MODEL["max_seq_len"],))
        self.assertTrue((y[:20] == -100).all())
        self.assertTrue((y[50:] == -100).all())
        self.assertTrue((y[20:50] != -100).all())

    def test_loss_masking_behavior(self):
        ds = SFTDataset(self.npz_path)
        x, y = ds[0]
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

        model = MiniLLM()
        logits = model(x)

        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        loss = criterion(logits.view(-1, MODEL["vocab_size"]), y.view(-1))
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
        self.assertGreater(loss.item(), 0.0)

        # Backward should calculate gradients
        loss.backward()
        has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.parameters())
        self.assertTrue(has_grad)


if __name__ == "__main__":
    unittest.main()
