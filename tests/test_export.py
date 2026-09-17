import json
import os
import shutil
import tempfile
import unittest
import torch

from export import export_model
from model import MiniLLM


class TestExport(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_minillm_export_")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_export_and_runtime_load(self):
        # 1. Run export_model
        export_model(output_dir=self.temp_dir)

        # 2. Check files exist
        expected_files = [
            "config.json",
            "weights.pth",
            "full_model.pth",
            "tokenizer.model",
            "model.py",
            "minillm.py",
            "__init__.py",
            "example_usage.py",
            "README.md",
        ]
        for fname in expected_files:
            fpath = os.path.join(self.temp_dir, fname)
            self.assertTrue(os.path.exists(fpath), f"Missing exported file: {fname}")

        # 3. Check config.json contents
        with open(
            os.path.join(self.temp_dir, "config.json"), "r", encoding="utf-8"
        ) as f:
            cfg = json.load(f)
        self.assertEqual(cfg["vocab_size"], 5000)
        self.assertEqual(cfg["embedding_dim"], 256)
        self.assertEqual(cfg["num_layers"], 6)

        # 4. Test loading via standalone MiniLLMRuntime
        import importlib.util

        runtime_file = os.path.join(self.temp_dir, "minillm.py")
        spec = importlib.util.spec_from_file_location("minillm", runtime_file)
        minillm_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(minillm_module)

        runtime = minillm_module.load_model(self.temp_dir, device=torch.device("cpu"))
        self.assertIsInstance(runtime, minillm_module.MiniLLMRuntime)

        # Test generation
        out = runtime.generate("Test prompt", max_new_tokens=5)
        self.assertIsInstance(out, str)
        self.assertTrue(len(out) > 0)


if __name__ == "__main__":
    unittest.main()
