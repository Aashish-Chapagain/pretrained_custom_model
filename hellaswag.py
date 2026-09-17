import os
import sys
import torch
import sentencepiece as spm
from deepeval.benchmarks import HellaSwag
from deepeval.benchmarks.tasks import HellaSwagTask
from deepeval.benchmarks.hellaswag.hellaswag import MultipleChoiceSchema
from deepeval.models.base_model import DeepEvalBaseLLM

from config.settings import TOKENIZER
from generate import load_model
from model import MiniLLM
from train import resolve_device

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class MiniLLMDeepEval(DeepEvalBaseLLM):
    def __init__(self, device: torch.device = None):
        self.device = device or resolve_device()
        tokenizer_path = f"{TOKENIZER['model_prefix']}.model"
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer model '{tokenizer_path}' not found.")
        self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
        self.model_instance = load_model(self.device)
        super().__init__(model="MiniLLM-6M")

        # Map multiple-choice letters to SentencePiece token IDs
        self.option_tokens = {
            "A": self.tokenizer.encode("A")[-1],
            "B": self.tokenizer.encode("B")[-1],
            "C": self.tokenizer.encode("C")[-1],
            "D": self.tokenizer.encode("D")[-1],
        }

    def load_model(self):
        return self.model_instance

    def generate(self, prompt: str, schema=None, **kwargs):
        tokens = self.tokenizer.encode(str(prompt))
        context = tokens[-self.model_instance.max_seq_len:]
        x = torch.tensor([context], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.model_instance(x)[0, -1]

        # Score candidate choice letters A, B, C, D from logits
        scores = {
            letter: logits[tok_id].item()
            for letter, tok_id in self.option_tokens.items()
        }
        best_letter = max(scores, key=scores.get)

        if schema == MultipleChoiceSchema:
            return MultipleChoiceSchema(answer=best_letter)
        return best_letter

    async def a_generate(self, prompt: str, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self) -> str:
        return "MiniLLM-6M"


def main():
    print("Loading MiniLLM for DeepEval HellaSwag benchmark...")
    custom_model = MiniLLMDeepEval()

    # Define benchmark with specific tasks and shots
    benchmark = HellaSwag(
        tasks=[HellaSwagTask.TRIMMING_BRANCHES_OR_HEDGES, HellaSwagTask.BATON_TWIRLING],
        n_shots=5,
    )

    print("Running evaluation...")
    benchmark.evaluate(model=custom_model)
    print(f"Overall HellaSwag Score: {benchmark.overall_score:.4f}")


if __name__ == "__main__":
    main()