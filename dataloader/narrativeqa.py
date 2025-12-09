from .base_evaluator import BaseEvaluator
from datasets import load_dataset

class NarrativeQAEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.max_tokens = 128
        self.truncate_input = True
        self.multiple_answers = True
        self.n_samples = 500
        self.data = self.load_data()

    def load_data(self):
        dataset = load_dataset("deepmind/narrativeqa", split="validation")
        dataset = self.random_sample(dataset)
        dataset = dataset.map(lambda x: {
            "prompt_A": x["document"]["text"],
            "prompt_B": x["question"]["text"], 
            "answers": [ans["text"] for ans in x["answers"]]
        })
        return dataset
