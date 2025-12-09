from .base_evaluator import BaseEvaluator
from datasets import load_dataset


class TwoWikiMQAEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.max_tokens = 32
        self.truncate_input = True
        self.multiple_answers = True
        self.data = self.load_data()

    def load_data(self):
        dataset = load_dataset('Xnhyacinth/LongBench', split='test', name='2wikimqa')
        dataset = dataset.map(lambda x: {
            "prompt_A": x["context"], 
            "prompt_B": x["question"], 
        })
        return dataset
