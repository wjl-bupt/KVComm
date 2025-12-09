from .base_evaluator import BaseEvaluator
from datasets import load_dataset

def construct_support(item):
    results = []
    for paragraph in item["paragraphs"]:
        if paragraph["is_supporting"]:
            results.append(paragraph["paragraph_text"])
    return results

def construct_answers(item):
    return [item["answer"]] + item["answer_aliases"]

class MuSiQueEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.max_tokens = 48
        self.truncate_input = True
        self.multiple_answers = True
        self.n_samples = 500
        self.data = self.load_data()
        
    def load_data(self):
        dataset = load_dataset("dgslibisey/MuSiQue", split="validation")
        dataset = self.random_sample(dataset)
        dataset = dataset.map(lambda x: {"support": construct_support(x)})
        dataset = dataset.map(lambda x: {"prompt_A": "\n".join(x["support"])})
        dataset = dataset.map(lambda x: {"prompt_B": x["question"]})
        dataset = dataset.map(lambda x: {"answers": construct_answers(x)})
        return dataset

