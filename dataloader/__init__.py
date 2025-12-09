from .countries import CountriesEvaluator
from .tipsheets import TipsheetsEvaluator
from .hotpotqa import HotpotQAEvaluator
from .narrativeqa import NarrativeQAEvaluator
from .qasper import QaSperEvaluator
from .musique import MuSiQueEvaluator
from .multifieldqa_en import MultiFieldQAEnEvaluator
from .twowikimqa import TwoWikiMQAEvaluator
from .tmath import TMathEvaluator

def get_evaluator(test_task: str):
    if test_task == "countries":
        return CountriesEvaluator()
    elif test_task == "tipsheets":
        return TipsheetsEvaluator()
    elif test_task == "hotpotqa":
        return HotpotQAEvaluator()
    elif test_task == "narrativeqa":
        return NarrativeQAEvaluator()
    elif test_task == "qasper":
        return QaSperEvaluator()
    elif test_task == "musique":
        return MuSiQueEvaluator()
    elif test_task == "multifieldqa_en":
        return MultiFieldQAEnEvaluator()
    elif test_task == "twowikimqa":
        return TwoWikiMQAEvaluator()
    elif test_task == "tmath":
        return TMathEvaluator()
    else:
        raise ValueError(f"Unsupported task name: {test_task}")