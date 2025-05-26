from typing import List, Tuple
import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
from .config import KL_THRESHOLD

def kl_score(sparse: List[Tuple[str,float]], dense: List[Tuple[str,float]]) -> float:
    ps = softmax([s for _,s in sparse])
    pd = softmax([s for _,s in dense])
    return float(entropy(ps, pd))

def is_contaminated(score: float, threshold: float = KL_THRESHOLD) -> bool:
    return score < threshold
