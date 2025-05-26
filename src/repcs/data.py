import json
from pathlib import Path
from typing import List, Tuple

def load_prompt_wnqa(dataset_dir: str) -> Tuple[List[str], List[int]]:
    """Load Prompt‑WNQA queries and binary labels (0 clean, 1 contaminated)."""
    dataset_dir = Path(dataset_dir)
    clean = json.loads((dataset_dir / 'clean_queries.json').read_text())
    contaminated = json.loads((dataset_dir / 'contaminated_queries.json').read_text())
    queries = clean + contaminated
    labels = [0] * len(clean) + [1] * len(contaminated)
    return queries, labels
