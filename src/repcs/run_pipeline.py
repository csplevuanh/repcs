import argparse, csv, pathlib, tqdm
from .data import load_prompt_wnqa
from .retrievers import SparseRetriever, DenseRetriever
from .detector import kl_score, is_contaminated
from .config import KL_THRESHOLD

def main():
    p = argparse.ArgumentParser(description='Run RePCS over dataset.')
    p.add_argument('--dataset', required=True)
    p.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--output_csv', default='outputs/scores.csv')
    args = p.parse_args()

    queries, labels = load_prompt_wnqa(args.dataset)
    sparse = SparseRetriever(queries)
    dense  = DenseRetriever(queries, args.model)

    rows = []
    for q,l in tqdm.tqdm(zip(queries, labels), total=len(queries)):
        score = kl_score(sparse.retrieve(q), dense.retrieve(q))
        rows.append({'query': q, 'label': l, 'score': score,
                     'flag': int(is_contaminated(score, KL_THRESHOLD))})

    out_path = pathlib.Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        import pandas as pd
        pd.DataFrame(rows).to_csv(f, index=False)

    flagged = sum(r['flag'] for r in rows)
    print(f'Done. {flagged}/{len(rows)} queries flagged (τ={KL_THRESHOLD}).')

if __name__ == '__main__':
    main()
