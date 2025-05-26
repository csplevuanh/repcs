import argparse, json, numpy as np, tqdm
from .retrievers import SparseRetriever, DenseRetriever
from .detector import kl_score

def main():
    p = argparse.ArgumentParser(description='Calibrate KL threshold on clean queries.')
    p.add_argument('--clean', required=True, help='Path to clean_queries.json')
    p.add_argument('--percentile', type=float, default=95, help='Desired percentile (e.g., 95 for 5%% FPR)')
    p.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2')
    args = p.parse_args()

    clean = json.load(open(args.clean))
    sparse = SparseRetriever(clean)
    dense  = DenseRetriever(clean, args.model)

    scores = [kl_score(sparse.retrieve(q), dense.retrieve(q)) for q in tqdm.tqdm(clean)]
    tau = float(np.percentile(scores, args.percentile))
    print(f'Calibrated threshold τ = {tau:.5f}')
    print('Export via:  export REPCS_KL_THRESHOLD={:.5f}'.format(tau))

if __name__ == '__main__':
    main()
