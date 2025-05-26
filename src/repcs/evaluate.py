import argparse, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--png', default='roc.png')
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    fpr, tpr, _ = roc_curve(df['label'], df['score'])
    auc = roc_auc_score(df['label'], df['score'])

    plt.figure()
    plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
    plt.plot([0,1],[0,1],'--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('RePCS ROC')
    plt.legend(loc='lower right')
    plt.savefig(args.png)
    print(f'Saved {args.png}')
if __name__ == '__main__':
    main()
