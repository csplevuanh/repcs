# RePCS: Retrieval‑Path Contamination Scoring

Reference implementation of **RePCS** – a lightweight detector that diagnoses data memorisation in Retrieval‑Augmented Generation (RAG) pipelines.  
Implements the methodology described in *RePCS: Diagnosing Data Memorisation in LLM‑Powered Retrieval‑Augmented Generation* (ICCPS 2025).

## Key features
* Dual‑path retrieval: BM25 lexical + dense encoder (FAISS)
* KL‑divergence contamination score with PAC‑bound threshold calibration
* Prompt‑WNQA benchmark loader, end‑to‑end pipeline & visualisation
* Minimal dependencies, GPU optional, resource‑constrained friendly
* CI‑ready with pytest and GitHub Actions

## Quick start
```bash
pip install -r requirements.txt
export REPCS_DATA_DIR=/absolute/path/to/data

# Calibrate detector on clean queries
python -m repcs.calibrate --clean $REPCS_DATA_DIR/prompt_wnqa/clean_queries.json

# Run detection and evaluation
python -m repcs.run_pipeline --dataset $REPCS_DATA_DIR/prompt_wnqa
python -m repcs.evaluate --csv outputs/scores.csv --png roc.png
```

For detailed usage see the docs in `src/repcs`.
