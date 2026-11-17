#!/usr/bin/env bash
# Full experiment run, meant for a GPU machine (Kaggle / Colab / lab PC). Run from clinical_system/.
# Rough T4 GPU time: data prep ~1 h, each holdout model 1-3 h, 5-fold CV of one model ~8-12 h.
# If a session stops, re-run only the lines that have no runs/<model>/<run>/model_card.json yet.
set -euo pipefail

python research/download_datasets.py --yes            # ~10 GB (see DATASETS.md for licences)
python research/build_dataset.py --cache-size 512      # clean, de-duplicate, split -> data/merged/

# 1) compare the five models on the same split (train 70% / val 15% / test 15%)
for m in efficientnet_b3 resnet50 densenet121 convnext_tiny vit_b16; do
  python research/train.py --model "$m"
done

# 2) results table, graphs, ensemble of the best 3, export best model for the web app
python research/evaluate.py --runs runs --results results --export models/best

# 3) 5-fold cross-validation of the final model (put the best model's name here)
BEST=${BEST:-efficientnet_b3}
for k in 0 1 2 3 4; do
  python research/train.py --model "$BEST" --cv-fold "$k"
done
python research/evaluate.py --runs runs --results results --export models/best   # adds results_cv.csv

# 4) list possibly mislabelled images for manual review
python research/find_label_issues.py --runs runs --model "$BEST"
