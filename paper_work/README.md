# paper_work: experiment code for the SkinLens paper

This folder is new. It does not change any of the existing project files.

| File | What it does | Where to run it |
|---|---|---|
| `train_eval.py` | Trains or evaluates the proposed model, its ablations and the baselines. It uses the same lesion-grouped split as `references/skinclass.ipynb`. For each run it saves validation and test logits to `results.npz`. | Kaggle, T4 GPU, with the HAM10000 dataset attached |
| `analyze.py` | Builds every table and figure the paper needs from the `results.npz` files: 95% CIs, AUC, calibration, conformal triage, McNemar tests and LaTeX tables. | Your laptop (numpy is enough; matplotlib adds the figures) |

## Step 0: get logits from your existing model (no training, about 10–20 minutes)

1. Make a Kaggle notebook and attach two inputs: the HAM10000 dataset and your uploaded `best_skinlens_hybrid.pt` model.
2. Upload `train_eval.py`, then run:

```bash
python train_eval.py --preset proposed --eval-only --checkpoint /kaggle/input/<your-model-folder>/best_skinlens_hybrid.pt --tta
```

3. Download `/kaggle/working/paper_runs/proposed_seed42_fold0/results.npz`.
4. On your laptop:

```bash
pip install numpy matplotlib
python analyze.py --runs "paper_runs/*/results.npz" --out paper_outputs
```

5. Open `paper_outputs/REPORT.md`. It already contains the calibration, triage and metadata-robustness results for the model you have now.

## Step 1 onward: baselines and ablations (each one is a training run)

```bash
python train_eval.py --preset effnet_only
python train_eval.py --preset swin_only
python train_eval.py --preset resnet50
python train_eval.py --preset densenet121
python train_eval.py --preset vit_b16
python train_eval.py --preset dual_image_only
python train_eval.py --preset effnet_meta
python train_eval.py --preset swin_meta
python train_eval.py --preset no_aux
python train_eval.py --preset meta_only
python train_eval.py --preset proposed --sampler none
python train_eval.py --preset proposed --class-weights none
python train_eval.py --preset proposed --seed 7
python train_eval.py --preset proposed --seed 123
python train_eval.py --preset proposed_film --meta-dropout 0.3
```

* If Kaggle stops at its 12-hour limit, run the same command again. It picks up from `last.pt`.
* Before a long run, try `--smoke` (tiny subset, 1 epoch) to make sure everything works.
* `--fold 1` to `--fold 4` give the other four cross-validation test folds. `--fold 0` is the notebook split.
* Copy each run folder back to your laptop, then run `analyze.py` once on all the `results.npz` files.

The full plan is in `../PAPER_REVIEW.md` (section 4): which runs come first, why, and the GPU-hour budget.
