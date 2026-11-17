#!/usr/bin/env bash
# =============================================================================
# Run ONE stage of the SkinLens research pipeline on Kaggle (or any Linux box).
#
#   bash research/kaggle_stage.sh <stage>
#
# Each stage fits in one Kaggle session (max ~12 h, ~20 GB in /kaggle/working).
# Every stage can be run again: finished work is skipped, unfinished training resumes
# from the last epoch checkpoint. See KAGGLE.md for the notebook-by-notebook guide.
#
# Environment variables (all optional):
#   OUT_DIR     where this session saves results          default /kaggle/working/out
#   DATA_DIR    processed data (folder with manifest.csv)  default: auto-detect in OUT_DIR or /kaggle/input
#   MODELS_DIR  outputs of earlier sessions (trained models, CV folds); several folders may be
#               given separated by commas                  default /kaggle/input (if it exists)
#   RAW_DIR     raw downloads (data stage)                 default /tmp/raw  (not counted in the 20 GB)
#   IMG_SIZE    short side of saved images (data stage)    default 384
#   MAX_DATA_GB size limit for data_processed              default 6
#   MAX_HOURS   stop cleanly before this many hours        default 11 (leaves time to save the output)
#   BEST_MODEL  model for the cv stage                     default: best on validation (from ensemble)
#   EPOCHS, BATCH_SIZE, WORKERS                            training overrides (defaults 30, 32, 4)
#   EPOCH_HINT  expected seconds per epoch for the time-limit check (default: from earlier runs)
#   INSTALL=0   skip "pip install -r requirements-kaggle.txt"
#   SKIP_DOWNLOAD=1  data stage uses files already in RAW_DIR
#   SMOKE=1     tiny, fast run for testing (CPU is fine)
#   PY          python executable                          default python
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
PY="${PY:-python}"
OUT_DIR="${OUT_DIR:-/kaggle/working/out}"
RAW_DIR="${RAW_DIR:-/tmp/raw}"
DATA_DIR="${DATA_DIR:-}"
if [ -z "${MODELS_DIR+x}" ]; then
  if [ -d /kaggle/input ]; then MODELS_DIR=/kaggle/input; else MODELS_DIR=""; fi
fi
IMG_SIZE="${IMG_SIZE:-384}"
MAX_DATA_GB="${MAX_DATA_GB:-6}"
MAX_HOURS="${MAX_HOURS:-11}"
INSTALL="${INSTALL:-1}"
SMOKE="${SMOKE:-0}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

MODEL_STAGES="train_efficientnet_b3 train_resnet50 train_densenet121 train_convnext train_vit"
EXPERIMENT_STAGES="audit leak_test focus lesion_focus_train calibrate external"
ALL_STAGES="data $MODEL_STAGES ensemble cv report $EXPERIMENT_STAGES"

usage() {
  cat <<EOF
Usage: bash research/kaggle_stage.sh <stage>

Valid stages (run them in this order, one Kaggle session each):
  data                    download, clean, de-duplicate and split the datasets;
                          saves resized images + CSVs to \$OUT_DIR/data_processed
  train_efficientnet_b3   train EfficientNet-B3
  train_resnet50          train ResNet-50
  train_densenet121       train DenseNet-121
  train_convnext          train ConvNeXt-Tiny
  train_vit               train ViT-B/16
  ensemble                ensemble of the best models + export for the web app
  cv                      5-fold cross-validation of the best model (resumable across sessions)
  report                  results.csv, per-class/CV/external tables and all graphs

Paper experiments (see EXPERIMENTS.md):
  audit                   why label conflicts / lesion grouping look the way they do (re-downloads
                          the raw data; the data stage writes the same audit automatically)
  leak_test               ResNet-50 on a naive random split vs the clean split -> leakage inflation
  focus                   where the best model looks: Grad-CAM energy inside public lesion masks
  lesion_focus_train      retrain the best model to look at the lesion (mask-guided loss +
                          background augmentation), then compare accuracy and focus before/after
  calibrate               temperature scaling: ECE before/after, reliability diagrams,
                          calibrated web-app model
  external                every model + ensemble on the external ISIC 2018 test set

Paths: OUT_DIR=$OUT_DIR  DATA_DIR=${DATA_DIR:-<auto>}  MODELS_DIR=${MODELS_DIR:-<none>}
EOF
  exit 2
}

[ $# -eq 1 ] || usage
STAGE="$1"
case " $ALL_STAGES " in *" $STAGE "*) ;; *) echo "Unknown stage: '$STAGE'"; echo; usage ;; esac

cd "$ROOT"
mkdir -p "$OUT_DIR"
NOW=$(date +%s)
DEADLINE=$(awk -v n="$NOW" -v h="$MAX_HOURS" 'BEGIN { printf "%d", n + h * 3600 }')
IFS=',' read -r -a MROOTS <<< "${MODELS_DIR}"
ROOTS=()
for r in "${MROOTS[@]:-}"; do [ -n "$r" ] && ROOTS+=("$r"); done
ROOTS+=("$OUT_DIR")   # last = this session's results win

log() { echo "[$(date '+%H:%M:%S')] $*"; }
status() { echo "$(date '+%Y-%m-%d %H:%M') $STAGE: $*" >> "$OUT_DIR/STATUS.txt"; log "STATUS: $*"; }

install_reqs() {
  [ "$INSTALL" = "1" ] || return 0
  log "Installing requirements-kaggle.txt (torch, torchvision, numpy, pandas are kept as installed)"
  local c; c="$(mktemp)"
  "$PY" -m pip freeze | grep -iE '^(torch|torchvision|torchaudio|numpy|pandas)==' > "$c" || true
  "$PY" -m pip install -q -r "$ROOT/requirements-kaggle.txt" -c "$c"
  rm -f "$c"
}

resolve_data() {
  DATA_DIR="$("$PY" research/kaggle_tools.py find-data --data-dir "$DATA_DIR" --out-dir "$OUT_DIR")"
  # images are stored as one .tar (Kaggle output file limit); extract to local disk once per session
  DATA_DIR="$("$PY" research/kaggle_tools.py unpack --src "$DATA_DIR" --dst "${LOCAL_DATA:-/tmp/skinlens_data}")"
  log "Using processed data: $DATA_DIR"
}

# Kaggle saves/shows only ~500 output files: pack every image folder (and any folder of >= 400 files) into one .tar
pack_output() {
  local n
  n="$("$PY" research/kaggle_tools.py pack --dir "$OUT_DIR" --min-files 400)"
  log "Output folder holds $n files"
}

best_model() {
  BEST="${BEST_MODEL:-$("$PY" research/kaggle_tools.py best-model --roots "${ROOTS[@]}")}"
  log "Best model: $BEST"
}

get_masks() {
  local extra=()
  if [ -n "${MASK_DIRS:-}" ]; then extra=(--mask-dirs "$MASK_DIRS"); fi
  MASKS="$("$PY" research/experiments.py masks --data "$DATA_DIR" --roots "${ROOTS[@]}" \
           --out "$OUT_DIR/masks" --raw "${MASK_RAW_DIR:-/tmp/raw_masks}" --size "$IMG_SIZE" \
           "${extra[@]}" | tail -1)"
  log "Lesion masks: $MASKS"
}

set_train_args() {
  TARGS=(--manifest "${T_MANIFEST:-$DATA_DIR/manifest.csv}" --out "${T_OUT:-$OUT_DIR/runs}" --deadline "$DEADLINE")
  if [ -n "${T_VARIANT:-}" ]; then TARGS+=(--variant "$T_VARIANT"); fi
  if [ -n "${EPOCHS:-}" ]; then TARGS+=(--epochs "$EPOCHS"); fi
  if [ -n "${BATCH_SIZE:-}" ]; then TARGS+=(--batch-size "$BATCH_SIZE"); fi
  if [ -n "${WORKERS:-}" ]; then TARGS+=(--workers "$WORKERS"); fi
  if [ "$SMOKE" = "1" ]; then TARGS+=(--smoke --workers 0 --batch-size 8); fi
}

# train_one <model> <tag> [extra train.py args]; returns 0 = finished, 3 = paused (time limit)
train_one() {
  local model="$1" tag="$2"; shift 2
  local state
  state="$("$PY" research/kaggle_tools.py seed-run --model "$model" --tag "$tag" \
           --models-dir "${ROOTS[@]:0:${#ROOTS[@]}-1}" --out-dir "${T_OUT:-$OUT_DIR/runs}" \
           --variant "${T_VARIANT:-}" ${LIGHT:+--light} | head -1)"
  if [ "$state" = "done" ]; then log "$model/$tag already finished - skipping"; return 0; fi
  [ "$state" = "resume" ] && log "$model/$tag: resuming from the last saved epoch"
  local hint
  hint="${EPOCH_HINT:-$("$PY" research/kaggle_tools.py epoch-hint --roots "${ROOTS[@]}" --model "$model")}"
  set_train_args
  set +e
  "$PY" research/train.py --model "$model" "${TARGS[@]}" --epoch-hint "$hint" "$@"
  local rc=$?
  set -e
  if [ $rc -eq 3 ]; then return 3; fi
  if [ $rc -ne 0 ]; then status "FAILED ($model/$tag, exit code $rc)"; exit $rc; fi
  return 0
}

paused_msg() {
  status "PAUSED - time limit reached, progress saved in $OUT_DIR/runs"
  cat <<EOF

==============================================================================
 Time limit reached ($MAX_HOURS h). Progress is saved (checkpoint after every epoch).
 To continue: save this notebook's output (Save Version), add that output as an
 input to a new session, and run the SAME stage again:
     bash research/kaggle_stage.sh $STAGE
==============================================================================
EOF
}

case "$STAGE" in
  data)
    install_reqs
    if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
      "$PY" research/download_datasets.py --yes --delete-zip --root "$RAW_DIR"
    fi
    rm -rf "$OUT_DIR/data_processed.tmp"
    "$PY" research/build_dataset.py --raw "$RAW_DIR" --out "$OUT_DIR/data_processed.tmp" \
        --cache-size "$IMG_SIZE" --workers "${WORKERS:-8}"
    "$PY" research/kaggle_tools.py pack --dir "$OUT_DIR/data_processed.tmp" --min-files 400 > /dev/null
    rm -rf "$OUT_DIR/data_processed"
    mv "$OUT_DIR/data_processed.tmp" "$OUT_DIR/data_processed"
    size=$("$PY" research/kaggle_tools.py size --dir "$OUT_DIR/data_processed" --max-gb "$MAX_DATA_GB")
    pack_output
    status "finished - data_processed is ${size} GB (audit: data_processed/audit/audit_report.md)"
    ;;

  train_*)
    case "$STAGE" in
      train_efficientnet_b3) MODEL=efficientnet_b3 ;;
      train_resnet50)        MODEL=resnet50 ;;
      train_densenet121)     MODEL=densenet121 ;;
      train_convnext)        MODEL=convnext_tiny ;;
      train_vit)             MODEL=vit_b16 ;;
    esac
    install_reqs
    resolve_data
    if train_one "$MODEL" holdout; then status "finished - $OUT_DIR/runs/$MODEL/holdout"
    else paused_msg; fi
    ;;

  ensemble)
    install_reqs
    "$PY" research/evaluate.py --runs "${ROOTS[@]}" --results "$OUT_DIR/ensemble" \
        --export "$OUT_DIR/models/best" --ensemble-size "${ENSEMBLE_SIZE:-3}"
    status "finished - ensemble results in $OUT_DIR/ensemble, web-app model in $OUT_DIR/models/best"
    ;;

  cv)
    install_reqs
    resolve_data
    BEST="${BEST_MODEL:-$("$PY" research/kaggle_tools.py best-model --roots "${ROOTS[@]}")}"
    log "Cross-validating: $BEST"
    LIGHT=1   # copy finished folds from earlier sessions without their weights (saves space)
    paused=0
    for k in 0 1 2 3 4; do
      if [ "$(date +%s)" -ge "$DEADLINE" ]; then paused=1; break; fi
      if ! train_one "$BEST" "cv$k" --cv-fold "$k"; then paused=1; break; fi
      status "fold $k of $BEST done"
    done
    if [ $paused -eq 1 ]; then paused_msg; exit 0; fi
    "$PY" research/find_label_issues.py --runs "${ROOTS[@]}" --model "$BEST" \
        --manifest "$DATA_DIR/manifest.csv" --out "$OUT_DIR/label_review" || log "label-issue check skipped"
    status "finished - all 5 folds of $BEST"
    ;;

  report)
    install_reqs
    "$PY" research/evaluate.py --runs "${ROOTS[@]}" --results "$OUT_DIR/results" --export ""
    if DATA_DIR="$("$PY" research/kaggle_tools.py find-data --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" 2>/dev/null)"; then
      for f in cleaning_report.md class_distribution.png; do
        [ -f "$DATA_DIR/$f" ] && cp "$DATA_DIR/$f" "$OUT_DIR/results/"
      done
    fi
    pack_output
    status "finished - tables and graphs in $OUT_DIR/results"
    ls -1 "$OUT_DIR/results"
    ;;

  audit)
    install_reqs
    if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
      "$PY" research/download_datasets.py --yes --delete-zip --root "$RAW_DIR" \
          --only isic2019_images isic2019_gt isic2019_meta ham_lesion_groups isic2018_test_images isic2018_test_gt
    fi
    rm -rf "$OUT_DIR/audit_work"
    "$PY" research/build_dataset.py --raw "$RAW_DIR" --out "$OUT_DIR/audit_work" --audit-only \
        --workers "${WORKERS:-8}" > /dev/null
    rm -rf "$OUT_DIR/audit"
    mv "$OUT_DIR/audit_work/audit" "$OUT_DIR/audit"
    rm -rf "$OUT_DIR/audit_work"
    cat "$OUT_DIR/audit/audit_report.md"
    pack_output
    status "finished - $OUT_DIR/audit/audit_report.md"
    ;;

  leak_test)
    install_reqs
    resolve_data
    if [ ! -f "$DATA_DIR/manifest_naive.csv" ]; then
      echo "manifest_naive.csv not found in $DATA_DIR - re-run the 'data' stage (the current version writes it)."
      exit 1
    fi
    # (a) naive random image-level split: no de-duplication, no lesion grouping
    if ! T_OUT="$OUT_DIR/leak_test/naive_runs" T_VARIANT=naive_split T_MANIFEST="$DATA_DIR/manifest_naive.csv" \
         train_one resnet50 holdout; then paused_msg; exit 0; fi
    # (b) clean leakage-free split, identical settings (re-uses a finished train_resnet50 run if attached)
    if ! LIGHT=1 train_one resnet50 holdout; then paused_msg; exit 0; fi
    NAIVE="$("$PY" research/experiments.py find-run --roots "${ROOTS[@]}" --model resnet50 --variant naive_split)"
    CLEAN="$("$PY" research/experiments.py find-run --roots "${ROOTS[@]}" --model resnet50)"
    "$PY" research/experiments.py leak --naive "$NAIVE" --clean "$CLEAN" \
        --naive-manifest "$DATA_DIR/manifest_naive.csv" --out "$OUT_DIR/leak_test"
    pack_output
    status "finished - $OUT_DIR/leak_test/leak_report.md"
    ;;

  focus)
    install_reqs
    resolve_data
    best_model
    get_masks
    RUN="$("$PY" research/experiments.py find-run --roots "${ROOTS[@]}" --model "$BEST")"
    FARGS=()
    if [ -n "${FOCUS_MAX_IMAGES:-}" ]; then FARGS=(--max-images "$FOCUS_MAX_IMAGES"); fi
    "$PY" research/experiments.py focus --run "$RUN" --data "$DATA_DIR" --masks "$MASKS" \
        --out "$OUT_DIR/focus/$BEST" "${FARGS[@]}"
    pack_output
    status "finished - $OUT_DIR/focus/$BEST/focus_report.md"
    ;;

  lesion_focus_train)
    install_reqs
    resolve_data
    best_model
    get_masks
    if ! T_OUT="$OUT_DIR/lesion_focus/runs" T_VARIANT=lesion_focus train_one "$BEST" holdout \
         --masks "$MASKS" --attn-weight "${ATTN_WEIGHT:-0.5}" --bg-aug "${BG_AUG:-0.5}"; then
      paused_msg; exit 0
    fi
    BEFORE="$("$PY" research/experiments.py find-run --roots "${ROOTS[@]}" --model "$BEST")"
    AFTER="$("$PY" research/experiments.py find-run --roots "${ROOTS[@]}" --model "$BEST" --variant lesion_focus)"
    FARGS=()
    if [ -n "${FOCUS_MAX_IMAGES:-}" ]; then FARGS=(--max-images "$FOCUS_MAX_IMAGES"); fi
    "$PY" research/experiments.py focus --run "$BEFORE" --data "$DATA_DIR" --masks "$MASKS" \
        --out "$OUT_DIR/lesion_focus/focus_before" "${FARGS[@]}" > /dev/null
    "$PY" research/experiments.py focus --run "$AFTER" --data "$DATA_DIR" --masks "$MASKS" \
        --out "$OUT_DIR/lesion_focus/focus_after" "${FARGS[@]}" > /dev/null
    "$PY" research/experiments.py compare-focus --before-run "$BEFORE" --after-run "$AFTER" \
        --before-focus "$OUT_DIR/lesion_focus/focus_before" --after-focus "$OUT_DIR/lesion_focus/focus_after" \
        --out "$OUT_DIR/lesion_focus"
    pack_output
    status "finished - $OUT_DIR/lesion_focus/lesion_focus_comparison.md"
    ;;

  calibrate)
    install_reqs
    "$PY" research/experiments.py calibrate --roots "${ROOTS[@]}" --out "$OUT_DIR/calibration"
    pack_output
    status "finished - $OUT_DIR/calibration/calibration_report.md (web-app model: calibration/models/best)"
    ;;

  external)
    install_reqs
    EXT_DATA=""
    if "$PY" research/kaggle_tools.py find-data --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" > /dev/null 2>&1; then
      resolve_data; EXT_DATA="$DATA_DIR"
    fi
    "$PY" research/experiments.py external --roots "${ROOTS[@]}" --out "$OUT_DIR/external" --data "$EXT_DATA"
    pack_output
    status "finished - $OUT_DIR/external/external_report.md"
    ;;
esac
