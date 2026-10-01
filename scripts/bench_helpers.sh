#!/bin/bash
# ================================================================
# bench_helpers.sh — shared helpers for all benchmark runs.
# Source this from the repo root:  source scripts/bench_helpers.sh
#
# Requires DATA_ROOT to point at your dataset root, e.g.:
#   export DATA_ROOT=/path/to/datasets
# Expected layout:
#   $DATA_ROOT/TempCompass/{videos,yes_no.json,caption_matching.json,multi-choice.json}
#     (videos/ must also contain the *_reverse.mp4 clips for τ profile extraction)
#   $DATA_ROOT/AoTBench/{data_files,...}
#   $DATA_ROOT/TVBench/{json,video}
#   $DATA_ROOT/MVBench/{json,video}
# ================================================================

: "${DATA_ROOT:?Set DATA_ROOT to your dataset root first (export DATA_ROOT=/path/to/datasets)}"

export TC_V=$DATA_ROOT/TempCompass/videos
export TC_D=$DATA_ROOT/TempCompass
export TC_QA=$TC_D/yes_no.json
export AOT=$DATA_ROOT/AoTBench
export AOT_JSONS="ReverseFilm UCF101 Rtime_t2v Rtime_v2t AoTBench_QA"
export TV=$DATA_ROOT/TVBench
export MV=$DATA_ROOT/MVBench

# Path check: source scripts/bench_helpers.sh && bench_paths_check
bench_paths_check () {
  echo "TC_V: $TC_V $([ -d "$TC_V" ] && echo OK || echo 'NOT FOUND')"
  echo "AOT:  $AOT $([ -d "$AOT/data_files" ] && echo OK || echo 'NOT FOUND (missing data_files/)')"
  echo "TV:   $TV $([ -d "$TV" ] && echo OK || echo 'NOT FOUND')"
  echo "MV:   $MV $([ -d "$MV/json" ] && echo OK || echo 'NOT FOUND (missing json/)')"
}

# Usage: bench_chain <model> <n_frames> <benches> [tai args...]
#   benches: space-separated subset of "tc aot tv mv"
#   with no extra args → baseline; otherwise → --method tai <args>
# Examples:
#   bench_chain $MODEL 16 "tc aot tv mv"                                  # baseline
#   bench_chain $MODEL 16 "tc aot tv mv" --tau_profile $PROFILE --beta $BETA
bench_chain () {
  local M=$1 NF=$2 BENCHES=$3; shift 3
  local ARGS=("$@")
  local METHOD_ARGS=()
  if [ ${#ARGS[@]} -eq 0 ]; then
    METHOD_ARGS=(--method baseline)
  else
    METHOD_ARGS=(--method tai "${ARGS[@]}")
  fi
  for B in $BENCHES; do
    case $B in
      tc)
        python evaluations/eval_tempcompass.py --video_dir $TC_V --data_dir $TC_D \
          --n_frames $NF --output_dir bench_all_tc --skip_existing \
          --model_name "$M" "${METHOD_ARGS[@]}" ;;
      aot)
        for j in $AOT_JSONS; do
          python evaluations/eval_aotbench.py --json_path $AOT/data_files/${j}.json --video_dir $AOT \
            --n_frames $NF --output_dir bench_all_aot --skip_existing \
            --model_name "$M" "${METHOD_ARGS[@]}"
        done ;;
      tv)
        python evaluations/eval_tvbench.py --tvbench_dir $TV \
          --n_frames $NF --output_dir bench_all_tv --skip_existing \
          --model_name "$M" "${METHOD_ARGS[@]}" ;;
      mv)
        python evaluations/eval_mvbench.py --mvbench_dir $MV \
          --n_frames $NF --output_dir bench_all_mv --skip_existing \
          --model_name "$M" "${METHOD_ARGS[@]}" ;;
    esac
  done
}
