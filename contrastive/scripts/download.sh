#!/usr/bin/env bash
set -e

BASE_URL="https://huggingface.co/cwh555/LearningFromShapes/resolve/main"

# -------- checkpoint (STN) --------
mkdir -p checkpoint

declare -A STN_FILES=(
  ["new_stn.pth"]="new_stn.pth"
  ["stn.pth"]="stn.pth"
)

for local in "${!STN_FILES[@]}"; do
  if [ ! -f "checkpoint/$local" ]; then
    echo "[INFO] Downloading checkpoint/$local"
    wget -q --show-progress \
      "$BASE_URL/${STN_FILES[$local]}" \
      -O "checkpoint/$local"
  else
    echo "[OK] checkpoint/$local exists"
  fi
done

# -------- deformation metric models --------
declare -A MODELS=(
  ["grid"]="grid.pth"
  ["cage"]="cage.pth"
  ["cage_wo_res"]="wo.pth"
  ["testing"]="test.pth"
)

for dir in "${!MODELS[@]}"; do
  target_dir="checkpoints/$dir"
  target_file="$target_dir/best_encoder.pth"
  src_file="${MODELS[$dir]}"

  mkdir -p "$target_dir"

  if [ ! -f "$target_file" ]; then
    echo "[INFO] Downloading $target_file"
    wget -q --show-progress \
      "$BASE_URL/$src_file" \
      -O "$target_file"
  else
    echo "[OK] $target_file exists"
  fi
done

echo "All required checkpoints are ready."
