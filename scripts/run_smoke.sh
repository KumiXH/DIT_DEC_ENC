#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
output_root="${1:-work/smoke}"

if [[ "$output_root" = /* ]]; then
  output_path="$output_root"
else
  output_path="$project_root/$output_root"
fi

data_path="$output_path/data"
runs_path="$output_path/runs"
configs=(
  "wan_encoder.yaml"
  "wan_decoder.yaml"
  "wan_autoencoder.yaml"
  "flashvsr_vae_encoder.yaml"
  "flashvsr_lq_proj.yaml"
  "flashvsr_decoder_unconditional.yaml"
  "flashvsr_decoder_conditional.yaml"
)

cd "$project_root"

"$python_bin" -m distill_codec.cli make-mock-data \
  --output "$data_path" \
  --count 4 \
  --size 32

for config_name in "${configs[@]}"; do
  config_path="$project_root/configs/smoke/$config_name"
  run_name="${config_name%.yaml}"
  run_path="$runs_path/$run_name"

  "$python_bin" -m distill_codec.cli probe \
    --config "$config_path" \
    --set "data.lq_root=$data_path/lq" \
    --set "data.gt_root=$data_path/gt" \
    --set "data.lq_size=[32,32]" \
    --set "data.gt_size=[32,32]" \
    --set "run.output_dir=$run_path"

  "$python_bin" -m distill_codec.cli train \
    --config "$config_path" \
    --set "data.lq_root=$data_path/lq" \
    --set "data.gt_root=$data_path/gt" \
    --set "data.lq_size=[32,32]" \
    --set "data.gt_size=[32,32]" \
    --set "run.output_dir=$run_path" \
    --set "trainer.max_steps=1" \
    --set "trainer.tensorboard=true"

  checkpoint_path="$run_path/checkpoints/step_00000001.pt"
  "$python_bin" -m distill_codec.cli train \
    --config "$config_path" \
    --set "data.lq_root=$data_path/lq" \
    --set "data.gt_root=$data_path/gt" \
    --set "data.lq_size=[32,32]" \
    --set "data.gt_size=[32,32]" \
    --set "run.output_dir=$run_path" \
    --set "trainer.max_steps=2" \
    --set "trainer.tensorboard=true" \
    --resume "$checkpoint_path"
done
