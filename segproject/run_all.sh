#!/usr/bin/env bash
# Full pipeline. Override with env vars, e.g.:  EP=15 AEP=4 TRIALS=6 bash run_all.sh   (fast mode)
set -euo pipefail
ROOT=${ROOT:-/kaggle/input}
EP=${EP:-40}          # epochs per full training run
AEP=${AEP:-8}         # epochs per agent proxy run
TRIALS=${TRIALS:-12}  # agent trials for loss weights
MODELS="segformer deeplab"
ev() { python evaluate.py --root "$ROOT" --ckpts "$1:out/$2/best.pt" --out "results/eval/$2"; }
tr() { python train.py --root "$ROOT" --model "$1" --loss "$2" --epochs "$EP" --tag "$3" "${@:4}"; ev "$1" "$3"; }

python dataset_stats.py --root "$ROOT"                       # Part: dataset statistics
python baseline_rf.py   --root "$ROOT"                       # Part A: traditional baseline

for m in $MODELS; do for l in ce wce focal dice tversky lovasz; do   # Part C: individual losses
  tr $m "$l:1" "loss_${m}_${l}"
done; done
for m in $MODELS; do tr $m "ce:1,dice:1" "comp_manual_${m}"; done      # manual composite

python agent_loss.py --root "$ROOT" --trials "$TRIALS" --epochs "$AEP" # agent fixes lambda_k
BEST=$(cat results/agent/best_loss.txt); echo "agent loss = $BEST"
for m in $MODELS; do tr $m "$BEST" "comp_agent_${m}"; done

for m in $MODELS; do for s in 0 1 2; do                                # final config, 3 seeds (strong aug = B-2)
  tr $m "$BEST" "final_${m}_s${s}" --aug strong --seed $s
done; done

python agent_post.py --root "$ROOT" --ckpts segformer:out/final_segformer_s0/best.pt deeplab:out/final_deeplab_s0/best.pt
python evaluate.py --root "$ROOT" \
  --ckpts segformer:out/final_segformer_s0/best.pt deeplab:out/final_deeplab_s0/best.pt \
  --post results/agent/post_config.json --out results/eval/final_ensemble --vis 8
python make_tables.py
