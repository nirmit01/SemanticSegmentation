# Semantic Segmentation project (CamVid, SegFormer vs DeepLabV3+)

## Dataset
CamVid on Kaggle: https://www.kaggle.com/datasets/carlolepelaars/camvid
Official split: 367 train / 101 val / 233 test, 960x720, 31 classes + Void (ignored), strong class imbalance.
Citation: Brostow, Fauqueur, Cipolla, "Semantic object classes in video: A high-definition ground truth database", Pattern Recognition Letters, 2009.

## Run on a Kaggle notebook (recommended)
1. New notebook -> Add Input -> search "camvid" by carlolepelaars. Settings: Accelerator GPU T4/P100, Internet ON.
2. Upload this folder as a Kaggle dataset (or paste the files), then in a cell:
   ```
   !pip install -q segmentation-models-pytorch optuna
   %cd /kaggle/working && !cp -r /kaggle/input/<your-code-dataset>/* .
   ```
3. Smoke test (2-3 minutes) before anything long:
   ```
   !python train.py --root /kaggle/input --model segformer --loss ce:1 --epochs 1 --max_train 16 --tag smoke
   !python train.py --root /kaggle/input --model deeplab   --loss ce:1 --epochs 1 --max_train 16 --tag smoke2
   ```
4. Full pipeline: `!bash run_all.sh` (or fast mode: `!EP=15 AEP=4 TRIALS=6 bash run_all.sh`).
   Kaggle sessions last ~9-12 h; if short on time, run stages from run_all.sh one at a time.

## Run locally
`kaggle datasets download -d carlolepelaars/camvid --unzip -p data`, `pip install -r requirements.txt`, then
`ROOT=data bash run_all.sh` (needs an NVIDIA GPU; the code searches ROOT recursively for class_dict.csv).

## Files
| file | what it does | project part |
|---|---|---|
| dataset_stats.py | split sizes, resolution, class-pixel distribution | Dataset |
| baseline_rf.py | Random Forest on hand-crafted pixel features | A (traditional baseline) |
| models.py | SegFormer-B2 (transformer) and DeepLabV3+-R50 (CNN+ASPP), params/FLOPs | B-1 |
| train.py | identical pipeline for both models; time/epoch, inference ms | B-1 |
| losses.py | CE, weighted CE, Focal, Dice, Tversky, Lovasz + composite sum(lambda_k L_k) | C |
| agent_loss.py | Optuna agent picks lambda_k on validation | C (agent) |
| agent_post.py | agent picks TTA, ensemble weight, per-class thresholds on validation | B-2 / C (agent) |
| evaluate.py | test metrics, boundary F1, ablation, confusion matrix, top confusions, visuals | D |
| make_tables.py | results/tables.md with all tables, mean +- std over 3 seeds | D |

## B-2 improvements implemented (each measured separately)
Training side: strong augmentation (`--aug strong`, compare with `loss_*`/`comp_*` runs which use basic aug).
Inference side: TTA (flip / multi-scale), 2-model ensemble, per-class probability scaling.
Class re-balancing: `wce` loss. Report the ones that did NOT help too, the tables will show them.

## Leakage rules followed
Test set is only read by evaluate.py / baseline_rf.py (final reporting). Checkpoint selection, agent search,
loss weights, TTA, ensemble weight and class scales all use the validation set only. Agent logs:
results/agent/loss_trials.jsonl|csv and results/agent/post_trials.csv.

## What you still must write / do yourself
- Part A literature study, comparison table (pick and cite your own papers; justify the 2 models with published benchmarks).
- Methodology diagram (the brief says self-designed, no AI tools), pseudocode, maths, complexity (Part E).
- Architecture diagrams for both models, the hyperparameter table (see `args` in out/*/val.json), error analysis text.
- Slides and viva preparation: you must be able to explain every line here.
