"""AGENT 1 - loss-weight agent.
Objective : maximise validation mIoU of a short proxy training run.
Space     : lambda_k in [0, 2] for each loss in --keys (weights < 0.05 are set to 0).
Strategy  : Optuna TPE (Bayesian), seeded; first trials are default/manual configs for comparison.
Budget    : --trials proxy runs of --epochs epochs each.
Stopping  : budget exhausted OR no improvement for --patience trials.
Data      : train set for fitting, VALIDATION set for scoring. Test data is never touched.
Log       : <outdir>/loss_trials.jsonl and loss_trials.csv ; best weights in best_loss.txt"""
import argparse, json, time
from pathlib import Path
import optuna
from train import get_args, run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/kaggle/input")
    p.add_argument("--model", default="segformer")
    p.add_argument("--trials", type=int, default=12)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--keys", default="ce,dice,focal,lovasz", help="comma-separated loss names to search over")
    p.add_argument("--outdir", default="results/agent")
    p.add_argument("--runs_dir", default="out_agent")
    a = p.parse_args()
    keys = a.keys.split(",")
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    logf = out / "loss_trials.jsonl"; logf.write_text("")

    def to_str(params):
        lam = {k: params[k] for k in keys if params[k] >= 0.05}
        return ",".join(f"{k}:{v:.3f}" for k, v in lam.items())

    def objective(trial):
        params = {k: trial.suggest_float(k, 0.0, 2.0) for k in keys}
        loss = to_str(params)
        if not loss:
            raise optuna.TrialPruned()
        t0 = time.time()
        args = get_args(["--root", a.root, "--model", a.model, "--loss", loss, "--epochs", str(a.epochs),
                         "--seed", str(a.seed), "--bs", str(a.bs), "--workers", str(a.workers),
                         "--tag", f"agent_loss_trial{trial.number}", "--out", a.runs_dir])
        score = run(args)
        with open(logf, "a") as f:
            f.write(json.dumps({"trial": trial.number, "loss": loss, "val_mIoU": score,
                                "seconds": time.time() - t0}) + "\n")
        return score

    def patience(study, trial):
        if trial.number - study.best_trial.number >= a.patience:
            study.stop()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=a.seed))
    base = {k: 0.0 for k in keys}
    study.enqueue_trial({**base, "ce": 1.0})                                   # default: plain CE
    if "dice" in keys: study.enqueue_trial({**base, "ce": 1.0, "dice": 1.0})   # manual composite
    if "wce" in keys: study.enqueue_trial({**base, "wce": 1.0})                # class-weighted CE
    study.optimize(objective, n_trials=a.trials, callbacks=[patience])
    study.trials_dataframe().to_csv(out / "loss_trials.csv", index=False)
    best = to_str(study.best_trial.params)
    (out / "best_loss.txt").write_text(best)
    print("BEST composite loss:", best, "val mIoU", study.best_value)


if __name__ == "__main__":
    main()
