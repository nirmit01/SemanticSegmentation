"""AGENT 2 - inference-side threshold agent (VALIDATION set only).
Objective : maximise validation mIoU.
Stages    : (1) pick TTA config from {none, flip, multi-scale, multi-scale+flip}
            (2) pick ensemble weight w in {0,0.1,...,1}
            (3) per-class probability scale s_c in [0.25, 4] by coordinate ascent
                (prediction = argmax_c s_c * p_c; this is the per-class decision threshold)
Stopping  : a full coordinate pass with no gain > 1e-5, or --max_pass passes.
Log       : results/agent/post_trials.csv ; result in results/agent/post_config.json"""
import argparse, csv
from pathlib import Path
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from common import CamVid, evaluate_models, metrics_from_cm, model_probs, update_cm, miou_from_cm, dump_json
from models import build_model


def load_model(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build_model(ck["cfg"], len(ck["names"]), False)
    m.load_state_dict(ck["model"])
    return m.to(dev).eval(), ck


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/kaggle/input")
    p.add_argument("--ckpts", nargs=2, required=True, help="label:path label:path")
    p.add_argument("--max_pass", type=int, default=3)
    p.add_argument("--out_dir", default="results/agent")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    labels, models = [], []
    for s in a.ckpts:
        lab, path = s.split(":", 1)
        m, ck = load_model(path, dev); labels.append(lab); models.append(m)
    names, (h, w) = ck["names"], ck["hw"]; nc = len(names)
    va = CamVid(a.root, "val", h, w)
    vl = DataLoader(va, batch_size=4, shuffle=False, num_workers=2)
    rows = []

    def log(stage, setting, score):
        rows.append({"stage": stage, "setting": setting, "val_mIoU": score}); print(stage, setting, f"{score:.4f}")

    # (1) TTA config
    cands = {"none": ((1.0,), False), "flip": ((1.0,), True),
             "ms": ((0.75, 1.0, 1.25), False), "ms_flip": ((0.75, 1.0, 1.25), True)}
    best_t, best_s = None, -1
    for name, (sc, fl) in cands.items():
        cm, _ = evaluate_models(models, [1, 1], vl, nc, dev, sc, fl, boundary=False)
        s = miou_from_cm(cm); log("tta", name, s)
        if s > best_s: best_t, best_s = name, s
    scales, flip = cands[best_t]

    # cache validation probabilities (network resolution) for the cheap searches below
    P = [[] for _ in models]; G = []
    with torch.no_grad():
        for x, y in vl:
            x, y = x.to(dev), y.to(dev)
            for i, m in enumerate(models):
                P[i].append(model_probs(m, x, (h, w), scales, flip).half())
            G.append(F.interpolate(y[:, None].float(), size=(h, w), mode="nearest")[:, 0].long())
    P = [torch.cat(v) for v in P]; G = torch.cat(G)

    def score(getp, scale):
        cm = torch.zeros(nc, nc, dtype=torch.long, device=dev)
        for i in range(0, len(G), 16):
            pr = (getp(i, i + 16) * scale.view(1, -1, 1, 1)).argmax(1)
            update_cm(cm, pr, G[i:i + 16], nc)
        return miou_from_cm(cm)

    ones = torch.ones(nc, device=dev)
    # (2) ensemble weight
    best_w, best_s = 0.5, -1
    for wgt in [i / 10 for i in range(11)]:
        s = score(lambda a_, b_: wgt * P[0][a_:b_].float() + (1 - wgt) * P[1][a_:b_].float(), ones)
        log("ens_weight", f"w_{labels[0]}={wgt}", s)
        if s > best_s: best_w, best_s = wgt, s
    Pe = (best_w * P[0].float() + (1 - best_w) * P[1].float()).half(); del P
    getp = lambda a_, b_: Pe[a_:b_].float()

    # (3) per-class scale by coordinate ascent
    scale = ones.clone(); cur = score(getp, scale); log("class_scale", "init(all 1.0)", cur)
    for ps in range(a.max_pass):
        improved = False
        for c in range(nc):
            best_v, best_sc = scale[c].item(), cur
            for mult in (0.5, 0.7, 0.85, 1.2, 1.5, 2.0):
                v = min(4.0, max(0.25, scale[c].item() * mult))
                trial = scale.clone(); trial[c] = v
                s = score(getp, trial)
                if s > best_sc + 1e-5: best_v, best_sc = v, s
            if best_sc > cur + 1e-5:
                scale[c] = best_v; cur = best_sc; improved = True
        log("class_scale", f"pass {ps+1}", cur)
        if not improved: break

    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    with open(str(Path(a.out_dir) / "post_trials.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["stage", "setting", "val_mIoU"]); wr.writeheader(); wr.writerows(rows)
    dump_json({"labels": labels, "tta": best_t, "scales": list(scales), "flip": flip, "ens_w": best_w,
               "class_scale": scale.tolist(), "val_mIoU_final": cur}, str(Path(a.out_dir) / "post_config.json"))
    print("saved", Path(a.out_dir) / "post_config.json")


if __name__ == "__main__":
    main()
