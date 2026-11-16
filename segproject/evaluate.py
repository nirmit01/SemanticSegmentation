"""Evaluate checkpoint(s) on the TEST set using only settings fixed on validation.
Single model : python evaluate.py --ckpts segformer:out/TAG/best.pt --out results/eval/TAG
Ablation     : python evaluate.py --ckpts a:p1 b:p2 --post results/agent/post_config.json --out results/eval/final_ensemble --vis 8"""
import argparse, csv, json
from pathlib import Path
import cv2, numpy as np, torch
from torch.utils.data import DataLoader
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import CamVid, evaluate_models, full_metrics, ensemble_probs, dump_json
from agent_post import load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/kaggle/input")
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--post", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--out", required=True)
    p.add_argument("--vis", type=int, default=0)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    labels, models = [], []
    for s in a.ckpts:
        lab, path = s.split(":", 1)
        m, ck = load_model(path, dev); labels.append(lab); models.append(m)
    names, (h, w) = ck["names"], ck["hw"]; nc = len(names)
    ds = CamVid(a.root, a.split, h, w)
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2)

    V = []  # (name, models, weights, scales, flip, class_scale)
    if a.post is None:
        for l, m in zip(labels, models):
            V.append((f"{l} plain", [m], [1.0], (1.0,), False, None))
        final = V[-1]
    else:
        P = json.load(open(a.post)); assert P["labels"] == labels, "ckpt order must match post_config labels"
        sc, fl, w0 = tuple(P["scales"]), P["flip"], P["ens_w"]; cs = torch.tensor(P["class_scale"], device=dev)
        A, B = models
        V = [(f"{labels[0]} plain", [A], [1.0], (1.0,), False, None),
             (f"{labels[1]} plain", [B], [1.0], (1.0,), False, None),
             (f"{labels[0]} +TTA({P['tta']})", [A], [1.0], sc, fl, None),
             (f"{labels[1]} +TTA({P['tta']})", [B], [1.0], sc, fl, None),
             (f"Ensemble (w={w0}) plain", [A, B], [w0, 1 - w0], (1.0,), False, None),
             (f"Ensemble +TTA", [A, B], [w0, 1 - w0], sc, fl, None),
             ("Ensemble +TTA +class-scale (FINAL)", [A, B], [w0, 1 - w0], sc, fl, cs)]
        final = V[-1]

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    abl, final_cm = [], None
    for name, ms, ws, sc, fl, cs in V:
        cm, bm = evaluate_models(ms, ws, dl, nc, dev, sc, fl, cs, boundary=True)
        met = full_metrics(cm, bm, names)
        abl.append({"name": name, "metrics": met}); final_cm = cm
        print(f"{name:45s} mIoU {met['mIoU']:.4f}  mDice {met['mDice']:.4f}  PA {met['pixel_acc']:.4f}  BF1 {met.get('boundary_f1', 0):.4f}")
    dump_json(abl, out / "ablation.json")
    dump_json(abl[-1]["metrics"], out / "metrics.json")

    # error analysis: confusion matrix + most confused pairs (final variant)
    cmn = final_cm.cpu().numpy().astype(np.float64)
    np.save(out / "confusion_matrix.npy", cmn)
    rown = cmn / np.maximum(cmn.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(11, 10)); im = ax.imshow(rown, cmap="viridis")
    ax.set_xticks(range(nc)); ax.set_yticks(range(nc)); ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_yticklabels(names, fontsize=6); ax.set_xlabel("predicted"); ax.set_ylabel("ground truth")
    plt.colorbar(im); plt.tight_layout(); plt.savefig(out / "confusion_matrix.png", dpi=150); plt.close()
    pairs = [(names[i], names[j], rown[i, j], int(cmn[i, j])) for i in range(nc) for j in range(nc)
             if i != j and cmn[i].sum() > 0]
    pairs.sort(key=lambda t: -t[2])
    with open(out / "top_confusions.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["gt", "predicted_as", "fraction_of_gt_pixels", "pixels"]); wr.writerows(pairs[:20])

    if a.vis:  # qualitative results
        (out / "vis").mkdir(exist_ok=True)
        name, ms, ws, sc, fl, cs = final
        for i in range(0, len(ds), max(1, len(ds) // a.vis))[:a.vis]:
            x, y = ds[i]; img, _ = ds.load(i)
            with torch.no_grad():
                P_ = ensemble_probs(ms, ws, x[None].to(dev), y.shape, sc, fl)
                if cs is not None: P_ = P_ * cs.view(1, -1, 1, 1)
                pr = P_.argmax(1)[0].cpu().numpy()
            col = lambda m_: np.where((m_ == 255)[..., None], 0, ds.colors[np.minimum(m_, nc - 1)]).astype(np.uint8)
            panel = np.concatenate([img, col(y.numpy()), col(pr)], 1)
            cv2.imwrite(str(out / "vis" / f"{i:03d}.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
