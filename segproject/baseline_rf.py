"""Traditional (non-deep) baseline: per-pixel Random Forest on hand-crafted features
(Lab colour, Gaussian scale-space, gradient magnitude, LoG, local std, normalised x/y position)."""
import argparse
import cv2, numpy as np, torch
from sklearn.ensemble import RandomForestClassifier
from common import CamVid, IGNORE, update_cm, BoundaryMeter, full_metrics, dump_json

W_, H_ = 240, 180  # work at 1/4 resolution, upsample predictions (nearest) to full resolution


def feats(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0]
    ch = [lab[..., 0], lab[..., 1], lab[..., 2]]
    for s in (1, 2, 4):
        ch.append(cv2.GaussianBlur(L, (0, 0), s))
    for s in (1, 2):
        g = cv2.GaussianBlur(L, (0, 0), s)
        gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
        ch.append(np.sqrt(gx ** 2 + gy ** 2))
    ch.append(cv2.Laplacian(cv2.GaussianBlur(L, (0, 0), 2), cv2.CV_32F))
    m, m2 = cv2.blur(L, (7, 7)), cv2.blur(L * L, (7, 7))
    ch.append(np.sqrt(np.maximum(m2 - m * m, 0)))
    for k in (1, 2):
        ch.append(cv2.GaussianBlur(lab[..., k], (0, 0), 4))
    h, w = L.shape
    yy, xx = np.mgrid[0:h, 0:w]
    ch += [(yy / h).astype(np.float32), (xx / w).astype(np.float32)]
    return np.stack(ch, -1).reshape(-1, len(ch))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/kaggle/input")
    p.add_argument("--trees", type=int, default=60)
    p.add_argument("--n_pix", type=int, default=1500, help="sampled pixels per training image")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    tr = CamVid(a.root, "train"); names, nc = tr.names, len(tr.names)
    X, Y = [], []
    for i in range(len(tr)):
        img, m = tr.load(i)
        img = cv2.resize(img, (W_, H_), interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, (W_, H_), interpolation=cv2.INTER_NEAREST)
        f, y = feats(img), m.ravel()
        idx = np.flatnonzero(y != IGNORE)
        idx = rng.choice(idx, min(a.n_pix, len(idx)), replace=False)
        X.append(f[idx]); Y.append(y[idx])
    X, Y = np.concatenate(X), np.concatenate(Y)
    print("training RF on", X.shape)
    rf = RandomForestClassifier(n_estimators=a.trees, max_depth=20, min_samples_leaf=5,
                                n_jobs=-1, random_state=a.seed).fit(X, Y)
    for split, fname in (("val", "val_metrics.json"), ("test", "metrics.json")):
        ds = CamVid(a.root, split)
        cm = torch.zeros(nc, nc, dtype=torch.long); bm = BoundaryMeter()
        for i in range(len(ds)):
            img, m = ds.load(i)
            small = cv2.resize(img, (W_, H_), interpolation=cv2.INTER_AREA)
            pred = rf.predict(feats(small)).reshape(H_, W_).astype(np.uint8)
            pred = cv2.resize(pred, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)
            pt, gt = torch.from_numpy(pred.astype(np.int64))[None], torch.from_numpy(m.astype(np.int64))[None]
            update_cm(cm, pt, gt, nc); bm.update(pt, gt)
        met = full_metrics(cm, bm, names)
        dump_json(met, f"results/eval/baseline_rf/{fname}")
        print(split, "mIoU", round(met["mIoU"], 4), "mDice", round(met["mDice"], 4), "PA", round(met["pixel_acc"], 4))


if __name__ == "__main__":
    main()
