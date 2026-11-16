"""Shared utilities: CamVid dataset, augmentation, inference/TTA, metrics (incl. boundary F1)."""
import json, os, random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

IGNORE = 255  # Void / unlabeled pixels are ignored everywhere
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)


def seed_everything(seed=0):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- data
def find_root(root):
    hits = list(Path(root).rglob("class_dict.csv"))
    if not hits:
        raise FileNotFoundError(f"class_dict.csv not found under {root}. Point --root at the CamVid folder.")
    return hits[0].parent


def load_classes(base):
    df = pd.read_csv(base / "class_dict.csv")
    df.columns = [c.strip().lower() for c in df.columns]
    names = [str(n).strip() for n in df["name"]]
    rgb = df[["r", "g", "b"]].values
    keep = [i for i, n in enumerate(names) if n.lower() != "void"]
    lut = np.full(1 << 24, IGNORE, np.uint8)
    for new, i in enumerate(keep):
        r, g, b = (int(v) for v in rgb[i])
        lut[(r << 16) | (g << 8) | b] = new
    return [names[i] for i in keep], lut, np.array([rgb[i] for i in keep], np.uint8)


def augment(img, mask, h, w, strong):
    if strong:
        s = random.uniform(0.75, 1.75)
        nh, nw = int(h * s), int(w * s)
    else:
        nh, nw = h, w
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    ph, pw = max(h - nh, 0), max(w - nw, 0)
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        mask = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=IGNORE)
    H, W = mask.shape
    y0, x0 = random.randint(0, H - h), random.randint(0, W - w)
    img, mask = img[y0:y0 + h, x0:x0 + w], mask[y0:y0 + h, x0:x0 + w]
    if random.random() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    if strong:  # photometric jitter
        img = np.clip(img.astype(np.float32) * random.uniform(0.7, 1.3) + random.uniform(-25, 25), 0, 255)
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


class CamVid(Dataset):
    """Train: random aug at (h,w). Eval: image resized to (h,w), mask kept at FULL resolution."""

    def __init__(self, root, split, h=352, w=480, train=False, strong=False, max_n=None):
        self.base = find_root(root)
        self.names, self.lut, self.colors = load_classes(self.base)
        self.files = sorted((self.base / split).glob("*.png"))
        if max_n:
            self.files = self.files[:max_n]
        self.lbl_dir = self.base / f"{split}_labels"
        self.h, self.w, self.train, self.strong = h, w, train, strong

    def __len__(self):
        return len(self.files)

    def load(self, i):
        f = self.files[i]
        img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        lab = cv2.cvtColor(cv2.imread(str(self.lbl_dir / f"{f.stem}_L.png")), cv2.COLOR_BGR2RGB).astype(np.uint32)
        mask = self.lut[(lab[..., 0] << 16) | (lab[..., 1] << 8) | lab[..., 2]]
        return img, mask

    def __getitem__(self, i):
        img, mask = self.load(i)
        if self.train:
            img, mask = augment(img, mask, self.h, self.w, self.strong)
        else:
            img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        img = (img.astype(np.float32) - MEAN) / STD
        return torch.from_numpy(img.transpose(2, 0, 1).copy()), torch.from_numpy(mask.astype(np.int64))


def class_freq(root, nc, cache="results/class_freq.npy"):
    if os.path.exists(cache):
        return np.load(cache)
    ds = CamVid(root, "train")
    cnt = np.zeros(nc)
    for i in range(len(ds)):
        _, m = ds.load(i)
        cnt += np.bincount(m[m != IGNORE].ravel(), minlength=nc)[:nc]
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, cnt)
    return cnt


# ----------------------------------------------------------------------------- inference
def model_probs(model, x, out_size, scales=(1.0,), flip=False):
    """Softmax probabilities averaged over scales / horizontal flip (test-time augmentation)."""
    H, W = x.shape[-2:]
    acc, n = 0, 0
    for s in scales:
        if s == 1.0:
            xs = x
        else:
            hh, ww = max(32, int(round(H * s / 32)) * 32), max(32, int(round(W * s / 32)) * 32)
            xs = F.interpolate(x, size=(hh, ww), mode="bilinear", align_corners=False)
        for fl in ([False, True] if flip else [False]):
            xi = torch.flip(xs, [3]) if fl else xs
            with torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):
                lg = model(xi)
            lg = lg.float()
            if fl:
                lg = torch.flip(lg, [3])
            lg = F.interpolate(lg, size=out_size, mode="bilinear", align_corners=False)
            acc = acc + lg.softmax(1)
            n += 1
    return acc / n


def ensemble_probs(models, weights, x, out_size, scales=(1.0,), flip=False):
    tot = float(sum(weights))
    return sum(w * model_probs(m, x, out_size, scales, flip) for m, w in zip(models, weights)) / tot


# ----------------------------------------------------------------------------- metrics
@torch.no_grad()
def update_cm(cm, pred, gt, nc):
    v = gt != IGNORE
    cm += torch.bincount(gt[v] * nc + pred[v], minlength=nc * nc).view(nc, nc)


def miou_from_cm(cm):
    cm = cm.double()
    tp = torch.diag(cm); den = cm.sum(0) + cm.sum(1) - tp
    present = cm.sum(1) > 0
    return float((tp[present] / den[present].clamp(min=1)).mean())


def metrics_from_cm(cm, names):
    cm = cm.double().cpu().numpy() if torch.is_tensor(cm) else cm.astype(np.float64)
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    with np.errstate(all="ignore"):
        iou = tp / (tp + fp + fn); dice = 2 * tp / (2 * tp + fp + fn)
        prec = tp / (tp + fp); rec = tp / (tp + fn)
    present = (tp + fn) > 0  # classes that occur in the ground truth
    mean = lambda v: float(np.mean(np.nan_to_num(v)[present]))
    per = {}
    for c, n in enumerate(names):
        per[n] = ({"iou": float(np.nan_to_num(iou[c])), "dice": float(np.nan_to_num(dice[c])),
                   "precision": float(np.nan_to_num(prec[c])), "recall": float(np.nan_to_num(rec[c])),
                   "gt_pixels": int(tp[c] + fn[c])} if present[c] else None)
    return {"mIoU": mean(iou), "mDice": mean(dice), "mPrecision": mean(prec), "mRecall": mean(rec),
            "pixel_acc": float(tp.sum() / cm.sum()), "n_classes_present": int(present.sum()), "per_class": per}


def _dilate(b, r):
    return F.max_pool2d(b.float().unsqueeze(1), 2 * r + 1, 1, r).squeeze(1) > 0


def _boundary(lbl):
    b = torch.zeros_like(lbl, dtype=torch.bool)
    dx = lbl[:, :, :-1] != lbl[:, :, 1:]
    dy = lbl[:, :-1, :] != lbl[:, 1:, :]
    b[:, :, :-1] |= dx; b[:, :, 1:] |= dx
    b[:, :-1, :] |= dy; b[:, 1:, :] |= dy
    return b


class BoundaryMeter:
    """Class-agnostic boundary precision/recall/F1 with a pixel tolerance (aggregated over dataset)."""

    def __init__(self, tol=3):
        self.tol, self.c = tol, None

    @torch.no_grad()
    def update(self, pred, gt):
        if self.c is None:
            self.c = torch.zeros(4, dtype=torch.float64, device=pred.device)
        ign = _dilate(gt == IGNORE, self.tol)
        bp = _boundary(pred) & ~ign
        bg = _boundary(gt) & ~ign
        self.c += torch.stack([(bp & _dilate(bg, self.tol)).sum(), bp.sum(),
                               (bg & _dilate(bp, self.tol)).sum(), bg.sum()]).double()

    def result(self):
        c = self.c.cpu().numpy()
        p, r = c[0] / max(c[1], 1), c[2] / max(c[3], 1)
        return {"boundary_precision": float(p), "boundary_recall": float(r),
                "boundary_f1": float(2 * p * r / (p + r + 1e-9))}


def full_metrics(cm, bm, names):
    d = metrics_from_cm(cm, names)
    if bm is not None and bm.c is not None:
        d.update(bm.result())
    return d


@torch.no_grad()
def evaluate_models(models, weights, loader, nc, device, scales=(1.0,), flip=False, class_scale=None, boundary=True):
    for m in models:
        m.eval()
    cm = torch.zeros(nc, nc, dtype=torch.long, device=device)
    bm = BoundaryMeter() if boundary else None
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device)
        P = ensemble_probs(models, weights, x, y.shape[-2:], scales, flip)
        if class_scale is not None:
            P = P * class_scale.view(1, -1, 1, 1)
        pred = P.argmax(1)
        update_cm(cm, pred, y, nc)
        if bm is not None:
            bm.update(pred, y)
    return cm, bm


def dump_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
