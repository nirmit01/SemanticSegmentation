"""Train one model with one loss configuration. Model selection uses the VALIDATION set only.
Example: python train.py --root /kaggle/input --model segformer --loss ce:1,dice:1 --tag demo"""
import argparse, csv, time
from pathlib import Path
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from common import (CamVid, seed_everything, evaluate_models, metrics_from_cm, full_metrics,
                    class_freq, dump_json)
from losses import Composite, parse_loss, class_weights
from models import build_model, profile


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/kaggle/input")
    p.add_argument("--model", default="segformer", choices=["segformer", "deeplab"])
    p.add_argument("--variant", default="nvidia/mit-b2")
    p.add_argument("--encoder", default="resnet50")
    p.add_argument("--loss", default="ce:1")
    p.add_argument("--aug", default="basic", choices=["basic", "strong"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=6e-5)       # encoder lr
    p.add_argument("--head_mult", type=float, default=10)  # decoder lr = lr * head_mult
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--h", type=int, default=352)
    p.add_argument("--w", type=int, default=480)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max_train", type=int, default=None, help="debug: use only N images")
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--out", default="out")
    p.add_argument("--tag", required=True)
    return p.parse_args(argv)


def run(a):
    seed_everything(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = bool(a.amp) and dev == "cuda"
    outdir = Path(a.out) / a.tag
    outdir.mkdir(parents=True, exist_ok=True)

    tr = CamVid(a.root, "train", a.h, a.w, train=True, strong=a.aug == "strong", max_n=a.max_train)
    va = CamVid(a.root, "val", a.h, a.w, train=False, max_n=a.max_train)
    names, nc = tr.names, len(tr.names)
    g = torch.Generator(); g.manual_seed(a.seed)
    tl = DataLoader(tr, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                    pin_memory=True, generator=g, persistent_workers=a.workers > 0)
    vl = DataLoader(va, batch_size=4, shuffle=False, num_workers=a.workers)

    cfg = {"model": a.model, "variant": a.variant, "encoder": a.encoder}
    model = build_model(cfg, nc, True).to(dev)
    params, trainable, flops = profile(model, a.h, a.w, dev)
    enc, head = model.groups()
    opt = torch.optim.AdamW([{"params": enc, "lr": a.lr}, {"params": head, "lr": a.lr * a.head_mult}],
                            weight_decay=a.wd)
    total = a.epochs * len(tl); warm = max(1, min(100, total // 10))
    sched = LambdaLR(opt, lambda it: (it + 1) / warm if it < warm else (1 - (it - warm) / max(1, total - warm)) ** 0.9)

    weights = parse_loss(a.loss)
    cw = class_weights(class_freq(a.root, nc)).to(dev) if "wce" in weights else None
    crit = Composite(weights, nc, cw).to(dev)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    best, best_ep, ep_times, log = -1.0, -1, [], []
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for x, y in tl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev, dtype=torch.float16, enabled=amp):
                lg = model(x)
            loss = crit(lg.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        if dev == "cuda": torch.cuda.synchronize()
        ep_times.append(time.time() - t0)
        cm, _ = evaluate_models([model], [1.0], vl, nc, dev, boundary=False)
        m = metrics_from_cm(cm, names)
        log.append({"epoch": ep, "train_loss": tot / len(tl), "val_mIoU": m["mIoU"], "time_s": ep_times[-1]})
        print(f"[{a.tag}] ep {ep+1}/{a.epochs} loss {tot/len(tl):.4f} val mIoU {m['mIoU']:.4f} ({ep_times[-1]:.0f}s)", flush=True)
        if m["mIoU"] > best:
            best, best_ep = m["mIoU"], ep
            torch.save({"model": model.state_dict(), "cfg": cfg, "names": names, "hw": (a.h, a.w)},
                       outdir / "best.pt")

    # final validation metrics (with boundary F1) from the best checkpoint
    model.load_state_dict(torch.load(outdir / "best.pt", map_location=dev, weights_only=False)["model"])
    cm, bm = evaluate_models([model], [1.0], vl, nc, dev, boundary=True)
    val = full_metrics(cm, bm, names)
    x = torch.randn(1, 3, a.h, a.w, device=dev); model.eval()
    with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=amp):
        for _ in range(10): model(x)
        if dev == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50): model(x)
        if dev == "cuda": torch.cuda.synchronize()
    inf_ms = (time.time() - t0) / 50 * 1000

    dump_json({"tag": a.tag, "args": vars(a), "params": params, "trainable_params": trainable,
               "GFLOPs": flops / 1e9, "epoch_time_s": float(np.mean(ep_times)), "inference_ms_per_image": inf_ms,
               "best_epoch": best_ep, "val": val}, outdir / "val.json")
    with open(outdir / "log.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(log[0].keys())); wr.writeheader(); wr.writerows(log)
    del model, opt
    if dev == "cuda": torch.cuda.empty_cache()
    return val["mIoU"]


if __name__ == "__main__":
    run(get_args())
