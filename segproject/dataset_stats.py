"""Dataset statistics: images per split, resolution, #classes, class-pixel distribution."""
import argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import CamVid, IGNORE, dump_json


def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", default="/kaggle/input"); a = p.parse_args()
    stats, dist = {}, {}
    for split in ("train", "val", "test"):
        ds = CamVid(a.root, split); nc = len(ds.names)
        cnt, sizes = np.zeros(256), set()
        for i in range(len(ds)):
            img, m = ds.load(i); sizes.add(tuple(img.shape[:2]))
            cnt += np.bincount(m.ravel(), minlength=256)
        valid = cnt[:nc].sum()
        stats[split] = {"images": len(ds), "resolutions_HxW": sorted(sizes), "void_fraction": float(cnt[IGNORE] / cnt.sum())}
        dist[split] = (cnt[:nc] / valid)
        stats[split]["class_pixel_fraction"] = {n: float(v) for n, v in zip(ds.names, dist[split])}
    stats["num_classes"] = nc; stats["class_names"] = ds.names
    dump_json(stats, "results/dataset_stats.json")
    order = np.argsort(-dist["train"])
    plt.figure(figsize=(12, 4))
    plt.bar(range(nc), dist["train"][order]); plt.yscale("log")
    plt.xticks(range(nc), [ds.names[i] for i in order], rotation=90, fontsize=7)
    plt.ylabel("pixel fraction (train, log)"); plt.title("CamVid class-pixel distribution"); plt.tight_layout()
    plt.savefig("results/class_distribution.png", dpi=150)
    for s in ("train", "val", "test"):
        print(s, stats[s]["images"], "images", stats[s]["resolutions_HxW"])
    print("classes:", nc)


if __name__ == "__main__":
    main()
