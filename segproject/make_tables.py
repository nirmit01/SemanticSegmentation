"""Collect all results into markdown tables (results/tables.md)."""
import glob, json
import numpy as np
from pathlib import Path

L = lambda p: json.load(open(p))
out = []

out.append("## Training runs (validation set)\n")
out.append("| run | params (M) | trainable (M) | GFLOPs | s/epoch | ms/img | val mIoU | val mDice | val BF1 |")
out.append("|---|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob("out/*/val.json")):
    d = L(f); v = d["val"]
    out.append(f"| {d['tag']} | {d['params']/1e6:.1f} | {d['trainable_params']/1e6:.1f} | {d['GFLOPs']:.1f} | "
               f"{d['epoch_time_s']:.1f} | {d['inference_ms_per_image']:.1f} | {v['mIoU']:.4f} | {v['mDice']:.4f} | {v.get('boundary_f1', float('nan')):.4f} |")

out.append("\n## Test results, one row per evaluated run\n")
out.append("| run | mIoU | mDice | mPrecision | mRecall | PixelAcc | BoundaryF1 |")
out.append("|---|---|---|---|---|---|---|")
for f in sorted(glob.glob("results/eval/*/metrics.json")):
    m = L(f)
    out.append(f"| {Path(f).parent.name} | {m['mIoU']:.4f} | {m['mDice']:.4f} | {m['mPrecision']:.4f} | "
               f"{m['mRecall']:.4f} | {m['pixel_acc']:.4f} | {m.get('boundary_f1', float('nan')):.4f} |")

out.append("\n## Final configuration, mean ± std over seeds (test)\n")
out.append("| model | seeds | mIoU | mDice | mPrecision | mRecall | PixelAcc | BoundaryF1 |")
out.append("|---|---|---|---|---|---|---|---|")
for pre, mdl in [(p_, m_) for p_ in ("final", "v2_final") for m_ in ("segformer", "deeplab")]:
    fs = sorted(glob.glob(f"results/eval/{pre}_{mdl}_s*/metrics.json"))
    if not fs: continue
    ms = [L(f) for f in fs]
    cell = lambda k: f"{np.mean([m[k] for m in ms]):.4f} ± {np.std([m[k] for m in ms]):.4f}"
    out.append(f"| {pre} {mdl} | {len(ms)} | " + " | ".join(cell(k) for k in
               ["mIoU", "mDice", "mPrecision", "mRecall", "pixel_acc", "boundary_f1"]) + " |")

for name in ("final_ensemble", "v2_final_ensemble"):
    f = Path(f"results/eval/{name}/ablation.json")
    if f.exists():
        out.append(f"\n## Improvement ablation: {name} (test)\n")
        out.append("| variant | mIoU | mDice | PixelAcc | BoundaryF1 |"); out.append("|---|---|---|---|---|")
        for r in L(f):
            m = r["metrics"]
            out.append(f"| {r['name']} | {m['mIoU']:.4f} | {m['mDice']:.4f} | {m['pixel_acc']:.4f} | {m.get('boundary_f1', float('nan')):.4f} |")

Path("results").mkdir(exist_ok=True)
Path("results/tables.md").write_text("\n".join(out))
print("\n".join(out))
