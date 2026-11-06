import sys
import numpy as np

z = np.load(sys.argv[1])
print("keys:", list(z.keys()))
for k in z.files:
    arr = z[k]
    print(f"  {k}: shape={arr.shape} dtype={arr.dtype} min={arr.min()} max={arr.max()}")

CLASSES = ["basophil", "eosinophil", "erythroblast", "immature granulocyte",
           "lymphocyte", "monocyte", "neutrophil", "platelet"]

# train and val only. This script originally looped over ("train", "val", "test")
# and printed the test label distribution too, and it was the FIRST thing run on the
# data — before the metric was chosen and before the "is the imbalance worth
# mitigating" question was answered. It showed labels, never images and never a
# score, and both of those questions were then decided on validation anyway, but a
# test statistic seen before a decision is the thing requirement 6 is about, so the
# split is dropped here rather than excused. The disclosure is in README.md.
for split in ("train", "val"):
    if f"{split}_labels" not in z.files:
        continue
    y = z[f"{split}_labels"].ravel()
    counts = np.bincount(y, minlength=8)
    print(f"\n{split}: n={y.size}")
    for c, n in enumerate(counts):
        bar = "#" * int(60 * n / counts.max())
        print(f"  {c} {CLASSES[c]:<22} {n:5d}  {100.0*n/y.size:5.1f}%  {bar}")
