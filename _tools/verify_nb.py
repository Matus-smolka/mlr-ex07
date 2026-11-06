import json

old = json.load(open("Copy_of_ex07_student.BACKUP.ipynb", encoding="utf-8"))
new = json.load(open("Copy_of_ex07_student.ipynb", encoding="utf-8"))

print(f"backup cells: {len(old['cells'])}   new cells: {len(new['cells'])}")
assert len(new['cells']) >= len(old['cells']), "cells were lost!"

diff = 0
for i, (a, b) in enumerate(zip(old["cells"], new["cells"])):
    if a.get("source") != b.get("source") or a.get("cell_type") != b.get("cell_type"):
        diff += 1
        print(f"  CHANGED cell {i}")
print(f"original cells compared: {len(old['cells'])}   changed: {diff}")

# your own worked answers must still be there verbatim
worked = {5: "grad_sig = (y_hat_sig - y) * (y_hat_sig*(1-y_hat_sig)) * x",
          9: "w -= eta * error * x",
          20: "grad_a2 = y_hat - y",
          24: "grad_a1 = grad_h1 *(a1>0)",
          27: "grad_a = grad_h * (a[k - 1] > 0)",
          39: "L = torch.sum(0.5*(y_hat-y)**2)"}
for idx, needle in worked.items():
    src = "".join(new["cells"][idx]["source"])
    print(f"  cell {idx}: your code still present = {needle in src}")

print("\nnew cells:")
for i, c in enumerate(new["cells"][len(old["cells"]):], start=len(old["cells"])):
    first = "".join(c["source"]).strip().splitlines()[0][:78] if c["source"] else ""
    print(f"  {i:>3} [{c['cell_type']:<8}] {first}")
