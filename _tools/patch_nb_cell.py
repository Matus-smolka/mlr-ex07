"""Patch the appended results.csv cell: seconds now live in runtime.json."""
import json

NB = "Copy_of_ex07_student.ipynb"

OLD_HEAD = 'import csv\ncols = ["label", "params", "optimizer", "sampling", "best_epoch",\n        "val_accuracy", "val_balanced_accuracy", "seconds"]\n'

NEW = '''import csv, json
# results.csv holds only what two identical runs reproduce exactly; the measured
# seconds are in runtime.json, so wall-clock never breaks a `diff` of results.csv.
cols = ["label", "params", "optimizer", "sampling", "best_epoch",
        "val_accuracy", "val_balanced_accuracy"]
seconds = json.load(open("runtime.json")).get("per_model_seconds", {})
with open("results.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
print(f"{'model':<26}{'parameters':>11}{'optimizer':>10}{'sampling':>10}"
      f"{'epoch':>7}{'val acc':>10}{'val bal acc':>13}{'seconds':>9}")
for r in rows:
    print(f"{r['label']:<26}{int(r['params']):>11,}{r['optimizer']:>10}{r['sampling']:>10}"
          f"{r['best_epoch']:>7}{float(r['val_accuracy']):>10.4f}"
          f"{float(r['val_balanced_accuracy']):>13.4f}"
          f"{float(seconds.get(r['label'], float('nan'))):>9.1f}")
print(f"\\n{len(rows)} models compared on validation; the test columns are empty for all "
      f"but the three evaluated at the end.")
'''

nb = json.load(open(NB, encoding="utf-8"))
patched = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if src.startswith("import csv\ncols = ["):
        nb["cells"][i]["source"] = NEW.splitlines(keepends=True)
        patched += 1
        print(f"patched cell {i}")
json.dump(nb, open(NB, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"cells patched: {patched}")
