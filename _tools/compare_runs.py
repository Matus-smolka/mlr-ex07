"""Diff two results.csv files on the columns they share.

Used after the test-split refactor: the new results.csv dropped the two always-empty
test columns, so the comparison is on the intersection of the headers. Any difference
in a number means the refactor changed behaviour, which it must not have - it only
moved when the test arrays are read from the file.
"""
import csv
import sys

a_path, b_path = sys.argv[1], sys.argv[2]
with open(a_path, encoding="utf-8") as f:
    a = list(csv.DictReader(f))
with open(b_path, encoding="utf-8") as f:
    b = list(csv.DictReader(f))

shared = [k for k in a[0] if k in b[0]]
print(f"{a_path}: {len(a)} rows, {len(a[0])} cols")
print(f"{b_path}: {len(b)} rows, {len(b[0])} cols")
print(f"shared columns ({len(shared)}): {','.join(shared)}\n")

if len(a) != len(b):
    sys.exit(f"DIFFERENT ROW COUNT: {len(a)} vs {len(b)}")

diffs = 0
compared = 0
for ra, rb in zip(a, b):
    if ra["label"] != rb["label"]:
        print(f"row order differs: {ra['label']!r} vs {rb['label']!r}")
        diffs += 1
    for k in shared:
        compared += 1
        if ra[k] != rb[k]:
            print(f"  {ra['label']:<26} {k}: {ra[k]!r} != {rb[k]!r}")
            diffs += 1

print(f"\n{compared} values compared across {len(a)} models, "
      f"{diffs} difference(s)")
sys.exit(1 if diffs else 0)
