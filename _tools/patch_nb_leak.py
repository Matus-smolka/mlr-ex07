"""Remove the test-split peeking from the appended Task 6 cells.

Three edits, all inside the cells appended for task 6 (44-63); the 44 original
cells are untouched and verified byte-identical afterwards.

1. cell 47: `test` is now not just a method but not even in memory, so the
   sentence about it is out of date.
2. cell 49: claimed "the same proportions in all three splits" - a statement
   about the test split, in the cell that motivates the decisions. The
   class-balance table no longer prints a test column, so this sentence could
   not have come from the cell above it.
3. cell 58: results.csv no longer has test columns, so "the test columns are
   empty for all but the three evaluated at the end" describes a file that does
   not exist. Its unused `cols` list goes too.
"""
import json

NB = "Copy_of_ex07_student.ipynb"

EDITS = [
    ("Note that `test` is a *method*, `data.test()`, not an\narray: opening the test split is an action you can see in the code and count.",
     "Note that `test` is a *method*, `data.test()`, not an\narray - and `load_splits()` does not even read the test images into memory, so there\nis no attribute to reach past it. Opening the test split is an action you can see\nin the code and count, and `data.test_reads` counts it."),
    ("a ratio of 2.74, and the same proportions in all three\nsplits.",
     "a ratio of 2.74, and the same proportions in the\nvalidation split. The balance of the *test* split is not printed here on purpose:\nthis is the section that decides whether to mitigate anything, and a test number\nshown before a choice can inform it."),
    ('cols = ["label", "params", "optimizer", "sampling", "best_epoch",\n        "val_accuracy", "val_balanced_accuracy"]\n', ""),
    ('print(f"\\n{len(rows)} models compared on validation; the test columns are empty for all "\n      f"but the three evaluated at the end.")',
     'print(f"\\n{len(rows)} models compared on validation. No test column exists in this file "\n      f"at all - the test numbers come once, from test_metrics.json below.")'),
]


def main():
    nb = json.load(open(NB, encoding="utf-8"))
    done = []
    for i, cell in enumerate(nb["cells"]):
        if i < 44:
            continue
        src = "".join(cell["source"])
        new = src
        for old, repl in EDITS:
            if old in new:
                new = new.replace(old, repl)
                done.append((i, old[:48]))
        if new != src:
            cell["source"] = new.splitlines(keepends=True)
    missing = [old[:48] for old, _ in EDITS
               if old[:48] not in [frag for _, frag in done]]
    json.dump(nb, open(NB, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for i, frag in done:
        print(f"  cell {i}: patched  {frag!r}")
    for frag in missing:
        print(f"  NOT FOUND: {frag!r}")


if __name__ == "__main__":
    main()
