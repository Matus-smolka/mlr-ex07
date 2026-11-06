"""Append the Task 6 cells to the student notebook, leaving the other cells alone."""
import json
import uuid

NB = "Copy_of_ex07_student.ipynb"

MD = "markdown"
CODE = "code"

CELLS = [
(MD, """## Task 6 — my network, deeper, on blood-cell images

Everything below is the **Task 4 script extended**, not a new network: the lists
`W[k]`, `b[k]` indexed by the layer number of the lecture, one sample per
column, `for k in range(1, K + 1)` forward and `for k in range(K, 0, -1)`
backward, ReLU in the hidden layers and the identity in the last one. Three
things change against Task 4:

| | Task 4 | Task 6 |
| --- | --- | --- |
| input | 1 number | 28 · 28 · 3 = 2,352 numbers |
| output | 1 neuron, squared error | 8 neurons, **softmax + cross-entropy** |
| layers | `[1, 128, 1]` | e.g. `[2352, 512, 256, 8]` |

The only line of the backward pass that had to change is its starting point:
Task 4 began with `grad_a = y_hat - y`, the softmax with cross-entropy begins
with `grad_a = (p - onehot) / N`. The loop under it is Task 4's loop, and with a
batch the bias gradient is **summed** over the columns while the loss is
**averaged** over them.

NumPy and plain Python throughout — no PyTorch, TensorFlow, JAX, autograd or
scikit-learn in the forward pass or the backward pass. matplotlib appears only
in the two figures.

**The files.** `nn_numpy.py` (network, softmax/cross-entropy, backward, Adam),
`download_data.py` (download + MD5 + the official split + standardization),
`train.py` (the comparison and the single test pass), `test_gradients.py`
(`pytest`), `make_report.py` (`report.html`)."""),

(MD, """### Getting the files in Colab

The four modules have to sit next to the notebook (that is how the ZIP folder is
laid out). Either upload them with the file manager on the left, or uncomment
the line below and fill in your repository. The 35 MB `bloodmnist.npz` is
deliberately **not** committed — `download_data.py` fetches it and checks its
MD5.""",),

(CODE, """# Uncomment for Colab, then run:
# from google.colab import drive
# drive.mount('/content/drive')
# %cd /content/drive/MyDrive/mlr-ex07

# !git clone https://github.com/<your-user>/mlr-ex07.git && %cd mlr-ex07

import os, sys
sys.path.append(os.getcwd())          # so `import nn_numpy` works from the notebook
!ls *.py
!python --version && python -c "import numpy; print('numpy', numpy.__version__)"
!grep -rn -E "import (torch|tensorflow|jax|sklearn|autograd)|from (torch|tensorflow|jax|sklearn|autograd)" nn_numpy.py train.py download_data.py || echo "no forbidden library in the network, the training or the optimizer"
"""),

(MD, """### 6a. The data, and how its classes are spread

The official split is used as it is. The one thing to look at before training is
the class balance — a classifier that sees 98% of its samples in one class is a
different problem from this one.

`download_data.py` checks the MD5 `7053d0359d879ad8a5505303e11de1dc` of the
file, then `load_splits()` standardizes with statistics computed **on the
training images only**. Note that `test` is a *method*, `data.test()`, not an
array: opening the test split is an action you can see in the code and count.""",),

(CODE, """import numpy as np
import nn_numpy as nn
from download_data import load_splits, report_class_balance, BLOOD_CLASSES, N_CLASSES

data = load_splits()
print(report_class_balance(data))

print()
print("one sample per column:", data.train.shape, "= 2352 numbers x", data.train.shape[1], "images")
print("train standardized  : mean %.3f std %.3f" % (data.train.mean(), data.train.std()))
print("val on the same scale: mean %.3f std %.3f  (never used to compute them)" % (data.val.mean(), data.val.std()))
"""),

(MD, """**What the balance says.** The largest class is 19.5% of the training images and
the smallest 7.1% — a ratio of 2.74, and the same proportions in all three
splits. That is uneven, but it is nowhere near the case of one class holding the
overwhelming majority of the samples, so it does not by itself demand a
mitigation. Instead of assuming one, task 6c **tests** one on the validation
split: the same network trained with class-balanced sampling, where every batch
draws its class uniformly instead of in the natural proportions.

*Your note: did balanced sampling help, and what does that tell you? (This goes
in your `REVIEW.md`, which you write yourself.)*""",),

(MD, """### 6b. The softmax and the cross-entropy, checked before they are trained with

Two things to verify here, both in the style of the gradient check of Task 2 and
Task 3.

First, **numerical safety**. A naive `exp(logits) / sum(exp(logits))` overflows
as soon as a logit is large, and `log(0)` is a real risk in the loss. Subtracting
the column maximum keeps the largest exponent at `exp(0) = 1`, and computing the
loss through log-sum-exp means the logarithm is taken of a sum that is at least
1 — never of 0. The task sheet's own test is the logits `(1000, 0, -1000)`.

Second, the **gradient of the loss itself**, against a central difference. This
check earned its place: written the obvious way,
`np.sum(logsumexp - y * logits)`, the loss adds `logsumexp` once per *class*
instead of once per *sample*, inflating it about eightfold. The gradient is
unaffected, so the network trains and the loss curve is simply wrong. An
all-`-1000` logit vector exposing a *negative* cross-entropy is what gave it
away.""",),

(CODE, """# 1. The extreme logits from the task sheet. The loss must be finite, and a
#    cross-entropy cannot be negative.
for row in ([1000.0, 0.0, -1000.0], [1000.0, 1000.0, 1000.0],
            [-1000.0, -1000.0, -1000.0], [1e300, 0.0, -1e300]):
    logits = np.array(row).reshape(-1, 1)
    line = []
    for target in range(3):
        y = np.zeros((3, 1)); y[target, 0] = 1.0
        loss, grad = nn.softmax_crossentropy(logits, y)
        assert np.isfinite(loss) and np.all(np.isfinite(grad)), "not finite!"
        assert loss >= -1e-12, "a cross-entropy cannot be negative"
        line.append(f"class {target}: {loss:8.4f}")
    print(f"logits {str(row):<28}  " + "   ".join(line))

# 2. Equal logits must give exactly log(n_classes): -log(1/8).
uniform = nn.softmax_crossentropy(np.zeros((8, 4)), nn.one_hot(np.arange(4), 8))[0]
print(f"\\nuniform logits -> loss {uniform:.6f}, log(8) = {np.log(8):.6f}")

# 3. The safe log-sum-exp route against the textbook -log p.
gen = np.random.default_rng(0)
logits = gen.standard_normal((N_CLASSES, 200)) * 8.0
targets = gen.integers(0, N_CLASSES, 200)
Y = nn.one_hot(targets, N_CLASSES)
p = nn.softmax(logits)
reference = -np.mean(np.log(p[targets, np.arange(200)]))
mine, grad_a = nn.softmax_crossentropy(logits, Y)
print(f"log-sum-exp loss {mine:.8f}   -log p loss {reference:.8f}   "
      f"difference {abs(mine - reference):.2e}")
"""),

(MD, """### 6c. The gradient test: every analytic gradient against a central difference

The same referee as in Task 2 and Task 3, now on a network with the
softmax-and-cross-entropy output and three layers, so that the gradient has to be
passed down through two hidden layers. Weights and biases of every layer, all of
them, `float64`, at a threshold far tighter than training would tolerate.

Run the whole thing from a terminal with `pytest -q test_gradients.py`. Note the
last two tests: a **deliberately broken** backward pass must *fail* the same
comparison. A check that cannot fail checks nothing — the missing ReLU mask below
is caught at about `1e7`, against `1e-7` for the correct one.""",),

(CODE, """!pytest -q test_gradients.py

# and the same check in the open, on a tiny network:
from test_gradients import tiny_problem, analytic, numeric, largest_relative_difference
W, b, X, Y = tiny_problem()
_, gW_a, gb_a = analytic(W, b, X, Y)
gW_n, gb_n = numeric(W, b, X, Y)
worst, where = largest_relative_difference(gW_a, gb_a, gW_n, gb_n)
n_checked = sum(w.size for w in W[1:]) + sum(v.size for v in b[1:])
print(f"\\nnotebook version: {n_checked} parameters, largest relative difference {worst:.2e} (at {where})")

# the gradient of a[k] always has the shape of a[k], and grad of W[k] the shape of W[k]
for k in range(1, len(W)):
    assert gW_a[k].shape == W[k].shape and gb_a[k].shape == b[k].shape
print("every gradient has the shape of its parameter")
"""),

(MD, """### 6d. A trial run before the full comparison

One network, a few epochs, to see the loss fall and the validation accuracy
climb before spending time on the comparison. This is the Task 4 loop with the
columns of a batch instead of one sample; the update is Adam, written out in
`nn_numpy.adam_step`.""",),

(CODE, """import time
from train import train_one, evaluate

sizes = [2352, 256, 8]
t0 = time.perf_counter()
out = train_one(sizes, data, optimizer="adam", eta=1e-3, epochs=5,
                label="trial 2352-256-8")
print(f"\\n{out['row']['params']:,} parameters, {time.perf_counter() - t0:.1f} s")
print(f"best epoch {out['best_epoch']}: val accuracy {out['row']['val_accuracy']:.4f}, "
      f"val balanced accuracy {out['row']['val_balanced_accuracy']:.4f}")
print("the test split has been opened", data.test_reads, "time(s) so far")
"""),

(MD, """### 6e. The comparison, and the one look at the test split

`train.py` does the comparison in the order the task asks for, each choice made
on validation:

0. the **learning rate** of each optimizer, over a short probe;
1. the **optimizer** (`adam` / `sgd`) and the **sampling** (`plain` /
   `balanced`), at one middle size;
2. the **layer sizes**, with the winning configuration, against the **softmax
   baseline** `[2352, 8]` (no hidden layer);
3. the **sampling confirmed** again at the size that won, so the decision does
   not rest on one architecture;
4. only then `data.test()` — once — for the chosen network, the softmax baseline
   and the majority-class rule.

Everything is ranked by `selection_key`: validation accuracy in
half-percentage-point bands, balanced accuracy breaking ties. Run it from a
terminal (`py train.py`, about 25 minutes) or the line below.""",),

(CODE, """# The full comparison. ~25 minutes on a laptop CPU; --smoke is the two-minute version.
# !python train.py --epochs 50
# !python train.py --smoke
"""),

(CODE, """import csv
cols = ["label", "params", "optimizer", "sampling", "best_epoch",
        "val_accuracy", "val_balanced_accuracy", "seconds"]
with open("results.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
print(f"{'model':<26}{'parameters':>11}{'optimizer':>10}{'sampling':>10}"
      f"{'epoch':>7}{'val acc':>10}{'val bal acc':>13}{'seconds':>9}")
for r in rows:
    print(f"{r['label']:<26}{int(r['params']):>11,}{r['optimizer']:>10}{r['sampling']:>10}"
          f"{r['best_epoch']:>7}{float(r['val_accuracy']):>10.4f}"
          f"{float(r['val_balanced_accuracy']):>13.4f}{float(r['seconds']):>9.1f}")
print(f"\\n{len(rows)} models compared on validation; the test columns are empty for all "
      f"but the three evaluated at the end.")
"""),

(MD, """The two figures. In the learning curves, the **left panel is the training loss and
is not a result** — it is there to document the runs. The epoch was taken from
the validation panels.""",),

(CODE, """from IPython.display import Image
Image("learning_curves.png")
"""),

(CODE, """Image("confusion_matrix.png")
"""),

(CODE, """import json
m = json.load(open("test_metrics.json"))
print("chosen on validation:", m["selected_on_validation"], "\\n")
for model in (m["network"], m["softmax_baseline"], m["majority_rule"]):
    print(f"{model['name']:<45} accuracy {model['accuracy']:.4f}   "
          f"balanced accuracy {model['balanced_accuracy']:.4f}")
print(f"\\nthe test split was opened {m['test_read_count']} time(s) in total")
print("\\nper-class recall of the chosen network:")
for name, value in m["network"]["per_class_recall"].items():
    print(f"  {name:<22} {value:.3f}")
"""),

(MD, """### 6f. Reading the result honestly

Before writing anything up, answer these to yourself — they are the eight points
`REVIEW.md` is built on, and the report should not say more than the figures
support:

* How much better is the network than the softmax baseline, and than the
  majority-class rule? Is the margin larger than the differences between the
  network sizes you compared?
* Do the learning curves actually support the epoch you kept, or is the curve
  still moving at that point? Is validation turning up while training keeps
  falling (overfitting)?
* Which classes are confused, and does the confusion have a biological or a
  pixel-level explanation?
* Was anything at all chosen after the test split was opened? The run log prints
  one line when it happens, and `test_metrics.json` records the count.
* Did you rerun the whole thing and diff the two `results.csv` files? Record both
  runtimes and the command.
* What can this not say about using such a classifier on patients? (One crop per
  cell, one stain protocol, no calibration, no abstention, correlated samples.)

*Your `REVIEW.md` is your own work — do not have the agent write it.*"""),
]


def new_cell(kind, text):
    src = text.strip("\n").splitlines(keepends=True)
    if kind == CODE:
        return {"cell_type": "code", "execution_count": None, "id": str(uuid.uuid4())[:8],
                "metadata": {}, "outputs": [], "source": src}
    return {"cell_type": "markdown", "id": str(uuid.uuid4())[:8],
            "metadata": {}, "source": src}


def main():
    with open(NB, encoding="utf-8") as f:
        nb = json.load(f)
    before = len(nb["cells"])
    nb["cells"].extend(new_cell(kind, text) for kind, text in CELLS)
    with open(NB, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"{NB}: {before} -> {len(nb['cells'])} cells "
          f"(+{len(CELLS)}: {sum(1 for k, _ in CELLS if k == CODE)} code, "
          f"{sum(1 for k, _ in CELLS if k == MD)} markdown)")


if __name__ == "__main__":
    main()
