# MLR exercise 7, task 6 — a neural network in NumPy on blood-cell images

The Task 4 network of the notebook, extended: 28·28·3 = 2352 inputs, wider hidden
layers, eight outputs with a **softmax** and the **cross-entropy** loss. The
forward pass and the backward pass are NumPy and plain Python — no PyTorch,
TensorFlow, JAX, autograd library or scikit-learn model anywhere in the network,
the training or the optimizer. matplotlib is imported only to save two PNGs.

The instructions that came in the exercise ZIP (the private `mlr-ex07` repository,
inviting **MLRteacher**, the 48-hour rule) are kept unchanged in
`HANDOUT-README.md`.

## Setup from a fresh clone

```bash
uv sync                                    # or: pip install numpy matplotlib pytest
uv run python download_data.py             # 35 MB from Zenodo, MD5 checked, not committed
uv run pytest -q test_gradients.py         # the gradient check, ~0.2 s
uv run python train.py --epochs 50         # the whole comparison, ~25 min on a laptop CPU
uv run python make_report.py               # report.html, self-contained
```

Without `uv`, drop the `uv run` prefix. `download_data.py` skips the download when
the file is already present and its MD5 matches
`7053d0359d879ad8a5505303e11de1dc`; it refuses the file if the checksum differs.

`uv.lock` is **not** in this folder yet: `uv` is not installed on the machine this
was developed on. Run `uv lock` once and commit the result. `pyproject.toml` pins
only `numpy>=2.0`, `matplotlib>=3.8` and `pytest>=8.0`; everything was run with
NumPy 2.3.5 and Python 3.12.

## What each file is

| file | what it holds |
| --- | --- |
| `nn_numpy.py` | the network: `forward`, `backward`, `softmax`, `softmax_crossentropy`, `init_params`, Adam and SGD updates, the metrics |
| `download_data.py` | download + MD5 check, the official split, `to_columns`, `standardize`, the class-balance report |
| `train.py` | the four validation stages, the single test evaluation, all the output files |
| `test_gradients.py` | the gradient check, runnable with `pytest` |
| `make_report.py` | writes `report.html` from the results |
| `results.csv` | one row per compared model, with its validation accuracy and balanced accuracy. No timing column (see Reproducibility) and no test or train column: it is the selection record only |
| `results_lrgrid.csv` | the wider learning-rate search, stage 0 + stage A, **validation only** — the test split is never opened by it |
| `test_metrics.json` | the single test evaluation: network, softmax baseline, majority rule |
| `runtime.json` | measured total and per-model seconds, NumPy version, thread count |
| `learning_curves.png`, `confusion_matrix.png` | the two figures |
| `Copy_of_ex07_student.ipynb` | my notebook, Tasks 1–5 mine, Task 6 appended (cells 44–63) |
| `run_log.txt` | the console log of the full run, including the line where the test split is opened |

## Where the conventions of Task 4 are

`nn_numpy.forward` and `nn_numpy.backward` are the Task 4 loop with columns of a
batch instead of one sample: the lists `W[k]`, `b[k]` indexed by the layer number
of the lecture (index 0 unused `None`), `W[k]` of shape
`(sizes[k], sizes[k-1])`, `for k in range(1, K+1)` forward and
`for k in range(K, 0, -1)` backward, ReLU in the hidden layers, identity in the
last one. The single line that had to change is the start of the backward pass:
Task 4's `grad_a = y_hat - y` becomes `grad_a = (proba - y) / N`.

Because a batch is used, the loss is **averaged** over the batch and the bias
gradient is the **sum** over its columns — that sum already *is* the average,
since every column of `grad_a` carries its `1/N`. Dividing by N a second time is
the trap; `test_bias_gradient_is_summed_over_the_batch` pins it.

## Reproducibility

`SEED = 0`, as in Task 3 and Task 4. Parameters and shuffling draw from two
separate `np.random.default_rng` streams, so a change to one does not silently
shift the other. BLAS threads are fixed before NumPy is imported
(`OMP_NUM_THREADS` and friends, `min(8, cpus)`), because a BLAS reduction splits
differently with a different thread count and that changes the last digits.

To check it:

```bash
python train.py --epochs 50 && copy results.csv run1.csv
python train.py --epochs 50 && fc run1.csv results.csv       # or: diff
```

**Measured, on this machine:** two independent runs of `python train.py --epochs 50`
gave the same validation accuracy and balanced accuracy for all ten models, to all
16 digits NumPy prints (e.g. `0.8761682242990654` both times), and the same best
epoch for every model. The only thing that differed was the wall-clock column —
61.34 s against 63.85 s for the same model — which is not a reproducibility
failure but did make `diff` non-empty. **The timing column was therefore moved out
of `results.csv` into `runtime.json`**, so `diff run1.csv results.csv` is now empty
and means exactly what it says: `results.csv` contains only quantities two
identical runs reproduce. Expect the same agreement on the same machine, NumPy and
thread count; across a different BLAS or CPU the last digits can move and a
different epoch can end up best. Record both runtimes from `runtime.json`.

## Order of decisions, and the one look at the test set

Every choice is made on the validation split, in this order, and the test split
is opened once at the end:

0. **learning rate** per optimizer, over an 8-epoch probe;
1. **optimizer** (`adam`/`sgd`) × **sampling** (`plain`/`balanced`), at `[2352, 256, 8]`;
2. **layer sizes** with the winning configuration, against the softmax baseline `[2352, 8]`;
3. **sampling** re-checked at the size that won;
4. `data.test()` — once — for the chosen network, the softmax baseline and the majority rule.

Ranking is by `train.py:selection_key`: validation accuracy in half-percentage-point
bands, balanced accuracy breaking ties. Models and the kept epoch are both ranked
this way. No training accuracy is reported as a result anywhere — `results.csv` has
no train column and no test column, and the only training quantity that appears is
the loss in the left panel of the learning curves, labelled "not a result".

### How the test split is kept out, and the two ways it leaked anyway

`Splits` **does not hold the test images.** `load_splits()` reads train and val,
computes the standardization vectors from train, and hands `Splits` the *path*; the
test arrays are read from the file inside `Splits.test()`, on its first call. There
is therefore no attribute to read past the guard, and `test()` is the only door.
`test_metrics.json` records `test_read_count`, and `train.py` now **raises** rather
than merely printing if it is anything but 1 — a run that spent the test split twice
is not allowed to write artifacts. `--select-only` has the mirror-image gate: it
raises if the count is not 0.

This replaced a weaker design that had a bypass, and the bypass was used — three
times, all of them found by grepping after the fact, none of them visible while
writing them:

1. `report_class_balance()` printed a **test** column. It read `splits._y_test`
   directly, which is not private in Python, and that table is printed at the top of
   the run — in front of the step that decides whether to mitigate the imbalance and
   which metric to rank by. The column is gone; the function reports train and val
   only, and the test class counts appear in `report.html` section 4 as the row sums
   of the confusion matrix, after the one evaluation.
2. The startup line printed `data._y_test.size`. Only the count 3421, which changes
   nothing — which is exactly why it got written, and why an underscore is not a
   guard.
3. `_tools/inspect_npz.py`, the first thing run on the downloaded file at all,
   printed the label distribution of `("train", "val", "test")`. This one is the real
   finding, because it happened during exploration rather than in shipped code, and
   before the metric and the mitigation question were settled. Labels only — no test
   image was read and no test score existed — and both questions were then answered
   on validation (balanced sampling was trained against plain sampling and lost, so
   the mitigation choice does not rest on the peek), and the counts are published
   MedMNIST metadata. It is still exactly what requirement 6 is about, so it is
   disclosed in `report.html` section 1 and the script now prints train and val only.

None of the three fed a gradient or a fit, and the single evaluation at the end is
unchanged by them, but "the test split is reachable through one door"
was not true while both existed. Deleting the attribute is what makes the claim
checkable: `grep -n "_y_test\|\.test()" *.py` now finds the accesses only inside
`Splits.test()` and after it in `train.py`.

### A choice re-examined without touching the test set

The first learning-rate probe put both winners on the edge of the rates it tested
(Adam at the lowest, 3e-4; SGD at the highest, 0.03), which is a sign the grid was
too narrow and that Adam may have been handicapped. So the grid was widened and
stage A repeated — **on validation only**:

```bash
python train.py --select-only --epochs 30 --probe-epochs 12 \
  --adam-etas 1e-4,3e-4,1e-3,3e-3 --sgd-etas 0.03,0.05,0.1 --out results_lrgrid.csv
```

`--select-only` runs stage 0 and stage A and stops before any test evaluation, then
prints `the test split was opened 0 time(s)`. That constraint is the point: the
graded experiment had already read the test split, so re-picking a configuration
and looking at the test set again would be the exact leakage requirement 6 forbids.
The result is reported as a check in `report.html`, not used to re-select.

It corrected Adam's rate to 1e-3 (a longer probe is what found it, so the concern
was real) but did **not** change the ranking: adam-plain 0.8721 / 0.8501 against
sgd-plain 0.8808 / 0.8617. SGD still wins, so the submitted configuration now
rests on better evidence. Class-balanced sampling lost again, independently.
SGD at 0.1 diverges (0.69), so 0.03–0.05 is a genuine basin and not a lucky spike.

### Imbalance: tested, not assumed

Class-balanced sampling was trained against plain sampling at two architectures and
with both optimizers. It lost every time, and was marginally worse even on balanced
accuracy (0.8568 against 0.8589). At a 2.74 ratio the rare classes are already seen
often enough. The imbalance is handled by measuring it — balanced accuracy,
per-class recall, the confusion matrix — not by altering the training data.

## Commands as actually run, with measured times

```bash
python -m pytest -q test_gradients.py     # 11 passed in 0.17 s
python download_data.py                   # MD5 matches, class balance printed
python train.py --smoke                   # 5.95 s, pipeline check on a subset
python train.py --epochs 50               # 482 s / 492 s / 484 s on three runs
```

Results of the graded run, all on 8 BLAS threads of a 16-thread laptop CPU with an
RTX 4060 deliberately unused (see the note in the report about why a GPU buys
nothing here): chosen network **0.8652** test accuracy / **0.8457** balanced,
softmax baseline 0.7995 / 0.7739, majority rule 0.1947 / 0.1250.

## Two bugs this code went through, kept in the open

1. **The cross-entropy added `logsumexp` once per class instead of once per
   sample** (`np.sum(lse - y*logits)` over an `(n_classes, N)` array), inflating
   the loss about eightfold. The gradient was unaffected, so training looked
   normal and only the reported loss was wrong. Caught because the all-`-1000`
   logit vector produced a *negative* cross-entropy. Fixed in
   `softmax_crossentropy`; `test_the_loss_check_fails_for_the_logsumexp_bug` keeps
   the buggy version around so the check is known to be able to fail.
2. **The bias gradient was divided by N twice**, making every bias learn N times
   too slowly. Caught on the first run of `test_gradients.py`, by
   `test_bias_gradient_is_summed_over_the_batch`.

## Reading the result cautiously

The network beats the softmax baseline and the majority-class rule; how much by
is in `test_metrics.json` and `report.html`. The dataset is one public benchmark,
one stain protocol, one already-cropped 28×28 crop per cell. `report.html` section
6 lists what the numbers cannot establish — no calibration, no abstention option,
correlated samples within patients, and two classes with few enough test images
that their recalls carry real uncertainty. Nothing here supports a claim about
clinical use.

`REVIEW.md` is the student's own work and is deliberately not written here.
