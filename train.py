"""train.py — the Task 4 network, deeper, classifying eight blood-cell types.

What is chosen where
--------------------
The rule the task sets: **every** choice is made on the validation split, and
the test split is opened once, at the end, for the model that was already
chosen. Concretely, in this file:

  stage 0  the learning rate of each optimizer, on validation (a short probe);
  stage A  the optimizer and the sampling, on validation, at one middle size;
  stage B  the layer sizes, on validation, with the winning configuration;
  stage C  the sampling confirmed again at the winning size;
  final    the test split, once, for the chosen network, the softmax baseline
           and the majority-class rule.

No training accuracy is reported as a result anywhere: the training loss only
ever goes into the learning curves, the numbers in results.csv are validation
numbers, and the numbers in test_metrics.json come from the single final pass.
`Splits.test()` prints a line the first time it is called and a warning after
it, so a leak would be visible in the run log.

Run:   py train.py            (the full comparison, then the single test pass)
       py train.py --smoke    (a couple of minutes on a subset, pipeline check)
NumPy and plain Python only; matplotlib is imported for the two PNGs and never
for anything that computes a gradient.
"""

import os

# Fixed before NumPy is imported: a BLAS reduction splits differently with a
# different thread count, so fixing it is part of getting the same numbers twice.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, str(min(8, os.cpu_count() or 4)))

import sys                                               # noqa: E402

# This project's own modules (nn_numpy, download_data). Appended to the END of
# sys.path rather than inserted at the front, so that if anything ever shadows a
# standard library or NumPy name in this folder, the real one still wins.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)

import argparse                                          # noqa: E402
import csv                                               # noqa: E402
import json                                              # noqa: E402
import time                                              # noqa: E402

import numpy as np                                       # noqa: E402

import nn_numpy as nn                                     # noqa: E402
from download_data import (BLOOD_CLASSES, N_CLASSES, load_splits,   # noqa: E402
                           report_class_balance)

SEED = 0                     # one seed for the whole script, as in Task 3 and Task 4
BATCH = 256                  # samples per column-block; the loss is averaged over it


def selection_key(accuracy, balanced_accuracy):
    """The number everything is ranked by on validation: epoch, size, optimizer.

    Validation accuracy leads, rounded to half-percentage-point bands; within a
    band the higher balanced accuracy wins. Why not balanced accuracy alone: the
    two smallest validation classes have 122 images each, so one image moves a
    per-class recall by 0.8% and those two classes would decide the architecture
    on their own noise. 1712 validation images make overall accuracy the steadier
    estimate, while the tiebreak keeps a model that is good overall and poor on
    the rare classes from winning quietly.

    Fixed here, before the test split is opened, which is what makes the choice
    honest rather than lucky.
    """
    return (round(accuracy * 200.0), balanced_accuracy)

# the architectures compared, all with 28*28*3 = 2352 inputs and 8 outputs
BASELINE = [2352, N_CLASSES]                  # the softmax baseline, no hidden layer
CANDIDATES = [[2352, 128, N_CLASSES],
              [2352, 256, N_CLASSES],
              [2352, 256, 128, N_CLASSES],
              [2352, 512, 256, N_CLASSES]]
PROBE_SIZE = [2352, 256, N_CLASSES]           # the middle size used to pick the config


# --------------------------------------------------------------------------
# one training run
# --------------------------------------------------------------------------
def minibatch_indices(n, batch, rng):
    """One epoch of column indices: the whole training set, in a shuffled order."""
    return rng.permutation(n)


def balanced_batch_indices(y_train, batch, rng, by_class):
    """One batch drawn class by class: a class at random, then a sample of it.

    The mitigation for the imbalance, if validation says one is wanted. Each of
    the `batch` columns picks a class uniformly out of the eight and then one of
    that class's images, so every class is offered as often as every other no
    matter how many images it has. The network sees the same images more often;
    nothing about the images themselves is invented.
    """
    return np.array([by_class[c][rng.integers(len(by_class[c]))]
                     for c in rng.integers(N_CLASSES, size=batch)])


def rank_row(row):
    """The selection_key of one results row — how every stage picks its winner."""
    return selection_key(row["val_accuracy"], row["val_balanced_accuracy"])


def evaluate(W, b, X, labels):
    """Accuracy and balanced accuracy of the network on a split. No fitting here."""
    pred = nn.predict(W, b, X)
    return {"accuracy": nn.accuracy(labels, pred),
            "balanced_accuracy": nn.balanced_accuracy(labels, pred, N_CLASSES),
            "confusion": nn.confusion_matrix(labels, pred, N_CLASSES),
            "pred": pred}


def train_one(sizes, data, *, optimizer="adam", eta=0.001, sampling="plain",
              epochs=40, batch=BATCH, seed=SEED, label="", verbose=True):
    """Train one network and return its best-on-validation parameters and history.

    The loop is the Task 4 loop with a batch of samples in the columns: forward,
    cross-entropy, backward, update. `best` holds a copy of the parameters at the
    epoch with the best **validation** balanced accuracy — which is how the epoch
    count is chosen on validation rather than guessed.
    """
    gen = np.random.default_rng(seed)          # the parameters
    rng = np.random.default_rng(seed + 1)      # the shuffling, a different stream
    W, b = nn.init_params(list(sizes), gen)
    state = {}
    by_class = {c: np.flatnonzero(data.y_train == c) for c in range(N_CLASSES)}
    n_train = data.y_train.size

    step = 0
    history, best = [], {"key": (-1, -1.0)}
    t0 = time.perf_counter()
    for epoch in range(epochs):
        epoch_losses = []
        if sampling == "plain":
            order = minibatch_indices(n_train, batch, rng)
            starts = range(0, n_train - n_train % batch, batch)
            for s in starts:
                idx = order[s:s + batch]
                Xb, Yb = data.train[:, idx], nn.one_hot(data.y_train[idx], N_CLASSES)
                a, h = nn.forward(W, b, Xb)
                loss, grad_a = nn.softmax_crossentropy(h[-1], Yb)
                gW, gb = nn.backward(W, a, h, grad_a)
                step += 1
                if optimizer == "adam":
                    nn.adam_step(W, b, gW, gb, state, step, eta=eta)
                else:
                    nn.sgd_step(W, b, gW, gb, eta=eta, momentum=0.9, state=state)
                epoch_losses.append(loss)
        else:                                   # class-balanced sampling
            for _ in range(n_train // batch):
                idx = balanced_batch_indices(data.y_train, batch, rng, by_class)
                Xb, Yb = data.train[:, idx], nn.one_hot(data.y_train[idx], N_CLASSES)
                a, h = nn.forward(W, b, Xb)
                loss, grad_a = nn.softmax_crossentropy(h[-1], Yb)
                gW, gb = nn.backward(W, a, h, grad_a)
                step += 1
                if optimizer == "adam":
                    nn.adam_step(W, b, gW, gb, state, step, eta=eta)
                else:
                    nn.sgd_step(W, b, gW, gb, eta=eta, momentum=0.9, state=state)
                epoch_losses.append(loss)

        val = evaluate(W, b, data.val, data.y_val)
        history.append({"epoch": epoch + 1,
                        "train_loss": float(np.mean(epoch_losses)),
                        "val_accuracy": val["accuracy"],
                        "val_balanced_accuracy": val["balanced_accuracy"]})
        key = selection_key(val["accuracy"], val["balanced_accuracy"])
        improved = key > best["key"]
        if improved:                      # keep the parameters of the best epoch
            best = {"key": key, "epoch": epoch + 1,
                    "accuracy": val["accuracy"],
                    "balanced_accuracy": val["balanced_accuracy"],
                    "W": [None] + [w.copy() for w in W[1:]],
                    "b": [None] + [v.copy() for v in b[1:]]}
        if verbose:
            print(f"    epoch {epoch + 1:>3}/{epochs}  loss {history[-1]['train_loss']:.4f}"
                  f"  val acc {val['accuracy']:.4f}  val bal acc "
                  f"{val['balanced_accuracy']:.4f}{'  <- best' if improved else ''}")

    seconds = time.perf_counter() - t0
    row = {"label": label or str(list(sizes)), "sizes": str(list(sizes)),
           "params": nn.count_params(W, b), "optimizer": optimizer, "eta": eta,
           "sampling": sampling, "epochs_run": epochs,
           "best_epoch": best["epoch"],
           "val_accuracy": best["accuracy"],
           "val_balanced_accuracy": best["balanced_accuracy"],
           "seconds": round(seconds, 2)}
    return {"W": best["W"], "b": best["b"], "history": history,
            "best_epoch": best["epoch"], "row": row,
            "final": {"accuracy": evaluate(W, b, data.val, data.y_val)["accuracy"]}}


# --------------------------------------------------------------------------
# the stages
# --------------------------------------------------------------------------
ADAM_ETAS = [3e-4, 1e-3, 3e-3]
SGD_ETAS = [3e-3, 1e-2, 3e-2]


def probe_learning_rates(data, epochs, grids=None):
    """Stage 0: the learning rate of each optimizer, on validation only."""
    grids = grids or {"adam": ADAM_ETAS, "sgd": SGD_ETAS}
    print(f"\n=== stage 0: the learning rate, on validation "
          f"({epochs} epochs at {PROBE_SIZE}) ===")
    chosen = {}
    for optimizer, etas in (("adam", grids["adam"]), ("sgd", grids["sgd"])):
        results = []
        for eta in etas:
            out = train_one(PROBE_SIZE, data, optimizer=optimizer, eta=eta,
                            epochs=epochs, label=f"probe {optimizer} eta={eta}",
                            verbose=False)
            results.append((out["row"]["val_balanced_accuracy"], eta))
            print(f"  {optimizer} eta={eta:<7g}  val bal acc "
                  f"{out['row']['val_balanced_accuracy']:.4f}")
        chosen[optimizer] = max(results)[1]
        print(f"  -> {optimizer}: eta = {chosen[optimizer]:g}")
    return chosen


def run_stage(data, configs, epochs):
    rows, store = [], {}
    for cfg in configs:
        label = cfg["label"]
        print(f"\n  --- {label}: {cfg['sizes']}  {cfg['optimizer']} eta={cfg['eta']}  "
              f"{cfg['sampling']}  {epochs} epochs ---")
        out = train_one(cfg["sizes"], data, optimizer=cfg["optimizer"], eta=cfg["eta"],
                        sampling=cfg["sampling"], epochs=epochs, label=label,
                        verbose=(epochs <= 12) or cfg.get("verbose", False))
        rows.append(out["row"])
        store[label] = out
        print(f"  {label}: best at epoch {out['best_epoch']}  "
              f"val acc {out['row']['val_accuracy']:.4f}  "
              f"val bal acc {out['row']['val_balanced_accuracy']:.4f}  "
              f"({out['row']['seconds']}s)")
    return rows, store


def majority_rule(data, labels_pred_to):
    """Always predict the most frequent training class."""
    top = int(np.bincount(data.y_train, minlength=N_CLASSES).argmax())
    pred = np.full(labels_pred_to.shape, top)
    return {"accuracy": nn.accuracy(labels_pred_to, pred),
            "balanced_accuracy": nn.balanced_accuracy(labels_pred_to, pred, N_CLASSES),
            "confusion": nn.confusion_matrix(labels_pred_to, pred, N_CLASSES),
            "majority_class": BLOOD_CLASSES[top]}


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
def write_results_csv(rows, path="results.csv"):
    """One row per compared model, with its validation accuracy and balanced accuracy.

    No wall-clock column here on purpose: the measured seconds live in
    runtime.json. Timing is the one thing two identical runs cannot reproduce,
    and keeping it out of results.csv is what lets `diff run1.csv results.csv`
    come out empty and mean exactly what it says.

    Two columns were also dropped on the way: `test_accuracy` and
    `test_balanced_accuracy` were always empty, because the test pass runs once
    and only for the chosen model — so every row carried two blanks, and the one
    row that could have been filled would have made this file a comparison partly
    on test and partly on validation. results.csv is now the selection artifact
    and nothing else: validation numbers only, which is also exactly what the task
    asks of it. The single test evaluation lives in `test_metrics.json`, written
    after `data.test()`, and the two files never feed each other.
    """
    fields = ["label", "sizes", "params", "optimizer", "eta", "sampling",
              "epochs_run", "best_epoch", "val_accuracy", "val_balanced_accuracy"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out.pop("seconds", None)                  # timings stay in runtime.json only
            out.pop("test_accuracy", None)            # test is not a selection number
            out.pop("test_balanced_accuracy", None)
            writer.writerow(out)


def plot_learning_curves(runs, path="learning_curves.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for label, out in runs.items():
        hist = out["history"]
        ep = [h["epoch"] for h in hist]
        axes[0].plot(ep, [h["train_loss"] for h in hist], label=label)
        axes[1].plot(ep, [h["val_balanced_accuracy"] for h in hist], label=label)
        axes[2].plot(ep, [h["val_accuracy"] for h in hist], label=label)
    axes[0].set_xlabel("epoch"), axes[0].set_ylabel("mean cross-entropy in the epoch")
    axes[0].set_yscale("log"), axes[0].set_title("training loss (not a result)")
    axes[1].set_xlabel("epoch"), axes[1].set_ylabel("validation balanced accuracy")
    axes[1].set_title("chosen on validation: balanced accuracy")
    axes[2].set_xlabel("epoch"), axes[2].set_ylabel("validation accuracy")
    axes[2].set_title("chosen on validation: accuracy")
    for ax in axes:
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_confusion(cm, path="confusion_matrix.png", title="Test split"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(N_CLASSES), [c.replace(" ", "\n") for c in BLOOD_CLASSES],
                  fontsize=8, rotation=90)
    ax.set_yticks(range(N_CLASSES), BLOOD_CLASSES, fontsize=8)
    ax.set_xlabel("predicted class"), ax.set_ylabel("true class")
    ax.set_title(f"{title}: rows normalized (per-class recall)")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="w" if norm[i, j] < 0.55 else "k")
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="a small subset and few epochs, to check the pipeline")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--select-only", action="store_true",
                    help="run stage 0 and stage A and stop: validation only, the test "
                         "split is never opened. For re-examining a choice made in an "
                         "experiment whose test split has already been read.")
    ap.add_argument("--adam-etas", default=None,
                    help="comma-separated learning rates for the Adam probe")
    ap.add_argument("--sgd-etas", default=None,
                    help="comma-separated learning rates for the SGD probe")
    ap.add_argument("--probe-epochs", type=int, default=None,
                    help="epochs in the stage 0 probe (default 8)")
    args = ap.parse_args()

    t_start = time.perf_counter()
    data = load_splits()

    if args.smoke:
        keep = np.random.default_rng(SEED).permutation(data.train.shape[1])[:1200]
        data.train, data.y_train = data.train[:, keep], data.y_train[keep]
        keep_v = np.random.default_rng(SEED).permutation(data.val.shape[1])[:400]
        data.val, data.y_val = data.val[:, keep_v], data.y_val[keep_v]
    # 8 epochs, not 3: a learning rate judged on 3 epochs mostly measures which
    # one moves fastest at the start, which is not the question a 50-epoch run asks.
    grids = {"adam": [float(x) for x in args.adam_etas.split(",")] if args.adam_etas
             else ADAM_ETAS,
             "sgd": [float(x) for x in args.sgd_etas.split(",")] if args.sgd_etas
             else SGD_ETAS}
    n_probe = args.probe_epochs or (8 if not args.smoke else 3)
    n_stage = args.epochs or (2 if args.smoke else 40)

    print(report_class_balance(data))
    print(f"\nseed = {SEED}, batch = {BATCH}, smoke = {args.smoke}")
    # No test number here, not even its size. An earlier version printed
    # `data._y_test.size`, which reached past the test() guard for the sake of one
    # number that changes nothing — and a guard with exceptions people make for
    # harmless reasons stops being a guard.
    print(f"train {data.train.shape[1]} images, val {data.val.shape[1]} images, "
          f"test held back (unread, {data.test_reads} accesses so far)")

    # ---- stage 0: learning rates, on validation --------------------------
    etas = probe_learning_rates(data, n_probe, grids)

    # ---- stage A: optimizer and sampling, on validation, one middle size --
    print(f"\n=== stage A: optimizer x sampling at {PROBE_SIZE} ===")
    configs = []
    for optimizer in ("adam", "sgd"):
        for sampling in ("plain", "balanced"):
            configs.append({"label": f"{optimizer}-{sampling}", "sizes": PROBE_SIZE,
                            "optimizer": optimizer, "eta": etas[optimizer],
                            "sampling": sampling})
    rowsA, storeA = run_stage(data, configs, n_stage)
    best_cfg = max(rowsA, key=rank_row)
    winner = (best_cfg["optimizer"], best_cfg["sampling"])
    print(f"\nstage A winner: optimizer={winner[0]}, sampling={winner[1]}")

    if args.select_only:
        # A validation-only investigation. This run must not open the test split:
        # the graded experiment's test numbers already exist, and re-picking a
        # configuration and then reading the test split again would be the very
        # leakage requirement 6 forbids. Write the comparison, report it, stop.
        write_results_csv(rowsA, args.out)
        plot_learning_curves(storeA, "learning_curves_selection.png")
        print(f"\nwrote {args.out} — stage 0 and stage A only")
        if data.test_reads != 0:
            raise RuntimeError(f"a --select-only run opened the test split "
                               f"{data.test_reads} time(s)")
        print(f"the test split was opened {data.test_reads} time(s): correct, "
              f"this run never touches it")
        # named after --out, so this investigation cannot overwrite the graded
        # experiment's runtime.json
        runtime_path = "runtime_" + os.path.splitext(os.path.basename(args.out))[0] + ".json"
        with open(runtime_path, "w", encoding="utf-8") as f:
            json.dump({"seed": SEED, "select_only": True, "grids": grids,
                       "total_seconds": round(time.perf_counter() - t_start, 2),
                       "per_model_seconds": {r["label"]: r["seconds"] for r in rowsA},
                       "numpy": np.__version__,
                       "threads": os.environ["OMP_NUM_THREADS"]}, f, indent=2)
        return

    # softmax baseline, also chosen on validation with the winning configuration
    base_cfg = {"label": "softmax baseline", "sizes": BASELINE,
                "optimizer": winner[0], "eta": etas[winner[0]], "sampling": winner[1]}
    rowsB0, storeB0 = run_stage(data, [base_cfg], n_stage)

    # ---- stage B: the layer sizes, on validation, with the winner --------
    print(f"\n=== stage B: the sizes, with {winner[0]} / {winner[1]} ===")
    configs = [{"label": str(s), "sizes": s, "optimizer": winner[0],
                "eta": etas[winner[0]], "sampling": winner[1]} for s in CANDIDATES]
    rowsB, storeB = run_stage(data, configs, max(n_stage, args.epochs or 0) if not args.smoke else n_stage)

    # ---- stage C: the sampling confirmed at the winning size -------------
    # stage C re-runs the runner-up sampling at the size stage B picked, so the
    # sampling decision does not rest on one architecture. sizes_by_label avoids
    # parsing the string back with eval().
    sizes_by_label = {str(list(s)): list(s) for s in CANDIDATES + [BASELINE]}
    chosen_size = max(rowsB, key=rank_row)["sizes"]
    other = "plain" if winner[1] == "balanced" else "balanced"
    print(f"\n=== stage C: sampling confirmed at the chosen size {chosen_size} ===")
    rowsC, storeC = run_stage(data, [{"label": f"{winner[0]}-{other} @ chosen size",
                                      "sizes": sizes_by_label[chosen_size],
                                      "optimizer": winner[0], "eta": etas[winner[0]],
                                      "sampling": other}], n_stage)

    # ---- the comparison, all on validation -------------------------------
    all_rows = rowsA + rowsB0 + rowsB + rowsC
    store = {**storeA, **storeB0, **storeB, **storeC}
    write_results_csv(all_rows, args.out)
    print(f"\nwrote {args.out} ({len(all_rows)} models compared on validation)")

    plot_learning_curves({k: v for k, v in store.items()
                          if k in ("adam-plain", "adam-balanced", "sgd-plain",
                                   "sgd-balanced") or k == "softmax baseline"},
                         "learning_curves.png")

    # every choice above is now fixed; the test split is opened once, here.
    best_network_label = max(rowsB, key=rank_row)["label"]
    print("\n=== final: the test split, once ===")
    X_test, y_test = data.test()

    chosen = store[best_network_label]
    pred = nn.predict(chosen["W"], chosen["b"], X_test)
    net_test = {"name": f"the chosen network {best_network_label}",
                "sizes": best_network_label,
                "best_epoch_chosen_on_validation": chosen["best_epoch"],
                "accuracy": nn.accuracy(y_test, pred),
                "balanced_accuracy": nn.balanced_accuracy(y_test, pred, N_CLASSES),
                "per_class_recall": {}}
    cm = nn.confusion_matrix(y_test, pred, N_CLASSES)
    recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    net_test["per_class_recall"] = {BLOOD_CLASSES[c]: round(float(recall[c]), 4)
                                    for c in range(N_CLASSES)}
    net_test["confusion_matrix"] = cm.tolist()
    # the test class counts, reported here rather than in the class-balance table
    # at the top of the run: this is the one place the test split is allowed to
    # inform what the experiment says, and these are the row sums of the matrix.
    net_test["per_class_test_count"] = {
        BLOOD_CLASSES[c]: int(cm[c].sum()) for c in range(N_CLASSES)}
    plot_confusion(cm, "confusion_matrix.png",
                   f"Chosen network {best_network_label} on test")

    baseline = storeB0["softmax baseline"]
    pred_b = nn.predict(baseline["W"], baseline["b"], X_test)
    base_test = {"name": "softmax baseline (no hidden layer)",
                 "accuracy": nn.accuracy(y_test, pred_b),
                 "balanced_accuracy": nn.balanced_accuracy(y_test, pred_b, N_CLASSES),
                 "confusion_matrix": nn.confusion_matrix(y_test, pred_b, N_CLASSES).tolist()}

    maj_test = majority_rule(data, y_test)
    maj_test = {"name": f"majority class rule ({maj_test['majority_class']})",
                "accuracy": maj_test["accuracy"],
                "balanced_accuracy": maj_test["balanced_accuracy"],
                "majority_class": maj_test["majority_class"]}

    # A printed warning is advice; this is a gate. `test_read_count` in
    # test_metrics.json is the record, this is the enforcement: a run that opened
    # the test split twice has already spent it, so it must not write artifacts.
    if data.test_reads != 1:
        raise RuntimeError(f"the test split was opened {data.test_reads} times; "
                           f"requirement 6 allows exactly one read")

    metrics = {"seed": SEED, "batch": BATCH, "standardization": "per pixel-channel, train split only",
               "selected_on_validation": {"optimizer": winner[0], "sampling": winner[1],
                                          "sizes": best_network_label,
                                          "epochs": chosen["best_epoch"]},
               "test_read_count": data.test_reads,
               "network": net_test, "softmax_baseline": base_test,
               "majority_rule": maj_test,
               "n_test": int(y_test.size)}
    with open("test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    runtime = {"seed": SEED, "smoke": args.smoke,
               "total_seconds": round(time.perf_counter() - t_start, 2),
               "per_model_seconds": {r["label"]: r["seconds"] for r in all_rows},
               "numpy": np.__version__, "threads": os.environ["OMP_NUM_THREADS"]}
    with open("runtime.json", "w", encoding="utf-8") as f:
        json.dump(runtime, f, indent=2)

    print(f"  chosen network : acc {net_test['accuracy']:.4f}  "
          f"bal acc {net_test['balanced_accuracy']:.4f}")
    print(f"  softmax baseline: acc {base_test['accuracy']:.4f}  "
          f"bal acc {base_test['balanced_accuracy']:.4f}")
    print(f"  majority rule  : acc {maj_test['accuracy']:.4f}  "
          f"bal acc {maj_test['balanced_accuracy']:.4f}")
    print(f"  test split was opened {data.test_reads} time(s)")
    print(f"\ntotal {runtime['total_seconds']}s   -> results.csv, test_metrics.json, "
          f"runtime.json, learning_curves.png, confusion_matrix.png")


if __name__ == "__main__":
    main()
