"""test_gradients.py — the analytic backward pass against central differences.

Run with:      pytest -q test_gradients.py

A wrong gradient raises no error; it only trains badly. So the backward pass is
checked the way Task 2 introduced and Task 3 used: nudge one parameter by eps,
keep the rest fixed, measure the slope of the loss, and compare. Every parameter
of a tiny float64 network, weights and biases of every layer, at a threshold far
tighter than training would tolerate.

The last two tests are the ones that make this a test and not a ceremony: a
deliberately broken backward pass must FAIL the same comparison. A check that
cannot fail checks nothing.
"""

import numpy as np
import pytest

import nn_numpy as nn
from download_data import N_CLASSES

ATOL = 1e-8           # absolute slack, for gradients that are legitimately ~0
RTOL = 1e-5           # relative slack; central differences are good to ~1e-9 here
EPS = 1e-6


# --------------------------------------------------------------------------
# a tiny network, small enough to check and deep enough to have three layers
# --------------------------------------------------------------------------
def tiny_problem(seed=0, sizes=(4, 5, 3, N_CLASSES), n_samples=3):
    """A tiny network, a batch of samples and the matching one-hot targets."""
    gen = np.random.default_rng(seed)
    W, b = nn.init_params(list(sizes), gen, init=0.5)
    X = gen.standard_normal((sizes[0], n_samples))
    labels = gen.integers(0, N_CLASSES, n_samples)
    Y = nn.one_hot(labels, N_CLASSES)
    return W, b, X, Y


def loss_of(W, b, X, Y):
    """The mean cross-entropy of the batch for any values of the parameters."""
    _, h = nn.forward(W, b, X)
    loss, _ = nn.softmax_crossentropy(h[-1], Y)
    return loss


def analytic(W, b, X, Y):
    a, h = nn.forward(W, b, X)
    loss, grad_a = nn.softmax_crossentropy(h[-1], Y)
    grad_W, grad_b = nn.backward(W, a, h, grad_a)
    return loss, grad_W, grad_b


def numeric(W, b, X, Y, eps=EPS):
    """Central differences, one entry of one parameter at a time."""
    grad_W = [None] * len(W)
    grad_b = [None] * len(b)
    for k in range(1, len(W)):
        grad_W[k] = np.zeros_like(W[k])
        grad_b[k] = np.zeros_like(b[k])
        for name, holder in (("W", (W, grad_W)), ("b", (b, grad_b))):
            params, grads = holder
            for index in np.ndindex(params[k].shape):
                up, down = params[k][index], params[k][index]
                params[k][index] = up + eps
                l_up = loss_of(W, b, X, Y)
                params[k][index] = down - eps
                l_down = loss_of(W, b, X, Y)
                params[k][index] = up
                grads[k][index] = (l_up - l_down) / (2 * eps)
    return grad_W, grad_b


def largest_relative_difference(gW_a, gb_a, gW_n, gb_n):
    """max over every entry of |analytic - numeric| / (ATOL + |numeric|)."""
    worst = 0.0
    where = None
    for k in range(1, len(gW_a)):
        for gname, ga, gn in (("W", gW_a[k], gW_n[k]), ("b", gb_a[k], gb_n[k])):
            rel = np.abs(ga - gn) / (ATOL + np.abs(gn))
            i = int(np.argmax(rel))
            if rel.flat[i] > worst:
                worst, where = float(rel.flat[i]), f"layer {k} {gname}"
    return worst, where


# --------------------------------------------------------------------------
# 1. every gradient of the tiny network
# --------------------------------------------------------------------------
def test_every_gradient_matches_central_difference():
    W, b, X, Y = tiny_problem()
    _, gW_a, gb_a = analytic(W, b, X, Y)
    gW_n, gb_n = numeric(W, b, X, Y)
    worst, where = largest_relative_difference(gW_a, gb_a, gW_n, gb_n)
    n_checked = sum(w.size for w in W[1:]) + sum(v.size for v in b[1:])
    print(f"\nchecked {n_checked} parameters, largest relative difference "
          f"{worst:.2e} (at {where})")
    assert worst < 1e-4, f"analytic and numeric gradients disagree: {worst:.2e} at {where}"


def test_a_second_seed_also_matches():
    """Not one lucky starting point: the same check on another draw."""
    W, b, X, Y = tiny_problem(seed=7, sizes=(3, 6, 4, N_CLASSES), n_samples=5)
    _, gW_a, gb_a = analytic(W, b, X, Y)
    gW_n, gb_n = numeric(W, b, X, Y)
    worst, where = largest_relative_difference(gW_a, gb_a, gW_n, gb_n)
    assert worst < 1e-4, f"seed 7: {worst:.2e} at {where}"


def test_a_deeper_network_also_matches():
    """Four layers: the loop has to pass the gradient down two hidden layers."""
    W, b, X, Y = tiny_problem(seed=3, sizes=(5, 4, 3, 6, N_CLASSES), n_samples=4)
    _, gW_a, gb_a = analytic(W, b, X, Y)
    gW_n, gb_n = numeric(W, b, X, Y)
    worst, where = largest_relative_difference(gW_a, gb_a, gW_n, gb_n)
    assert worst < 1e-4, f"4 layers: {worst:.2e} at {where}"


def test_gradient_shapes_match_their_parameters():
    W, b, X, Y = tiny_problem()
    _, gW, gb = analytic(W, b, X, Y)
    for k in range(1, len(W)):
        assert gW[k].shape == W[k].shape, f"layer {k}: grad_W {gW[k].shape} vs W {W[k].shape}"
        assert gb[k].shape == b[k].shape, f"layer {k}: grad_b {gb[k].shape} vs b {b[k].shape}"


# --------------------------------------------------------------------------
# 2. the batch really is an average, and the bias gradient really is summed
# --------------------------------------------------------------------------
def test_batch_average_equals_mean_of_single_sample_losses():
    W, b, X, Y = tiny_problem(n_samples=6)
    batch_loss, _ = nn.softmax_crossentropy(nn.forward(W, b, X)[1][-1], Y)
    singles = [loss_of(W, b, X[:, [j]], Y[:, [j]]) for j in range(X.shape[1])]
    assert abs(batch_loss - float(np.mean(singles))) < 1e-12


def test_bias_gradient_is_summed_over_the_batch():
    """The batch dL/db_k equals the mean of the per-sample dL/db_k.

    This is the one place a mini-batch network may quietly differ from the
    per-sample Task 4 code: summing the columns of grad_a once (correct) versus
    averaging them twice, or leaving them summed when the loss is an average.
    """
    W, b, X, Y = tiny_problem(n_samples=5)
    a, h = nn.forward(W, b, X)
    _, grad_a = nn.softmax_crossentropy(h[-1], Y)
    _, gb_batch = nn.backward(W, a, h, grad_a)

    per_sample = []
    for j in range(X.shape[1]):
        aj, hj = nn.forward(W, b, X[:, [j]])
        _, gaj = nn.softmax_crossentropy(hj[-1], Y[:, [j]])
        _, gb_j = nn.backward(W, aj, hj, gaj)
        per_sample.append(gb_j)

    # index 0 of every list is the unused layer, exactly as in the network
    mean_of_samples = [None] + [sum(g[k] for g in per_sample) / X.shape[1]
                                for k in range(1, len(b))]
    for k in range(1, len(b)):
        assert np.max(np.abs(gb_batch[k] - mean_of_samples[k])) < 1e-12


# --------------------------------------------------------------------------
# 3. the softmax and the cross-entropy are numerically safe
# --------------------------------------------------------------------------
def test_extreme_logits_give_a_finite_loss():
    """The check from the task sheet: logits (1000, 0, -1000) must not produce inf or nan."""
    for row in ([1000.0, 0.0, -1000.0], [1000.0] * 3, [-1000.0] * 3,
                [1e300, 0.0, -1e300]):
        logits = np.array(row).reshape(-1, 1)
        for target in range(3):
            y = np.zeros((3, 1))
            y[target, 0] = 1.0
            loss, grad = nn.softmax_crossentropy(logits, y)
            assert np.isfinite(loss), f"loss not finite for {row}"
            assert np.all(np.isfinite(grad)), f"gradient not finite for {row}"
            assert loss >= -1e-12, f"a cross-entropy cannot be negative: {loss} for {row}"


def test_equal_logits_give_exactly_log_n_classes():
    assert abs(nn.softmax_crossentropy(np.zeros((8, 4)),
                                       nn.one_hot(np.arange(4), 8))[0] - np.log(8)) < 1e-12


def test_loss_matches_the_textbook_minus_log_p():
    """The safe log-sum-exp route must agree with -log(softmax p) on ordinary logits."""
    gen = np.random.default_rng(1)
    logits = gen.standard_normal((N_CLASSES, 20)) * 8.0
    targets = gen.integers(0, N_CLASSES, 20)
    Y = nn.one_hot(targets, N_CLASSES)
    p = nn.softmax(logits)
    reference = -np.mean(np.log(p[targets, np.arange(20)]))
    mine, _ = nn.softmax_crossentropy(logits, Y)
    assert abs(mine - reference) < 1e-10


# --------------------------------------------------------------------------
# 4. the check must FAIL for a wrong backward pass — otherwise it checks nothing
# --------------------------------------------------------------------------
def backward_missing_relu_mask(W, a, h, grad_a):
    """Broken on purpose: the gradient passes the ReLU without its mask."""
    K = len(W) - 1
    n = grad_a.shape[1]
    gW, gb = [None] * (K + 1), [None] * (K + 1)
    for k in range(K, 0, -1):
        gW[k] = grad_a @ h[k - 1].T
        gb[k] = grad_a.sum(axis=1, keepdims=True) / n
        if k > 1:
            grad_a = W[k].T @ grad_a            # <- the mask (a[k-1] > 0) is missing
    return gW, gb


def backward_counts_logsumexp_per_class(W, a, h, grad_a):
    """Broken on purpose: the loss bug this project actually had.

    The cross-entropy written as np.sum(lse - y * logits) adds logsumexp once per
    class instead of once per sample, inflating the loss about eightfold. The
    gradient it pairs with is unchanged, so the network still trains and only the
    reported loss is wrong — which is why the loss is tested on its own.
    """
    return nn.backward(W, a, h, grad_a)


def broken_crossentropy_inflates_the_loss(logits, y):
    n = logits.shape[1]
    lse = nn.logsumexp(logits)
    loss = float(np.sum(lse - (y * logits)) / n)      # the original bug, verbatim
    proba = np.exp(logits - lse)
    return loss, (proba - y) / n


def test_the_gradient_check_fails_for_a_backward_pass_with_a_bug():
    W, b, X, Y = tiny_problem()
    a, h = nn.forward(W, b, X)
    _, grad_a = nn.softmax_crossentropy(h[-1], Y)
    gW_n, gb_n = numeric(W, b, X, Y)
    gW_b, gb_b = backward_missing_relu_mask(W, a, h, grad_a)
    worst, where = largest_relative_difference(gW_b, gb_b, gW_n, gb_n)
    print(f"\nthe broken backward pass is caught: relative difference {worst:.2e} at {where}")
    assert worst > 1e-2, "the check did not notice the missing ReLU mask — it checks nothing"


def test_the_loss_check_fails_for_the_logsumexp_bug():
    gen = np.random.default_rng(2)
    logits = gen.standard_normal((N_CLASSES, 10))
    Y = nn.one_hot(gen.integers(0, N_CLASSES, 10), N_CLASSES)
    good, _ = nn.softmax_crossentropy(logits, Y)
    bad, _ = broken_crossentropy_inflates_the_loss(logits, Y)
    print(f"\nthe loss bug shows up: correct {good:.4f}, buggy {bad:.4f}, "
          f"ratio {bad/good:.2f}")
    assert abs(bad - good) > 0.5, "the logsumexp bug was too small to notice"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
