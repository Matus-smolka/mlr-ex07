"""Task 6 — the Task 4 network, extended to eight-class classification.

Everything here is NumPy and plain Python: no PyTorch, TensorFlow, JAX, autograd
library or scikit-learn, in the forward pass *and* in the backward pass.

Conventions kept from the Task 4 script of the notebook
-------------------------------------------------------
* a sample is a **column**, a batch is several samples in the columns;
* the parameters live in the Python lists ``W`` and ``b``, indexed by the
  **layer number of the lecture** (``W[k]``, ``b[k]`` belong to layer k,
  ``h[0]`` is the input, ``h[K]`` the prediction, index 0 unused ``None``);
* ``W[k]`` has shape ``(sizes[k], sizes[k - 1])`` and ``b[k]`` ``(sizes[k], 1)``;
* the forward pass runs ``for k in range(1, K + 1)``, the backward pass
  ``for k in range(K, 0, -1)``, with ReLU in the hidden layers and the identity
  in the last layer, so the prediction is ``h[K] = a[K]`` (the logits);
* the backward pass is the lecture formulas
      grad_W[k] = grad_a_k @ h[k-1].T
      grad_b[k] = grad_a_k
      grad_h    = W[k].T @ grad_a_k,   grad_a[k-1] = grad_h * (a[k-1] > 0)
  — with the single change that a batch brings several columns, so the bias
  gradient is **summed** over the batch and the loss **averaged** over it.

What Task 6 adds
----------------
* the input has 28 * 28 * 3 = 2352 numbers instead of one,
* the output has 8 neurons and the loss is the **cross-entropy** of the
  **softmax** instead of 1/2 (y_hat - y)^2.

The one place this touches the Task 4 loop is the starting point of the
backward pass. For the squared error it was ``grad_a = y_hat - y``; for the
softmax with cross-entropy it is ``grad_a = (p - onehot) / N``. Every line of
the loop below it is the Task 4 loop unchanged.
"""

import numpy as np

__all__ = ["relu", "init_params", "forward", "logits_to_proba", "softmax",
           "logsumexp", "one_hot", "softmax_crossentropy", "backward",
           "accuracy", "balanced_accuracy", "confusion_matrix", "adam_step"]


# --------------------------------------------------------------------------
# activation
# --------------------------------------------------------------------------
def relu(a):
    """f(alpha) = max(0, alpha), exactly as in Task 3 and Task 4."""
    return np.maximum(a, 0.0)


def relu_grad(a):
    """f'(alpha) = 1 for alpha > 0 and 0 otherwise (0 at alpha = 0, as the note)."""
    return a > 0


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------
def init_params(sizes, gen, init="he"):
    """The parameters W[k], b[k] of the layers k = 1..K, drawn from gen.

    Same shapes and same order of drawing as Task 4 (W of a layer, then its b).
    Task 4 draws with standard deviation 0.1, which suits one input number;
    with 2352 standardized inputs the inner potential of the first layer would
    start with a standard deviation of about 0.1 * sqrt(2352) ~ 4.9. So the
    default is the He initialization, std = sqrt(2 / n_in), which keeps the
    signal of a ReLU layer at a fixed size. Pass init=0.1 for Task 4's rule.
    """
    K = len(sizes) - 1
    W = [None] * (K + 1)
    b = [None] * (K + 1)
    for k in range(1, K + 1):
        n_in = sizes[k - 1]
        std = np.sqrt(2.0 / n_in) if init == "he" else float(init)
        W[k] = gen.normal(0.0, std, (sizes[k], sizes[k - 1]))
        b[k] = gen.normal(0.0, std, (sizes[k], 1))
    return W, b


def count_params(W, b):
    return int(sum(w.size for w in W[1:]) + sum(v.size for v in b[1:]))


# --------------------------------------------------------------------------
# forward pass — the Task 4 loop, unchanged
# --------------------------------------------------------------------------
def forward(W, b, h0):
    """The forward pass of the Task 4 loop, over a batch of column samples.

    h0: sizes[0] x N   (one sample per column; N = 1 is the Task 4 case)
    returns the lists a, h indexed by the layer number; h[K] is the logits.
    """
    K = len(W) - 1
    a = [None] * (K + 1)
    h = [None] * (K + 1)
    h[0] = h0
    for k in range(1, K + 1):
        a[k] = W[k] @ h[k - 1] + b[k]
        if k < K:
            h[k] = relu(a[k])                    # ReLU in the hidden layers
        else:
            h[k] = a[k]                          # linear output layer: the logits
    return a, h


def logits_to_proba(logits):
    """Softmax over the classes, one class per row, one sample per column."""
    return np.exp(logits - logsumexp(logits))


# Kept under its classic name for readability of the checks.
softmax = logits_to_proba


def logsumexp(a):
    """log(sum_c exp(a_c)) per column, with the maximum subtracted first.

    Subtracting the column maximum is what makes this safe: exp() of a large
    logit would overflow, and the largest exponent that survives is exp(0) = 1.
    """
    m = np.max(a, axis=0, keepdims=True)
    return (np.log(np.exp(a - m).sum(axis=0, keepdims=True)) + m)


def one_hot(labels, n_classes):
    """labels (n,) of integers 0..n_classes-1  ->  n_classes x n indicator matrix."""
    labels = np.asarray(labels).ravel().astype(int)
    y = np.zeros((n_classes, labels.size))
    y[labels, np.arange(labels.size)] = 1.0
    return y


def softmax_crossentropy(logits, y):
    """The cross-entropy of the softmax, averaged over the batch.

    logits: n_classes x N, y: the same shape, one 1 per column.
    Returns (loss, grad_a), where grad_a is the gradient of the *mean* loss
    with respect to the logits — the starting point of the backward pass, in
    place of Task 4's (y_hat - y).

    Numerical safety. The loss is computed through log-sum-exp rather than as
    log(p): with m the column maximum,
        -log p_c = logsumexp(a) - a_c = m + log(sum exp(a - m)) - a_c ,
    where every exponent is at most exp(0) = 1, so nothing overflows, and the
    log is taken of a sum of positive terms that is at least 1, never of 0.
    A logit of 1000 or -1000 is therefore harmless.

    Shape care: logsumexp returns one number PER SAMPLE, shape (1, N), so it
    must be paired with (y * logits).sum(axis=0), also one per sample. Adding
    it to the whole (n_classes, N) array and summing everything — the obvious
    mistake — counts logsumexp n_classes times and inflates the loss.
    """
    n = logits.shape[1]
    lse = logsumexp(logits)                    # (1, N), never inf and never nan
    correct_logit = (y * logits).sum(axis=0)   # (N,) the logit of the true class
    loss = float(np.mean(lse[0] - correct_logit))
    proba = np.exp(logits - lse)               # columns sum to 1
    grad_a = (proba - y) / n                   # mean over the batch
    return loss, grad_a


# --------------------------------------------------------------------------
# backward pass — the Task 4 loop, with the bias gradient summed over the batch
# --------------------------------------------------------------------------
def backward(W, a, h, grad_a):
    """Backpropagation from grad_a down to the first layer: Task 4's loop.

    grad_a is dL/d a[K] for the whole batch (N columns), and it is the gradient
    of the loss **averaged** over the batch — softmax_crossentropy already puts
    the 1/N into every column. Everything below inherits that, so:

        grad_W[k] = grad_a @ h[k-1].T          sum over the columns, no further /N
        grad_b[k] = grad_a.sum(axis=1)         sum over the columns, no further /N

    Both sums ARE the average, because each column already carries its 1/N.
    Dividing by N a second time — the trap here, and the one requirement 1 of
    the task asks you to check — shrinks the gradient by a factor N and the
    biases learn N times too slowly. The gradient test catches it; it did catch
    it here, in this line, on the first run of test_gradients.py.
    """
    K = len(W) - 1
    grad_W = [None] * (K + 1)
    grad_b = [None] * (K + 1)
    for k in range(K, 0, -1):
        grad_W[k] = grad_a @ h[k - 1].T
        grad_b[k] = grad_a.sum(axis=1, keepdims=True)       # summed over the batch
        if k > 1:
            grad_h = W[k].T @ grad_a                        # gradient of h[k-1]
            grad_a = grad_h * relu_grad(a[k - 1])           # through the ReLU of layer k-1
    return grad_W, grad_b


# --------------------------------------------------------------------------
# one Adam update (the optimizer of the lecture), written out in NumPy
# --------------------------------------------------------------------------
def adam_step(W, b, grad_W, grad_b, state, step, eta=0.001, beta1=0.9,
              beta2=0.999, eps=1e-8):
    """W <- W - eta * m_hat / (sqrt(v_hat) + eps), in place, for every layer.

    state is a dict holding the first and second moments; it is created here
    on the first call. Nothing but NumPy arrays and arithmetic.
    """
    if "m" not in state:
        state["m"] = [np.zeros_like(p) for p in W[1:]] + [np.zeros_like(p) for p in b[1:]]
        state["v"] = [np.zeros_like(p) for p in W[1:]] + [np.zeros_like(p) for p in b[1:]]
        state["g"] = [None] * (len(W) - 1)
    grads = grad_W[1:] + grad_b[1:]
    params = W[1:] + b[1:]
    for i, g in enumerate(grads):
        state["m"][i] = beta1 * state["m"][i] + (1 - beta1) * g
        state["v"][i] = beta2 * state["v"][i] + (1 - beta2) * g * g
        m_hat = state["m"][i] / (1 - beta1 ** step)
        v_hat = state["v"][i] / (1 - beta2 ** step)
        params[i] -= eta * m_hat / (np.sqrt(v_hat) + eps)


def sgd_step(W, b, grad_W, grad_b, eta=0.05, momentum=0.9, state=None):
    """The Task 4 update W <- W - eta * grad, over a whole batch of columns.

    Same rule as Task 4 and Task 3b, only the gradient now comes from a batch
    (and is already averaged over it). Optional heavy-ball momentum, kept in
    `state`; pass state=None for plain gradient descent.
    """
    if state is not None and "m" not in state:
        state["m"] = [np.zeros_like(p) for p in W[1:]] + [np.zeros_like(p) for p in b[1:]]
    grads = grad_W[1:] + grad_b[1:]
    params = W[1:] + b[1:]
    for i, g in enumerate(grads):
        if state is None:
            params[i] -= eta * g
        else:
            state["m"][i] = momentum * state["m"][i] + g
            params[i] -= eta * state["m"][i]


# --------------------------------------------------------------------------
# metrics, written in NumPy (no scikit-learn)
# --------------------------------------------------------------------------
def confusion_matrix(labels, pred, n_classes):
    """cm[c, p] = how many samples of class c were predicted as class p."""
    labels = np.asarray(labels).ravel().astype(int)
    pred = np.asarray(pred).ravel().astype(int)
    return np.bincount(labels * n_classes + pred, minlength=n_classes ** 2
                       ).reshape(n_classes, n_classes)


def accuracy(labels, pred):
    return float(np.mean(np.asarray(labels).ravel() == np.asarray(pred).ravel()))


def balanced_accuracy(labels, pred, n_classes):
    """The mean of the per-class recalls — the accuracy that ignores the balance."""
    cm = confusion_matrix(labels, pred, n_classes)
    per_class = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    return float(np.mean(per_class))


def predict(W, b, X, batch=512):
    """Class index of every column of X, in batches to keep memory bounded."""
    K = len(W) - 1
    out = np.empty(X.shape[1], dtype=int)
    for start in range(0, X.shape[1], batch):
        _, h = forward(W, b, X[:, start:start + batch])
        out[start:start + batch] = np.argmax(h[K], axis=0)
    return out


def predict_proba(W, b, X, batch=512):
    """Softmax probabilities n_classes x N for every column of X."""
    K = len(W) - 1
    n_classes = W[K].shape[0]
    out = np.empty((n_classes, X.shape[1]))
    for start in range(0, X.shape[1], batch):
        _, h = forward(W, b, X[:, start:start + batch])
        out[:, start:start + batch] = logits_to_proba(h[K])
    return out
