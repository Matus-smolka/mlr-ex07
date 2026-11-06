"""Numerical-safety check + per-epoch timing of the real network, on this CPU."""
import sys, time
sys.path.insert(0, sys.argv[1])          # directory holding nn_numpy.py
import numpy as np
import nn_numpy as nn

# --- 1. the extreme logits from the task sheet: (1000, 0, -1000) ------------
for row in ([1000.0, 0.0, -1000.0], [1000.0, 1000.0, 1000.0], [-1000.0, -1000.0, -1000.0]):
    logits = np.array(row).reshape(-1, 1)
    for target in range(3):
        y = np.zeros((3, 1)); y[target, 0] = 1.0
        loss, grad = nn.softmax_crossentropy(logits, y)
        ok = np.isfinite(loss) and np.all(np.isfinite(grad))
        print(f"logits={row} class={target}  loss={loss:12.4f}  finite={ok}")
p = nn.softmax(np.array([[3.0], [1.0], [0.2]]))
print("softmax column sums to 1:", float(p.sum()))

# --- 2. timing: how long is one epoch on 11959 training images? -------------
N, D = 11959, 2352
gen = np.random.default_rng(0)
X = gen.standard_normal((D, N))
labels = gen.integers(0, 8, N)
Y = nn.one_hot(labels, 8)

for sizes in ([2352, 8], [2352, 128, 8], [2352, 256, 8], [2352, 256, 128, 8], [2352, 512, 256, 8]):
    for batch in (128, 256):
        W, b = nn.init_params(sizes, np.random.default_rng(0))
        state, step = {}, 0
        t0 = time.perf_counter()
        order = gen.permutation(N)
        for s in range(0, N - batch + 1, batch):
            idx = order[s:s + batch]
            Xb, Yb = X[:, idx], Y[:, idx]
            a, h = nn.forward(W, b, Xb)
            loss, grad_a = nn.softmax_crossentropy(h[-1], Yb)
            gW, gb = nn.backward(W, a, h, grad_a)
            step += 1
            nn.adam_step(W, b, gW, gb, state, step, eta=0.001)
        dt = time.perf_counter() - t0
        n_batches = (N - N % batch) // batch
        print(f"{str(sizes):<22} params={nn.count_params(W,b):>9,}  batch={batch:>3}  "
              f"epoch={dt:6.2f}s  ({n_batches} updates, {dt/n_batches*1000:5.1f} ms/update)")

import multiprocessing
print("threads/cpus:", multiprocessing.cpu_count())
