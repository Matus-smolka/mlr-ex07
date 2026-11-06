import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import nn_numpy as nn

# reference: the textbook loss, -log p_correct, from an explicitly safe softmax
rng = np.random.default_rng(0)
for trial in range(3):
    logits = rng.standard_normal((8, 5)) * 10.0
    targets = rng.integers(0, 8, 5)
    y = nn.one_hot(targets, 8)
    p = nn.softmax(logits)
    ref = -np.mean(np.log(p[targets, np.arange(5)]))
    mine, _ = nn.softmax_crossentropy(logits, y)
    print(f"trial {trial}: reference(batch mean) {ref:8.4f}   mine={mine:8.4f}   ratio={mine/ref:5.2f}")
print("\nC = 8 -> if the ratio is ~8, logsumexp is being summed over classes instead of once per sample")
