"""The BloodMNIST data: download, MD5 check, official split, standardization.

Everything about the data lives here, so that the two questions the review asks
have one answer each:

* where do the standardization statistics come from?  -> `standardize()`, and it
  is only ever given the TRAINING images;
* is the file checked and never committed?             -> `download()`, MD5 below.

The official split of MedMNIST v2 (11,959 train / 1,712 val / 3,421 test) is
used as it is: it is a real split, it is the one everybody reports on, and it
keeps the three parts free of each other. NumPy only.
"""

import hashlib
import os
import urllib.request

import numpy as np

URL = "https://zenodo.org/records/10519652/files/bloodmnist.npz?download=1"
MD5 = "7053d0359d879ad8a5505303e11de1dc"
FILENAME = "bloodmnist.npz"

# the eight classes, in the order of the integer labels 0..7
BLOOD_CLASSES = ["basophil", "eosinophil", "erythroblast", "immature granulocyte",
                 "lymphocyte", "monocyte", "neutrophil", "platelet"]
N_CLASSES = len(BLOOD_CLASSES)
PIXELS = 28 * 28 * 3          # the 2352 numbers one image becomes


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(path=FILENAME, url=URL, md5=MD5, quiet=False):
    """The 35 MB file from Zenodo, downloaded only when it is missing or wrong."""
    if os.path.exists(path) and md5_of(path) == md5:
        if not quiet:
            print(f"{path}: already here, MD5 matches")
        return path
    if not quiet:
        print(f"downloading {url}\n         -> {path}")
    tmp = path + ".part"
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
    if md5_of(tmp) != md5:
        os.remove(tmp)
        raise RuntimeError(f"MD5 of the downloaded file does not match {md5}")
    os.replace(tmp, path)
    if not quiet:
        print(f"{path}: MD5 matches")
    return path


class Splits:
    """The three parts of the official split, as columns of standardized pixels.

    `.train` and `.val` are ordinary arrays. **The test images are not held in
    memory at all** until `test()` is called: the constructor keeps only the path
    and the standardization vectors, and `test()` reads the test arrays from the
    file on its first call. There is therefore no `data.test_images` or
    `data._y_test` to reach past the guard — an earlier version of this class
    stored the test split in a private attribute, and two places in this project
    then read `data._y_test` directly and bypassed the counter they thought was
    protecting them. Removing the attribute is what makes the rule enforceable;
    a name starting with an underscore protects nothing in Python.

    Requirement 6 of the task asks "find every use of the test images": `test()`
    is the only door, and `test_reads` counts who walked through it.
    """

    def __init__(self, train, y_train, val, y_val, path, mean, std):
        self.train, self.y_train = train, y_train
        self.val, self.y_val = val, y_val
        self._path = path                      # the path, not the test arrays
        self.mean, self.std = mean, std
        self.test_reads = 0
        self._test = None

    def test(self):
        """The test split, read from the file on the first call. Once, at the end."""
        self.test_reads += 1
        if self._test is None:
            z = np.load(self._path)
            test = to_columns(z["test_images"])
            self._test = (test - self.mean) / self.std
            self._y_test = z["test_labels"].ravel().astype(int)
        if self.test_reads == 1:
            print(f"  [test split opened — call {self.test_reads}, "
                  f"{self._y_test.size} images]")
        else:
            print(f"  [WARNING: test split opened for the {self.test_reads}th time]")
        return self._test, self._y_test

    @property
    def n_train(self):
        return self.y_train.size

    @property
    def n_val(self):
        return self.y_val.size


def to_columns(images):
    """(N, 28, 28, 3) uint8  ->  (2352, N) float64, one sample per column.

    The flattening is C order over (height, width, channel), so a feature index
    is ((row * 28) + col) * 3 + channel. The same function is used for every
    split and for training and inference, which is what keeps the pixels in the
    same place in the vector everywhere.
    """
    flat = images.reshape(images.shape[0], -1).astype(np.float64).T
    return np.ascontiguousarray(flat)


def standardize(train_columns):
    """The mean and standard deviation of the TRAINING images, per pixel-channel.

    Returns (mean, std), each (2352, 1). A pixel that is constant over the
    training images (the black corner of every crop) has std 0; dividing by it
    would spread noise to infinity, so such a feature is only centered.
    """
    mean = train_columns.mean(axis=1, keepdims=True)
    std = train_columns.std(axis=1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def load_splits(path=FILENAME, download_if_missing=True):
    """Train and val, standardized with statistics from the training split only.

    The test arrays are deliberately not read here, even though the file is open
    and they are one key away: `Splits` gets the path and loads them itself, on
    the first `test()` call. See the class docstring for why.
    """
    if download_if_missing:
        download(path)
    z = np.load(path)
    train = to_columns(z["train_images"])
    val = to_columns(z["val_images"])
    mean, std = standardize(train)                  # train only — never val, never test
    return Splits((train - mean) / std, z["train_labels"].ravel().astype(int),
                  (val - mean) / std, z["val_labels"].ravel().astype(int),
                  path, mean, std)


def class_counts(labels):
    return np.bincount(labels, minlength=N_CLASSES)


def report_class_balance(splits):
    """The data analysis: how the eight classes are spread over train and val.

    **Train and validation only, never test.** This function is printed before
    any decision is made — whether to mitigate the imbalance, which sampling to
    use, which metric to rank by — and a number printed before a choice is made
    can inform that choice. That is the reason the test column is absent here,
    and it is also the reason an earlier version of this function, which printed
    the test percentages too, had to be changed: it read `splits._y_test`
    directly, which bypassed the `test()` guard and put test-set statistics in
    front of the deciding step. The test class counts are not secret (they are
    the official MedMNIST split) but they are not *this experiment's* evidence
    until the single evaluation at the end, and the confusion matrix reported
    there gives them as row sums.

    The imbalance judgement this function supports is made on the training
    counts, which is the split the sampling could be changed for anyway.
    """
    lines = []
    lines.append("Class balance of BloodMNIST (training and validation splits)")
    lines.append(f"{'class':<22} {'train':>7} {'%':>6} {'val':>6} {'%':>6}")
    counts = {name: class_counts(getattr(splits, f"y_{name}"))
              for name in ("train", "val")}
    n = {name: counts[name].sum() for name in counts}
    for c, name in enumerate(BLOOD_CLASSES):
        lines.append(f"{c} {name:<20} {counts['train'][c]:>7} "
                     f"{100*counts['train'][c]/n['train']:>5.1f}%"
                     f" {counts['val'][c]:>6} {100*counts['val'][c]/n['val']:>5.1f}%")
    tr = counts["train"]
    hi, lo = int(tr.argmax()), int(tr.argmin())
    lines.append(f"totals: train {n['train']}, val {n['val']}"
                 f"   (test split: not read, see Splits.test)")
    lines.append(f"most frequent: {BLOOD_CLASSES[hi]} {tr[hi]} ({100*tr[hi]/n['train']:.1f}%)"
                 f"   least frequent: {BLOOD_CLASSES[lo]} {tr[lo]} ({100*tr[lo]/n['train']:.1f}%)")
    lines.append(f"ratio most/least frequent: {tr[hi]/tr[lo]:.2f}")
    return "\n".join(lines)


if __name__ == "__main__":
    s = load_splits()
    print(report_class_balance(s))
    print(f"\ntrain as columns: {s.train.shape}, mean {s.train.mean():.3f}, "
          f"std {s.train.std():.3f}   (val: mean {s.val.mean():.3f}, std {s.val.std():.3f})")
