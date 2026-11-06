## Task 6 — Your network, deeper, on blood-cell images (compulsory, with an agent)

<p style="color: #c00000;"><strong>! Compulsory hand-in: complete and submit Task 6 within 48 hours after your lab. Attendance will not be credited unless you complete this requirement.</strong></p>

**The practical problem.** A complete blood count with a *differential* tells
a physician how many cells of each type are in a patient's blood. Analysers
photograph single cells from a peripheral blood smear, and a laboratory
technician still checks many of them under the microscope. A classifier that
pre-sorts the cells into types, so that the technician only reviews the
uncertain ones, saves real work. You will build the core of such a classifier
from the **NumPy network of Task 4**.

**The data: BloodMNIST** (MedMNIST v2, licence CC BY 4.0) — 17,092 microscope
images of individual normal blood cells from the Hospital Clinic of Barcelona
(Acevedo et al., 2020), centre-cropped and downsampled to 28 × 28 RGB pixels,
in **eight classes**: basophil, eosinophil, erythroblast, immature granulocyte,
lymphocyte, monocyte, neutrophil and platelet. The classes are **not
balanced** (from 849 to 2,330 training images). The official split is 11,959
training, 1,712 validation and 3,421 test images; one file of 35 MB:
`https://zenodo.org/records/10519652/files/bloodmnist.npz?download=1`, MD5
`7053d0359d879ad8a5505303e11de1dc`.

The notebook previews four training images of every class (a 66 kB sample,
not the whole file). Some classes are easy to tell apart by eye, others much less
so — and the network sees each image only as 2,352 numbers.

### The goal and what is graded

**The goal:** extend the Task 4 network into a classifier of the eight cell
types, compare a few network sizes on the validation split against a softmax
baseline, evaluate the chosen network once on the test split, and write a
short report. Against Task 4, the input has 28 · 28 · 3 = 2,352 numbers
instead of one, the hidden layers are wider, and the output has eight
neurons with a **softmax** and the **cross-entropy** loss from the lecture.
The backward pass stays your loop over the layers. A network of a few
hundred thousand parameters still trains on a laptop CPU in minutes.

**Writing the instructions for the agent is your job.** There is no
ready-made prompt. Turn the goal and the requirements below into your own
instructions — what to build, in what order, how to check it — and refine
them as you see what the agent does. A good habit: ask for the gradient test
and a quick run on a small part of the data before the full comparison.

**Requirements, and how you check each one in `REVIEW.md`:**

1. **The Task 4 network, extended:** the network code starts from the Task 4
   script and keeps its conventions (one sample per column, the lists
   `W[k]`, `b[k]`, the loops over the layers, the backward formulas of the
   lecture). → Compare the forward and backward pass in `nn_numpy.py` with
   the Task 4 code: are the conventions still there? If it trains on
   mini-batches (several samples as columns), is the bias gradient summed and
   the loss averaged over the batch?
2. **NumPy only** for the network, the training and the optimizer (for
   example Adam from the lecture): no PyTorch, TensorFlow, JAX, autograd
   library or scikit-learn model; scikit-learn may only cross-check a metric.
   → Search every file the agent wrote for `torch`, `tensorflow`, `jax`,
   `autograd`, `sklearn` (your own notebook may use PyTorch in Task 5). Is the
   backward pass really computed in `nn_numpy.py`?
3. **A numerically safe softmax and cross-entropy.** → Is the maximum
   subtracted, is there a `log(0)` risk, is the loss averaged over the batch?
   Feed it the logits (1000, 0, −1000) yourself: the loss must be finite.
4. **The data** are downloaded by a script from the URL above, the MD5 is
   checked, the file is never committed; the official split; standardization
   from the training images only. → Where are the standardization statistics
   computed? Are the images flattened consistently, the labels 0–7 named
   correctly, the class balance reported?
5. **A real gradient test** (`test_gradients.py`, runnable with `pytest`): a
   tiny `float64` network, every analytic gradient, biases included, against a
   central difference with a strict threshold, and a demonstration that it
   fails for a deliberately broken backward pass. → Run it yourself. A check
   that cannot fail checks nothing.
6. **Validation for every choice, test once:** the network size and the epoch
   are chosen on validation; then the test split is used once, for the chosen
   network, the softmax baseline (no hidden layer) and the majority-class
   rule, with accuracy, balanced accuracy, per-class recall and a confusion
   matrix. → Find every use of the test images. Was everything chosen before
   the test set was opened? Is no training accuracy reported as a result?
7. **Reproducible:** fixed seeds, a README with the commands from a fresh
   clone, the measured runtime, and a second run that gives the same
   `results.csv` (or a reported and explained difference). → Inspect the
   seeds, rerun yourself, compare the two `results.csv`, record the command
   and your runtime.
8. **Cautious conclusions** in the report. → Do the learning curves support
   them? How much better than the softmax baseline and the majority rule is
   the network? Which classes are confused? What can this result not
   establish about clinical use?

For each point record in `REVIEW.md` the location (file/line or notebook
cell), what you checked, your finding and how any problem was fixed. "No
problem found" needs evidence too.

**Required files:** `ex07_notebook.ipynb`, `nn_numpy.py`, `download_data.py`,
`train.py`, `test_gradients.py`, `pyproject.toml`, `uv.lock`, `README.md`,
`results.csv` (one row per compared model with its validation accuracy and
balanced accuracy), `test_metrics.json`, `runtime.json`,
`learning_curves.png`, `confusion_matrix.png`, a self-contained `report.html`
and your own `REVIEW.md`. Do not let the agent write `REVIEW.md`.

**In class:** create the repository, invite the teacher, write your first
instructions, and get the gradient test and a short trial run going. **At
home:** the full run, your own review and the hand-in. The setup (your own
notebook in the ZIP folder, the private repository `mlr-ex07`, the invitation
of **MLRteacher**) is described on the exercise page and in the ZIP's
`README.md`.

Submit by pushing within **48 hours after the end of your lab**. What is pushed
before the deadline counts. Hand-ins are reviewed automatically by AI.

<p style="color: #c00000;"><strong>! Plagiarism check: all hand-ins will be checked for plagiarism.</strong></p>

Without a working agent, meet the same requirements yourself at home.
*Optional after the required run:* L2 weight decay or dropout from the
lecture, chosen on validation only.

**What to expect:** the softmax baseline reaches roughly 0.8 validation
accuracy and a NumPy network with hidden layers clearly more; the whole
comparison takes minutes, not hours, on a laptop CPU. Much better or much
worse results deserve an investigation, not automatic acceptance.
