# Exercise 7 — agent task materials

> Preserved copy. This is the `README.md` that came in the exercise ZIP. The
> project README (commands, files, reproducibility) is now the top-level
> `README.md`, so the hand-in instructions from the handout are kept here
> unchanged.

<p style="color: #c00000;"><strong>! Compulsory hand-in: complete and submit Task 6 within 48 hours after your lab. Attendance will not be credited unless you complete this requirement.</strong></p>

Extract the ZIP into a new folder. It contains:

- `ex07_notebook.ipynb`: the current student notebook, including starter code.
- `TASK.md`: Task 6 from the notebook: the goal, the graded requirements,
  each with how you check it in `REVIEW.md`, the required files and the
  hand-in rules. It is **not** a
  prompt: writing the instructions for your agent is your job.

The blood-cell data (`bloodmnist.npz`, 35 MB) are not included: your code
downloads them from Zenodo and checks the MD5 (the URL and the checksum are in
`TASK.md`). Do not commit the data file.

## When to use AI agents

- **Tasks 1–5: work without AI agents.** Write and understand the code yourself. Use the formulas, the step comments in every cell and teacher help.
- **Task 6: use your coding agent.** Have it extend **the network from Task 4** into a deeper classifier, run the experiments and generate the report.
- **Your review is your own work.** Check the agent's output and write `REVIEW.md` yourself; do not ask the agent to write it.

## Start from your own work

Download your completed or partially completed notebook from Colab/Jupyter.
Replace the bundled blank `ex07_notebook.ipynb` with **your own notebook**;
keep your original copy safe. The Task 4 script (the loop over the layers)
must run in your notebook and reproduce your Task 3 loss curve before you
start the agent: the agent extends **that code**. The bundled
notebook is a starting copy, not a substitute for your own work or a completed
solution.

**Optional editor:** if you want to use VS Code on Windows, install it from PowerShell or Windows Terminal:

```powershell
winget install --id Microsoft.VisualStudioCode --exact --scope user --accept-source-agreements --accept-source-agreements
```

Open this folder in the coding agent you set up in the fifth lab and give it
**your own instructions**, written from the goal and the requirements in
`TASK.md`. A good start: have it read your notebook and the Task 4 script
before it extends it, and get the gradient test and a
short trial run working before the full comparison.

Create a **private** GitHub repository `mlr-ex07` and invite **MLRteacher**:
repository **Settings → Collaborators → Add people → MLRteacher** (a pending
invitation is sufficient; your public repository from the fifth lab is not a
suitable hand-in location). Then follow the hand-in rules in `TASK.md`. You write `REVIEW.md` yourself. The ZIP
contains the input materials; the network code, results and review are the
work you and your agent produce.

If GitHub CLI (`gh`) is installed and authenticated (`gh auth status`), you can simply ask your coding agent:

> Use `gh` to invite `MLRteacher` as a collaborator to my private `mlr-ex07` repository. Check the repository owner and confirm that the invitation is pending or access already exists.

Your signed-in GitHub account must have permission to manage collaborators for that repository.

<p style="color: #c00000;"><strong>! Plagiarism check: all hand-ins will be checked for plagiarism.</strong></p>
