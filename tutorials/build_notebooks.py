#!/usr/bin/env python3
"""Build the tutorial ``.ipynb`` files from their jupytext percent-format ``.py``.

**Single source of truth.** Each tutorial is authored as one percent-format
``NN_*.py`` (the runnable, reviewable source). The committed ``NN_*.ipynb`` is
*generated* from it with ``jupytext``, never hand-edited. This replaces the old
bespoke ``_build_notebooks.py`` (which hardcoded every notebook's content twice,
once as a ``.py`` mirror and once as inline cell dicts) and the ad-hoc
``jupytext --to ipynb`` one-liners. See ``docs/deferred/DEF-0003``.

The generated ``.ipynb`` is deterministic and committed, so Colab/nbviewer links
resolve and CI can assert it is in sync with the source.

Usage (from ``public/`` or anywhere)::

    python tutorials/build_notebooks.py            # rebuild every notebook
    python tutorials/build_notebooks.py 04_ge_huggett.py   # just one
    python tutorials/build_notebooks.py --check    # CI: assert .ipynb in sync, no write

House style of the emitted notebook: no cell outputs, ``execution_count`` null,
metadata stripped to a bare ``kernelspec``, and **deterministic cell ids** (so
re-running the build is byte-stable and ``--check`` can diff).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import jupytext

HERE = Path(__file__).resolve().parent

# Files that are not jupytext sources, even though they live here.
SKIP = {"build_notebooks.py", "calibration.py"}

KERNELSPEC = {"display_name": "Python 3", "language": "python", "name": "python3"}

# --- Colab affordance injection (DEF-0008) -----------------------------------
# Each generated notebook gets two cells prepended: an "Open in Colab" badge
# (markdown) and a setup cell (code) that, *on Colab only*, clones the repo and
# installs the package. The notebook ``.py`` sources stay Colab-agnostic. The
# boilerplate lives here, in the single generator, so it re-points cleanly at
# the public cut by editing the seam below.
#
# The SUBDIR seam: the tutorials live at the repo root, so SUBDIR is empty and
# badge/clone paths point straight at the public repo.
REPO_OWNER = "StructuralRL"
REPO_NAME = "SRL"
SUBDIR = ""
DEFAULT_BRANCH = "main"

# nb00 is pure NumPy (calibration.py + a saved reference array, no srl/JAX): it
# needs the clone but not the package install. Every other notebook imports srl.
NO_SRL_INSTALL = {"00_household_numpy"}


def _repo_subpath(*parts: str) -> str:
    """A path *inside the cloned repo*.

    ``_repo_subpath()`` -> ``SRL``; ``_repo_subpath("tutorials")`` -> ``SRL/tutorials``.
    """
    return "/".join(p for p in (REPO_NAME, SUBDIR, *parts) if p)


def _blob_subpath(stem: str) -> str:
    """The notebook's path *from the repo root* (for the github/Colab blob URL)."""
    return "/".join(p for p in (SUBDIR, "tutorials", f"{stem}.ipynb") if p)


def _badge_cell(stem: str) -> str:
    """Markdown: the Colab badge."""
    url = (f"https://colab.research.google.com/github/{REPO_OWNER}/{REPO_NAME}"
           f"/blob/{DEFAULT_BRANCH}/{_blob_subpath(stem)}")
    return (
        f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})\n"
        "\n"
        "**Running on Colab?** Just run the setup cell below. It clones the repo and "
        "installs the package. For a free GPU: Runtime → Change runtime type → GPU."
    )


def _setup_cell(stem: str) -> str:
    """Code: clone + (optionally) install the package on Colab; a no-op locally.

    Local runs (Jupyter, ``jupytext``, plain ``python``) skip the whole block:
    ``calibration.py`` and ``data/`` already sit next to the notebook. On Colab
    the working directory is moved into ``tutorials/`` so the notebooks'
    ``from calibration import ...`` and relative ``data/`` paths resolve, and that
    directory is put on ``sys.path`` (Colab does not always carry the cwd there).
    The basename guard makes re-running the cell idempotent.
    """
    install = "" if stem in NO_SRL_INSTALL else (
        f'\n    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", '
        f'"{_repo_subpath()}"], check=True)'
        '\n    # Import srl straight from the clone. A mid-session `pip install -e`'
        '\n    # registers the package only through a .pth file that Python reads at'
        '\n    # startup, so it does not take effect in the already-running Colab'
        '\n    # kernel. Putting the repo root on sys.path makes `import srl` work now.'
        f'\n    sys.path.insert(0, os.path.abspath("{_repo_subpath()}"))')
    return (
        "# --- Colab setup (auto-injected by build_notebooks.py; do not edit the .ipynb) ---\n"
        "import os, sys, subprocess\n"
        "\n"
        'if "google.colab" in sys.modules and os.path.basename(os.getcwd()) != "tutorials":\n'
        "    # Clone the repo (this also brings calibration.py and the data/ reference\n"
        "    # files the notebooks load) and install the package.\n"
        f'    if not os.path.isdir("{REPO_NAME}"):\n'
        f'        url = "https://github.com/{REPO_OWNER}/{REPO_NAME}.git"\n'
        f'        subprocess.run(["git", "clone", "--depth=1", url, "{REPO_NAME}"], check=True)'
        f"{install}\n"
        f'    os.chdir("{_repo_subpath("tutorials")}")\n'
        "    sys.path.insert(0, os.getcwd())"
    )


def is_percent_source(path: Path) -> bool:
    """True if ``path`` is a jupytext percent-format notebook source."""
    if path.name in SKIP:
        return False
    head = path.read_text(encoding="utf-8")[:600]
    return "format_name: percent" in head or "\n# %%" in path.read_text(encoding="utf-8")


def _cell_id(stem: str, i: int) -> str:
    """Deterministic, notebook-stable cell id (so the build is reproducible)."""
    return hashlib.sha1(f"{stem}:{i}".encode()).hexdigest()[:12]


def render(py_path: Path) -> str:
    """Return the house-style ``.ipynb`` JSON text for a percent-format source."""
    nb = jupytext.read(py_path, fmt="py:percent")
    doc = json.loads(jupytext.writes(nb, fmt="ipynb"))   # -> standard nbformat dict

    doc["metadata"] = {"kernelspec": dict(KERNELSPEC)}    # drop jupytext/language_info noise
    doc["nbformat"], doc["nbformat_minor"] = 4, 5

    # Prepend the Colab badge (markdown) + setup (code) cells (see DEF-0008).
    stem = py_path.stem
    doc["cells"] = [
        {"cell_type": "markdown", "metadata": {}, "source": _badge_cell(stem)},
        {"cell_type": "code", "metadata": {}, "source": _setup_cell(stem)},
    ] + doc["cells"]

    for i, cell in enumerate(doc["cells"]):
        cell["metadata"] = {}
        cell["id"] = _cell_id(py_path.stem, i)
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    return json.dumps(doc, indent=1) + "\n"


def sources(selected: list[str]) -> list[Path]:
    if selected:
        paths = [HERE / s for s in selected]
        missing = [p for p in paths if not p.exists()]
        if missing:
            sys.exit(f"no such source: {', '.join(str(m) for m in missing)}")
        return paths
    return sorted(p for p in HERE.glob("*.py") if is_percent_source(p))


def main(argv: list[str]) -> int:
    check = "--check" in argv
    selected = [a for a in argv if not a.startswith("--")]
    paths = sources(selected)
    if not paths:
        sys.exit("no percent-format tutorial sources found")

    stale: list[str] = []
    for py_path in paths:
        text = render(py_path)
        ipynb_path = py_path.with_suffix(".ipynb")
        if check:
            current = ipynb_path.read_text(encoding="utf-8") if ipynb_path.exists() else None
            status = "ok" if current == text else "STALE"
            if status == "STALE":
                stale.append(ipynb_path.name)
            print(f"  [{status}] {ipynb_path.name}")
        else:
            ipynb_path.write_text(text, encoding="utf-8")
            print(f"  wrote {ipynb_path.name}  ({len(json.loads(text)['cells'])} cells)")

    if check and stale:
        print(f"\n{len(stale)} notebook(s) out of sync with their .py source: "
              f"{', '.join(stale)}\n  run: python tutorials/build_notebooks.py", file=sys.stderr)
        return 1
    print(f"\n{'checked' if check else 'built'} {len(paths)} notebook(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
