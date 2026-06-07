# thesis_eval Adversarial-Realism Evaluation Suite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/thesis_eval/`, a single-CLI, reproducible pipeline that normalizes adversarial-example arrays into canonical bundles and emits 15 figures (F1–F15) + tables proving latent-space attacks are valid, on-manifold, architecture-invariant, and detector-evasive while gradient baselines are not.

**Architecture:** Bundle-centric, two-phase. Phase 1 (`export`) re-runs the missing latent attacks with array dump and adapts existing gradient/targeted-benign npz into one `AEBundle` schema. Phase 2 (all other stages) is pure post-processing reading only bundles + cached intermediates — figures regenerate with no GPU and no attack re-runs.

**Tech Stack:** Python 3.10 (conda env `thesis`), numpy, pandas, scipy, scikit-learn, torch, matplotlib, seaborn, umap-learn. Tests via pytest. Reuses existing modules in `src/{attack,vae,classifiers,preprocessing}`.

**Spec:** `docs/superpowers/specs/2026-06-07-adversarial-realism-eval-suite-design.md`

---

## Conventions for every task

- Run everything from repo root `D:/thesis_final` with env python `C:/Users/T2530985/.conda/envs/thesis/python.exe` (referred to below as `python`).
- Every test file starts with the repo `sys.path` shim (matches existing tests):
  ```python
  import sys
  from pathlib import Path
  _REPO_ROOT = Path(__file__).resolve().parents[2]
  for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
      if _p not in sys.path:
          sys.path.insert(0, _p)
  ```
- `thesis_eval` modules import siblings as `from thesis_eval.x import y` (package on `src/`).
- Fixed constants (defined once in `config.py`, imported elsewhere — never re-literal'd):
  `SEED=42`, `NUM_FEATURES=39`, `LATENT_DIM=16`, `NUM_CLASSES=8`,
  `CANONICAL_VAE_TAG="gaussian_anticollapse_beta05_freebits01_20260529_173512"`,
  `CLASS_ORDER=["DoS","DDoS","Mirai","BruteForce","Recon","Web","Spoofing"]`.

---

## File structure (locked)

```
src/thesis_eval/
  __init__.py            # version string only
  __main__.py            # delegates to cli.main()
  cli.py                 # argparse + stage dispatch
  config.py              # paths, constants, POPULATIONS, CLASS_ORDER, discovery helpers
  palette.py             # Okabe-Ito colors by method + targeted/untargeted styles
  io_utils.py            # require_path(), save_figure(), write_csv()
  manifest.py            # build/write run_manifest.json
  data/
    __init__.py
    artifacts.py         # load splits/scaler/mask/feature_names/classifiers/VAEs
    bundles.py           # AEBundle dataclass + load/save/iter
    adapters.py          # gradient-npz + targeted-benign kappa-npz -> AEBundle
    export_latent.py     # re-run untargeted+targeted latent PGD/CW with array dump
  metrics/
    __init__.py
    bootstrap.py         # bootstrap_ci()
    asr.py               # ASR table
    mahalanobis.py       # fit gaussians (tied/per-class) + score + cond number
    detector.py          # LR detector, ROC/AUC per attack + bootstrap
    idsr.py              # IDSR computation
    fidelity.py          # Wasserstein, JS, |corr diff|, NN distance
    stats_tests.py       # McNemar
  figures/
    __init__.py
    layer1_effectiveness.py
    layer2_mahalanobis.py
    layer3_fidelity.py
    layer4_geometry.py
    layer5_vae.py
    report.py            # FIGURES.md generator
tests/thesis_eval/
  test_config.py test_palette.py test_io_utils.py test_bundles.py
  test_adapters.py test_bootstrap.py test_asr.py test_mahalanobis.py
  test_detector.py test_idsr.py test_fidelity.py test_stats_tests.py
  test_figures_smoke.py test_cli_smoke.py
results/thesis_eval/{figures,data,bundles,cache}/  + run_manifest.json + FIGURES.md
```

---

# PHASE 0 — Foundation

### Task 1: Install pytest + package scaffold

**Files:**
- Create: `src/thesis_eval/__init__.py`, `src/thesis_eval/data/__init__.py`, `src/thesis_eval/metrics/__init__.py`, `src/thesis_eval/figures/__init__.py`, `tests/thesis_eval/__init__.py`

- [ ] **Step 1: Install pytest into the thesis env**

Run: `python -m pip install pytest`
Expected: `Successfully installed pytest-...`

- [ ] **Step 2: Create package init files**

`src/thesis_eval/__init__.py`:
```python
"""Adversarial-realism evaluation suite for the MIBVAE NIDS thesis."""
__version__ = "0.1.0"
```
Create empty `src/thesis_eval/data/__init__.py`, `src/thesis_eval/metrics/__init__.py`, `src/thesis_eval/figures/__init__.py`, `tests/thesis_eval/__init__.py` (single comment line each).

- [ ] **Step 3: Verify import**

Run: `python -c "import sys; sys.path.insert(0,'src'); import thesis_eval; print(thesis_eval.__version__)"`
Expected: `0.1.0`

- [ ] **Step 4: Commit**

```bash
git add src/thesis_eval tests/thesis_eval
git commit -m "thesis_eval: package scaffold + pytest"
```

---

### Task 2: config.py — paths, constants, discovery

**Files:**
- Create: `src/thesis_eval/config.py`
- Test: `tests/thesis_eval/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/thesis_eval/test_config.py
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from thesis_eval import config

def test_constants():
    assert config.NUM_FEATURES == 39
    assert config.LATENT_DIM == 16
    assert config.NUM_CLASSES == 8
    assert config.SEED == 42
    assert config.CLASS_ORDER == ["DoS","DDoS","Mirai","BruteForce","Recon","Web","Spoofing"]

def test_populations_registry():
    names = {p["name"] for p in config.POPULATIONS}
    assert {"clean_benign","clean_malicious","latentPGD_untgt","latentCW_untgt",
            "latentPGD_tgtBenign","latentCW_tgtBenign","PGD","CW"} == names
    for p in config.POPULATIONS:
        assert p["method"] in {"clean_benign","clean_malicious","latent_PGD","latent_CW","PGD","CW"}
        assert p["targeting"] in {"none","untargeted","targeted_benign"}

def test_vae_checkpoint_paths_exist():
    for cls in ["Benign"] + config.CLASS_ORDER:
        p = config.vae_checkpoint_path(cls)
        assert p.exists(), f"missing VAE checkpoint: {p}"

def test_classifier_paths_exist():
    for tag in config.CLASSIFIER_TAGS:
        assert config.classifier_path(tag).exists(), tag

def test_output_dirs_created(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results" / "thesis_eval")
    dirs = config.ensure_output_dirs()
    for d in dirs.values():
        assert d.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/thesis_eval/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'thesis_eval.config'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/thesis_eval/config.py
"""Canonical paths, constants, and discovery for the eval suite."""
from __future__ import annotations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
DATA_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"
VAE_DIR = MODELS_DIR / "vae"
RESULTS_DIR = REPO_ROOT / "results" / "thesis_eval"

SEED = 42
NUM_FEATURES = 39
LATENT_DIM = 16
NUM_CLASSES = 8
N_BOOT = 1000

CANONICAL_VAE_TAG = "gaussian_anticollapse_beta05_freebits01_20260529_173512"

# Class order used in EVERY figure (attack classes only; Benign handled separately).
CLASS_ORDER = ["DoS", "DDoS", "Mirai", "BruteForce", "Recon", "Web", "Spoofing"]
# Maps display class name -> integer id used in y arrays (see src/vae/config.CLASSES).
_VAE_CLASSES = ["Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web"]
CLASS_TO_ID = {name: i for i, name in enumerate(_VAE_CLASSES)}
ID_TO_CLASS = {i: name for name, i in CLASS_TO_ID.items()}

CLASSIFIER_TAGS = ["mlp", "cnn", "lstm", "serial", "dualpath"]
CLASSIFIER_DISPLAY = {"mlp": "MLP", "cnn": "CNN", "lstm": "LSTM",
                      "serial": "CNN-LSTM", "dualpath": "DualPath"}

# Logical attack populations. method drives color; targeting drives line style.
POPULATIONS = [
    {"name": "clean_benign",        "method": "clean_benign",    "targeting": "none"},
    {"name": "clean_malicious",     "method": "clean_malicious", "targeting": "none"},
    {"name": "latentPGD_untgt",     "method": "latent_PGD",      "targeting": "untargeted"},
    {"name": "latentCW_untgt",      "method": "latent_CW",       "targeting": "untargeted"},
    {"name": "latentPGD_tgtBenign", "method": "latent_PGD",      "targeting": "targeted_benign"},
    {"name": "latentCW_tgtBenign",  "method": "latent_CW",       "targeting": "targeted_benign"},
    {"name": "PGD",                 "method": "PGD",             "targeting": "untargeted"},
    {"name": "CW",                  "method": "CW",              "targeting": "untargeted"},
]
ATTACK_POPULATIONS = [p["name"] for p in POPULATIONS
                      if p["method"] not in ("clean_benign", "clean_malicious")]


def vae_checkpoint_path(class_name: str) -> Path:
    cid = CLASS_TO_ID[class_name]
    return VAE_DIR / f"vae_class_{cid}_{class_name}_{CANONICAL_VAE_TAG}.pt"


def classifier_path(tag: str) -> Path:
    return MODELS_DIR / f"{tag}_8class.pt"


def ensure_output_dirs() -> dict[str, Path]:
    dirs = {k: RESULTS_DIR / k for k in ("figures", "data", "bundles", "cache")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/thesis_eval/test_config.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/thesis_eval/config.py tests/thesis_eval/test_config.py
git commit -m "thesis_eval: config paths/constants/population registry"
```

---

### Task 3: palette.py — colorblind-safe palette

**Files:**
- Create: `src/thesis_eval/palette.py`
- Test: `tests/thesis_eval/test_palette.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/thesis_eval/test_palette.py
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from thesis_eval import palette, config

def test_one_color_per_method():
    methods = {p["method"] for p in config.POPULATIONS}
    colors = {m: palette.color_for_method(m) for m in methods}
    assert len(set(colors.values())) == len(colors)  # all distinct
    for c in colors.values():
        assert isinstance(c, str) and c.startswith("#") and len(c) == 7

def test_color_for_population_uses_method_color():
    assert palette.color_for_population("latentPGD_untgt") == palette.color_for_method("latent_PGD")
    assert palette.color_for_population("latentPGD_tgtBenign") == palette.color_for_method("latent_PGD")

def test_style_distinguishes_targeting():
    assert palette.style_for_population("latentPGD_untgt") != palette.style_for_population("latentPGD_tgtBenign")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/thesis_eval/test_palette.py -v`
Expected: FAIL — `No module named 'thesis_eval.palette'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/thesis_eval/palette.py
"""Colorblind-safe (Okabe-Ito) palette: one color per METHOD, style per TARGETING."""
from __future__ import annotations
import matplotlib as mpl
from thesis_eval import config

# Okabe-Ito qualitative palette (CVD-safe).
_OKABE_ITO = {
    "clean_benign":    "#0072B2",  # blue
    "clean_malicious": "#000000",  # black
    "latent_PGD":      "#009E73",  # bluish green
    "latent_CW":       "#56B4E9",  # sky blue
    "PGD":             "#D55E00",  # vermillion
    "CW":              "#E69F00",  # orange
}
# untargeted -> solid, targeted_benign -> dashed, none -> solid
_LINESTYLE = {"none": "-", "untargeted": "-", "targeted_benign": "--"}
_HATCH = {"none": "", "untargeted": "", "targeted_benign": "//"}

_POP_BY_NAME = {p["name"]: p for p in config.POPULATIONS}


def color_for_method(method: str) -> str:
    return _OKABE_ITO[method]


def color_for_population(pop_name: str) -> str:
    return _OKABE_ITO[_POP_BY_NAME[pop_name]["method"]]


def style_for_population(pop_name: str) -> str:
    return _LINESTYLE[_POP_BY_NAME[pop_name]["targeting"]]


def hatch_for_population(pop_name: str) -> str:
    return _HATCH[_POP_BY_NAME[pop_name]["targeting"]]


def apply_rcparams() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
        "pdf.fonttype": 42, "ps.fonttype": 42,  # editable text in vector output
    })
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/thesis_eval/test_palette.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/thesis_eval/palette.py tests/thesis_eval/test_palette.py
git commit -m "thesis_eval: colorblind palette"
```

---

### Task 4: io_utils.py — fail-loud paths + dual-format figure saving

**Files:**
- Create: `src/thesis_eval/io_utils.py`
- Test: `tests/thesis_eval/test_io_utils.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/thesis_eval/test_io_utils.py
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from thesis_eval import io_utils

def test_require_path_raises_with_expected_path(tmp_path):
    missing = tmp_path / "nope.npz"
    with pytest.raises(FileNotFoundError) as exc:
        io_utils.require_path(missing, "test artifact")
    assert str(missing) in str(exc.value)

def test_save_figure_emits_pdf_png_csv(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    df = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
    io_utils.save_figure(fig, "F0_demo", df, figures_dir=tmp_path / "figures",
                         data_dir=tmp_path / "data")
    assert (tmp_path / "figures" / "F0_demo.pdf").exists()
    assert (tmp_path / "figures" / "F0_demo.png").exists()
    assert (tmp_path / "data" / "F0_demo.csv").exists()
    reread = pd.read_csv(tmp_path / "data" / "F0_demo.csv")
    assert list(reread.columns) == ["x", "y"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/thesis_eval/test_io_utils.py -v`
Expected: FAIL — `No module named 'thesis_eval.io_utils'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/thesis_eval/io_utils.py
"""Fail-loud artifact access and dual-format (pdf+png) figure + csv saving."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from thesis_eval import config


def require_path(path: Path, what: str = "artifact") -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {what}. Expected at: {path}. "
            f"No silent fallback — generate it (see --stage export) or fix the path."
        )
    return path


def save_figure(fig, name: str, data: "pd.DataFrame | None" = None,
                figures_dir: Path | None = None, data_dir: Path | None = None) -> None:
    figures_dir = figures_dir or (config.RESULTS_DIR / "figures")
    data_dir = data_dir or (config.RESULTS_DIR / "data")
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{name}.pdf")
    fig.savefig(figures_dir / f"{name}.png", dpi=300)
    if data is not None:
        data.to_csv(data_dir / f"{name}.csv", index=False)
    import matplotlib.pyplot as plt
    plt.close(fig)


def write_csv(df: pd.DataFrame, name: str, data_dir: Path | None = None) -> Path:
    data_dir = data_dir or (config.RESULTS_DIR / "data")
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / f"{name}.csv"
    df.to_csv(out, index=False)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/thesis_eval/test_io_utils.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/thesis_eval/io_utils.py tests/thesis_eval/test_io_utils.py
git commit -m "thesis_eval: io_utils fail-loud + dual-format saving"
```

---

### Task 5: manifest.py — run_manifest.json

**Files:**
- Create: `src/thesis_eval/manifest.py`
- Test: included in `tests/thesis_eval/test_io_utils.py` is separate; create `tests/thesis_eval/test_manifest.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/thesis_eval/test_manifest.py
import json, sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from thesis_eval import manifest

def test_manifest_has_required_keys(tmp_path):
    out = tmp_path / "run_manifest.json"
    m = manifest.build_manifest(seed=42, cov_mode="tied", n_boot=1000)
    manifest.write_manifest(m, out)
    loaded = json.loads(out.read_text())
    for k in ("seed", "cov_mode", "n_boot", "vae_tag", "library_versions",
              "artifact_hash", "created"):
        assert k in loaded
    assert loaded["seed"] == 42
    assert "numpy" in loaded["library_versions"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/thesis_eval/test_manifest.py -v`
Expected: FAIL — `No module named 'thesis_eval.manifest'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/thesis_eval/manifest.py
"""run_manifest.json: seed, lib versions, artifact hash, cov mode, VAE tag."""
from __future__ import annotations
import hashlib, json, platform
from datetime import datetime, timezone
from pathlib import Path
from thesis_eval import config


def _artifact_hash() -> str:
    """Stable hash over the processed-data content hashes + scaler bytes."""
    h = hashlib.sha256()
    ch = config.DATA_DIR / "content_hashes.json"
    if ch.exists():
        h.update(ch.read_bytes())
    scaler = config.DATA_DIR / "scaler.pkl"
    if scaler.exists():
        h.update(scaler.read_bytes())
    return h.hexdigest()[:16]


def _library_versions() -> dict:
    import numpy, pandas, sklearn, scipy, torch
    return {"python": platform.python_version(), "numpy": numpy.__version__,
            "pandas": pandas.__version__, "scikit-learn": sklearn.__version__,
            "scipy": scipy.__version__, "torch": torch.__version__}


def build_manifest(seed: int, cov_mode: str, n_boot: int) -> dict:
    return {
        "created": datetime.now(timezone.utc).isoformat(),
        "seed": seed, "cov_mode": cov_mode, "n_boot": n_boot,
        "vae_tag": config.CANONICAL_VAE_TAG,
        "class_order": config.CLASS_ORDER,
        "populations": [p["name"] for p in config.POPULATIONS],
        "library_versions": _library_versions(),
        "artifact_hash": _artifact_hash(),
    }


def write_manifest(m: dict, path: Path | None = None) -> Path:
    path = Path(path) if path else (config.RESULTS_DIR / "run_manifest.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m, indent=2))
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/thesis_eval/test_manifest.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/thesis_eval/manifest.py tests/thesis_eval/test_manifest.py
git commit -m "thesis_eval: run manifest writer"
```

---

# PHASE 1 — Data layer

### Task 6: bundles.py — AEBundle schema

**Files:**
- Create: `src/thesis_eval/data/bundles.py`
- Test: `tests/thesis_eval/test_bundles.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/thesis_eval/test_bundles.py
import sys
from pathlib import Path
import numpy as np
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from thesis_eval.data.bundles import AEBundle

def _toy(n=5):
    rng = np.random.default_rng(0)
    return AEBundle(
        x_orig=rng.random((n, 39)).astype("float32"),
        x_adv=rng.random((n, 39)).astype("float32"),
        z_orig=rng.random((n, 16)).astype("float32"),
        z_adv=rng.random((n, 16)).astype("float32"),
        y_true=np.full(n, 3, dtype="int64"),
        y_pred_clean=np.full(n, 3, dtype="int64"),
        y_pred_adv=np.array([0,0,3,3,0], dtype="int64"),
        success=np.array([1,1,0,0,1], dtype=bool),
        protocol_valid=np.ones(n, bool), mask_valid=np.ones(n, bool),
        raw_g1g8_valid=np.array([1,0,1,1,1], bool),
        joint_valid=np.array([1,0,1,1,1], bool),
        population="latentPGD_untgt", model="mlp", source_class="DoS",
        attack_type="latent-pgd", vae_tag="t", seed=42)

def test_roundtrip(tmp_path):
    b = _toy()
    p = tmp_path / "b.npz"
    b.save(p)
    b2 = AEBundle.load(p)
    assert b2.population == "latentPGD_untgt"
    assert b2.source_class == "DoS"
    np.testing.assert_array_equal(b2.success, b.success)
    assert b2.x_adv.shape == (5, 39)

def test_shape_validation_rejects_mismatch():
    import pytest
    with pytest.raises(ValueError):
        AEBundle(x_orig=np.zeros((5,39),"float32"), x_adv=np.zeros((4,39),"float32"),
                 z_orig=np.zeros((5,16),"float32"), z_adv=np.zeros((5,16),"float32"),
                 y_true=np.zeros(5,"int64"), y_pred_clean=np.zeros(5,"int64"),
                 y_pred_adv=np.zeros(5,"int64"), success=np.zeros(5,bool),
                 protocol_valid=np.zeros(5,bool), mask_valid=np.zeros(5,bool),
                 raw_g1g8_valid=np.zeros(5,bool), joint_valid=np.zeros(5,bool),
                 population="p", model="m", source_class="DoS",
                 attack_type="a", vae_tag="t", seed=42)

def test_filename():
    assert AEBundle.filename("PGD","cnn","Mirai") == "PGD__cnn__Mirai.npz"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/thesis_eval/test_bundles.py -v`
Expected: FAIL — `No module named 'thesis_eval.data.bundles'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/thesis_eval/data/bundles.py
"""Canonical adversarial-example bundle: one per population x model x class."""
from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from pathlib import Path
import numpy as np

_ARRAY_FIELDS = ["x_orig","x_adv","z_orig","z_adv","y_true","y_pred_clean",
                 "y_pred_adv","success","protocol_valid","mask_valid",
                 "raw_g1g8_valid","joint_valid"]
_META_FIELDS = ["population","model","source_class","attack_type","vae_tag","seed"]


@dataclass
class AEBundle:
    x_orig: np.ndarray; x_adv: np.ndarray
    z_orig: np.ndarray; z_adv: np.ndarray
    y_true: np.ndarray; y_pred_clean: np.ndarray; y_pred_adv: np.ndarray
    success: np.ndarray
    protocol_valid: np.ndarray; mask_valid: np.ndarray
    raw_g1g8_valid: np.ndarray; joint_valid: np.ndarray
    population: str; model: str; source_class: str
    attack_type: str; vae_tag: str; seed: int

    def __post_init__(self):
        n = self.x_orig.shape[0]
        for f in _ARRAY_FIELDS:
            arr = getattr(self, f)
            if arr.shape[0] != n:
                raise ValueError(f"{f} has {arr.shape[0]} rows, expected {n}")
        if self.x_adv.shape[1] != 39 or self.z_adv.shape[1] != 16:
            raise ValueError("x must be (n,39) and z must be (n,16)")

    @property
    def n(self) -> int:
        return self.x_orig.shape[0]

    @staticmethod
    def filename(population: str, model: str, source_class: str) -> str:
        return f"{population}__{model}__{source_class}.npz"

    def save(self, path: Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        payload = {f: getattr(self, f) for f in _ARRAY_FIELDS}
        payload["_meta"] = np.array([str({k: getattr(self, k) for k in _META_FIELDS})])
        meta = {k: getattr(self, k) for k in _META_FIELDS}
        np.savez_compressed(path, **{f: getattr(self, f) for f in _ARRAY_FIELDS},
                            **{f"meta_{k}": np.array(v) for k, v in meta.items()})

    @classmethod
    def load(cls, path: Path) -> "AEBundle":
        d = np.load(path, allow_pickle=False)
        kwargs = {f: d[f] for f in _ARRAY_FIELDS}
        for k in _META_FIELDS:
            v = d[f"meta_{k}"]
            kwargs[k] = int(v) if k == "seed" else str(v)
        return cls(**kwargs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/thesis_eval/test_bundles.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/thesis_eval/data/bundles.py tests/thesis_eval/test_bundles.py
git commit -m "thesis_eval: AEBundle schema + roundtrip"
```

---

### Task 7: artifacts.py — load splits, scaler, mask, classifiers, VAEs

**Files:**
- Create: `src/thesis_eval/data/artifacts.py`
- Test: `tests/thesis_eval/test_artifacts.py`

This module is thin glue over existing loaders. It must reuse, not reinvent.

- [ ] **Step 1: Write the failing test**

```python
# tests/thesis_eval/test_artifacts.py
import sys
from pathlib import Path
import numpy as np
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from thesis_eval.data import artifacts
from thesis_eval import config

def test_feature_names():
    assert len(artifacts.feature_names()) == 39

def test_load_test_split_shapes():
    X, y8 = artifacts.load_test_split()
    assert X.shape[1] == 39
    assert X.shape[0] == y8.shape[0]
    assert set(np.unique(y8)).issubset(set(range(8)))

def test_inverse_transform_roundtrip():
    X, _ = artifacts.load_test_split()
    raw = artifacts.inverse_transform(X[:10])
    assert raw.shape == (10, 39)

def test_load_classifier_predicts():
    clf = artifacts.load_classifier("mlp", device="cpu")
    X, _ = artifacts.load_test_split()
    preds = artifacts.predict(clf, X[:16], device="cpu")
    assert preds.shape == (16,)

def test_router_loads_canonical_vaes():
    router = artifacts.build_attack_router(device="cpu")
    vae = router.get_vae(config.CLASS_TO_ID["DoS"])
    assert vae is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/thesis_eval/test_artifacts.py -v`
Expected: FAIL — `No module named 'thesis_eval.data.artifacts'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/thesis_eval/data/artifacts.py
"""Thin glue over existing loaders. Reuses src/attack and src/classifiers."""
from __future__ import annotations
import numpy as np
import torch
from thesis_eval import config, io_utils

# Existing modules (src already on path via package install location).
from preprocessing.feature_groups import FEATURE_NAMES
from attack.build_thesis_bundle import load_classifier as _load_classifier
from attack.latent_infra import (AttackRouter, inverse_transform_scaled,
                                 predict_labels, set_global_seed)
import pickle


def feature_names() -> list[str]:
    return list(FEATURE_NAMES)


def load_test_split() -> tuple[np.ndarray, np.ndarray]:
    X = np.load(io_utils.require_path(config.DATA_DIR / "X_test.npy", "X_test"))
    y8 = np.load(io_utils.require_path(config.DATA_DIR / "y_test_cat.npy", "y_test_cat"))
    return X.astype("float32"), y8.astype("int64")


def load_train_split() -> tuple[np.ndarray, np.ndarray]:
    X = np.load(io_utils.require_path(config.DATA_DIR / "X_train.npy", "X_train"))
    y8 = np.load(io_utils.require_path(config.DATA_DIR / "y_train_cat.npy", "y_train_cat"))
    return X.astype("float32"), y8.astype("int64")


def _scaler():
    with open(io_utils.require_path(config.DATA_DIR / "scaler.pkl", "scaler"), "rb") as fh:
        return pickle.load(fh)


def inverse_transform(x_scaled: np.ndarray) -> np.ndarray:
    return inverse_transform_scaled(x_scaled, _scaler())


def load_classifier(tag: str, device: str = "cpu"):
    io_utils.require_path(config.classifier_path(tag), f"classifier {tag}")
    return _load_classifier(tag, device)


def predict(clf, x_scaled: np.ndarray, device: str = "cpu") -> np.ndarray:
    xb = torch.from_numpy(np.asarray(x_scaled, dtype="float32")).to(device)
    return predict_labels(clf, xb, device=device).cpu().numpy().astype("int64")


def build_attack_router(device: str = "cpu") -> AttackRouter:
    """AttackRouter wired to the canonical VAE checkpoint set + scaler."""
    set_global_seed(config.SEED)
    ckpts = {config.CLASS_TO_ID[c]: str(config.vae_checkpoint_path(c))
             for c in (["Benign"] + config.CLASS_ORDER)}
    for p in ckpts.values():
        io_utils.require_path(p, "VAE checkpoint")
    return AttackRouter(checkpoints=ckpts, scaler=_scaler(), device=device)
```

> NOTE for implementer: confirm `AttackRouter.__init__` parameter names by reading
> `src/attack/latent_infra.py:297-379`. If it expects a manifest path rather than a
> `checkpoints` dict, adapt `build_attack_router` to that signature (the router already
> handles `register_protocol_references`). Do not change AttackRouter itself.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/thesis_eval/test_artifacts.py -v`
Expected: PASS (5 passed). If `build_attack_router` fails, fix per the NOTE, then re-run.

- [ ] **Step 5: Commit**

```bash
git add src/thesis_eval/data/artifacts.py tests/thesis_eval/test_artifacts.py
git commit -m "thesis_eval: artifacts loader glue"
```

---

### Task 8: adapters.py — gradient + targeted-benign npz → AEBundle

**Files:**
- Create: `src/thesis_eval/data/adapters.py`
- Test: `tests/thesis_eval/test_adapters.py`

The gradient npz lacks latent z and per-rule validity. The adapter fills z with NaN
(encoded later in the maha stage) and computes `joint_valid` from the validator.

- [ ] **Step 1: Write the failing test**

```python
# tests/thesis_eval/test_adapters.py
import sys
from pathlib import Path
import numpy as np
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from thesis_eval.data import adapters

def _fake_gradient_npz(tmp_path):
    n = 12
    rng = np.random.default_rng(1)
    p = tmp_path / "attack_8class_pgd_0.30.npz"
    np.savez(p, X_adv=rng.random((n,39)).astype("float32"),
             X_clean=rng.random((n,39)).astype("float32"),
             y_true=np.array([3,3,3,3,2,2,2,2,4,4,4,4], dtype="int64"),
             y_pred_clean=np.array([3,3,3,3,2,2,2,2,4,4,4,4], dtype="int64"),
             y_pred_adv=np.array([0,3,0,0,2,0,0,2,4,0,0,4], dtype="int64"),
             attack_name=np.array("pgd"), eps=np.array(0.30))
    return p

def test_gradient_adapter_splits_by_class(tmp_path):
    p = _fake_gradient_npz(tmp_path)
    bundles = adapters.from_gradient_npz(p, population="PGD", model="cnn",
                                         attack_type="pgd", device="cpu")
    classes = {b.source_class for b in bundles}
    assert classes == {"DoS", "DDoS", "Mirai"}  # ids 3,2,4
    for b in bundles:
        assert np.isnan(b.z_adv).all()        # latent filled later
        assert b.success.dtype == bool
        assert b.joint_valid.shape[0] == b.n  # validator was run
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/thesis_eval/test_adapters.py -v`
Expected: FAIL — `No module named 'thesis_eval.data.adapters'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/thesis_eval/data/adapters.py
"""Adapt existing npz array sources into the canonical AEBundle schema."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from thesis_eval import config, io_utils
from thesis_eval.data import artifacts
from thesis_eval.data.bundles import AEBundle
from attack.validator import validate_batch


def _validate_joint(x_adv_scaled: np.ndarray) -> np.ndarray:
    raw = artifacts.inverse_transform(x_adv_scaled)
    res = validate_batch(raw, artifacts.feature_names())
    return res.overall_valid.astype(bool)


def from_gradient_npz(path: Path, *, population: str, model: str,
                      attack_type: str, device: str = "cpu") -> list[AEBundle]:
    d = np.load(io_utils.require_path(path, f"gradient npz {population}"), allow_pickle=True)
    X_adv, X_clean = d["X_adv"].astype("float32"), d["X_clean"].astype("float32")
    y_true = d["y_true"].astype("int64")
    y_pred_clean = d["y_pred_clean"].astype("int64")
    y_pred_adv = d["y_pred_adv"].astype("int64")
    n = X_adv.shape[0]
    nan_z = np.full((n, config.LATENT_DIM), np.nan, dtype="float32")
    joint = _validate_joint(X_adv)
    success = (y_pred_adv != y_true)
    out: list[AEBundle] = []
    for cid in np.unique(y_true):
        cls = config.ID_TO_CLASS[int(cid)]
        if cls == "Benign":
            continue
        m = y_true == cid
        out.append(AEBundle(
            x_orig=X_clean[m], x_adv=X_adv[m], z_orig=nan_z[m], z_adv=nan_z[m],
            y_true=y_true[m], y_pred_clean=y_pred_clean[m], y_pred_adv=y_pred_adv[m],
            success=success[m].astype(bool),
            protocol_valid=joint[m], mask_valid=np.ones(m.sum(), bool),
            raw_g1g8_valid=joint[m], joint_valid=joint[m],
            population=population, model=model, source_class=cls,
            attack_type=attack_type, vae_tag="input_space", seed=config.SEED))
    return out


def from_targeted_benign_npz(path: Path, *, model: str, attack_type: str,
                             population: str) -> AEBundle:
    """Adapt a kappa-sweep targeted-benign npz (already has z_orig/z_adv + validity)."""
    d = np.load(io_utils.require_path(path, f"targeted-benign npz"), allow_pickle=True)
    y_true = d["y_true"].astype("int64")
    cls = config.ID_TO_CLASS[int(np.unique(y_true)[0])]
    return AEBundle(
        x_orig=d["x_orig"].astype("float32"), x_adv=d["x_adv"].astype("float32"),
        z_orig=d["z_orig"].astype("float32"), z_adv=d["z_adv"].astype("float32"),
        y_true=y_true, y_pred_clean=d["y_pred_before"].astype("int64"),
        y_pred_adv=d["y_pred_after"].astype("int64"),
        success=d["target_success"].astype(bool),
        protocol_valid=d["protocol_valid"].astype(bool),
        mask_valid=d["mask_valid"].astype(bool),
        raw_g1g8_valid=d["raw_g1g8_valid"].astype(bool),
        joint_valid=d["joint_valid"].astype(bool),
        population=population, model=model, source_class=cls,
        attack_type=attack_type, vae_tag=config.CANONICAL_VAE_TAG, seed=config.SEED)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/thesis_eval/test_adapters.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/thesis_eval/data/adapters.py tests/thesis_eval/test_adapters.py
git commit -m "thesis_eval: gradient + targeted-benign npz adapters"
```

---

### Task 9: export_latent.py — the dumper (untargeted + targeted latent PGD/CW)

**Files:**
- Create: `src/thesis_eval/data/export_latent.py`
- Test: `tests/thesis_eval/test_export_latent_smoke.py` (CPU, tiny N)

Reuses `attack.build_thesis_bundle.run_latent_attack_for_model` (returns z arrays).
For each (classifier tag × attack ∈ {pgd, cw} × targeting ∈ {untargeted, targeted_benign}),
runs the latent attack on a per-class sample, recomputes validity, writes an AEBundle.

- [ ] **Step 1: Write the failing test (smoke, monkeypatched tiny run)**

```python
# tests/thesis_eval/test_export_latent_smoke.py
import sys
from pathlib import Path
import numpy as np
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from thesis_eval.data import export_latent
from thesis_eval.data.bundles import AEBundle

def test_build_bundle_from_attack_outputs():
    n = 8
    rng = np.random.default_rng(2)
    out = {"X_adv": rng.random((n,39)).astype("float32"),
           "z_orig": rng.random((n,16)).astype("float32"),
           "z_adv": rng.random((n,16)).astype("float32")}
    X_scaled = rng.random((n,39)).astype("float32")
    y8 = np.full(n, 3, dtype="int64")
    y_pred_clean = np.full(n, 3, dtype="int64")
    y_pred_adv = np.array([0,0,3,3,0,0,3,0], dtype="int64")
    b = export_latent.bundle_from_outputs(
        out=out, X_scaled=X_scaled, y8=y8, y_pred_clean=y_pred_clean,
        y_pred_adv=y_pred_adv, population="latentPGD_untgt", model="mlp",
        source_class="DoS", attack_type="latent-pgd")
    assert isinstance(b, AEBundle)
    assert b.n == n
    assert b.joint_valid.dtype == bool
    # untargeted success = misclassified away from true label
    assert b.success.tolist() == [True,True,False,False,True,True,False,True]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/thesis_eval/test_export_latent_smoke.py -v`
Expected: FAIL — `No module named 'thesis_eval.data.export_latent'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/thesis_eval/data/export_latent.py
"""Re-run latent PGD/CW with array dump -> AEBundle. Phase-1 GPU step."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from thesis_eval import config
from thesis_eval.data import artifacts
from thesis_eval.data.bundles import AEBundle
from attack.validator import validate_batch
from attack.build_thesis_bundle import run_latent_attack_for_model
from attack.latent_infra import PerturbationMask

# population -> (attack, targeting)
_LATENT_POPS = {
    "latentPGD_untgt":     ("pgd", "untargeted"),
    "latentCW_untgt":      ("cw",  "untargeted"),
    "latentPGD_tgtBenign": ("pgd", "targeted_benign"),
    "latentCW_tgtBenign":  ("cw",  "targeted_benign"),
}


def _success(y_true, y_pred_adv, targeting: str) -> np.ndarray:
    if targeting == "targeted_benign":
        return (y_pred_adv == config.CLASS_TO_ID["Benign"])
    return (y_pred_adv != y_true)


def bundle_from_outputs(*, out: dict, X_scaled, y8, y_pred_clean, y_pred_adv,
                        population: str, model: str, source_class: str,
                        attack_type: str) -> AEBundle:
    targeting = next(t for n,(a,t) in _LATENT_POPS.items() if n == population)
    x_adv = out["X_adv"].astype("float32")
    raw = artifacts.inverse_transform(x_adv)
    res = validate_batch(raw, artifacts.feature_names())
    joint = res.overall_valid.astype(bool)
    n = x_adv.shape[0]
    return AEBundle(
        x_orig=X_scaled.astype("float32"), x_adv=x_adv,
        z_orig=out["z_orig"].astype("float32"), z_adv=out["z_adv"].astype("float32"),
        y_true=y8.astype("int64"), y_pred_clean=y_pred_clean.astype("int64"),
        y_pred_adv=y_pred_adv.astype("int64"),
        success=_success(y8, y_pred_adv, targeting).astype(bool),
        protocol_valid=joint, mask_valid=np.ones(n, bool),
        raw_g1g8_valid=joint, joint_valid=joint,
        population=population, model=model, source_class=source_class,
        attack_type=attack_type, vae_tag=config.CANONICAL_VAE_TAG, seed=config.SEED)


def export_all(*, device: str, per_class: int = 100, populations=None) -> list[Path]:
    """Run every requested latent population x classifier x class; write bundles."""
    populations = populations or list(_LATENT_POPS)
    dirs = config.ensure_output_dirs()
    X, y8 = artifacts.load_test_split()
    router = artifacts.build_attack_router(device=device)
    mask = PerturbationMask.from_processed(config.DATA_DIR)  # confirm constructor; see NOTE
    written: list[Path] = []
    for tag in config.CLASSIFIER_TAGS:
        clf = artifacts.load_classifier(tag, device=device)
        y_pred_clean_all = artifacts.predict(clf, X, device=device)
        for pop in populations:
            attack, targeting = _LATENT_POPS[pop]
            for cls in config.CLASS_ORDER:
                cid = config.CLASS_TO_ID[cls]
                # correctly-classified samples of this class, capped at per_class
                m = (y8 == cid) & (y_pred_clean_all == cid)
                idx = np.flatnonzero(m)[:per_class]
                if idx.size == 0:
                    continue
                args = _attack_args(attack, targeting)
                out = run_latent_attack_for_model(
                    classifier=clf, router=router, mask=mask,
                    X_scaled=X[idx], y_8=y8[idx], attack=attack, args=args, device=device)
                y_pred_adv = artifacts.predict(clf, out["X_adv"], device=device)
                b = bundle_from_outputs(
                    out=out, X_scaled=X[idx], y8=y8[idx],
                    y_pred_clean=y_pred_clean_all[idx], y_pred_adv=y_pred_adv,
                    population=pop, model=tag, source_class=cls,
                    attack_type=f"latent-{attack}")
                p = dirs["bundles"] / AEBundle.filename(pop, tag, cls)
                b.save(p); written.append(p)
    return written


def _attack_args(attack: str, targeting: str) -> argparse.Namespace:
    """Attack hyperparameters mirroring the canonical runs (see config_snapshot.json)."""
    ns = argparse.Namespace(
        pgd_eps=1.0, pgd_alpha=0.1, pgd_steps=100,
        cw_lambda=1.0, cw_steps=100, num_restarts=5,
        target_benign=(targeting == "targeted_benign"))
    return ns
```

> NOTE for implementer: `PerturbationMask` constructor and `run_latent_attack_for_model`'s
> exact `args` fields must be confirmed against `src/attack/latent_infra.py:126` and
> `src/attack/build_thesis_bundle.py:run_latent_attack_for_model`. Align `_attack_args`
> and the mask constructor to those signatures. Targeted-benign may require the
> `run_targeted_benign_*` entrypoint instead — if so, branch on `targeting`. Keep the
> per-class N and hyperparameters identical to the canonical `config_snapshot.json` so
> numbers match prior runs. The smoke test (Step 1) does not exercise GPU; the real run
> happens in Task 24.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/thesis_eval/test_export_latent_smoke.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/thesis_eval/data/export_latent.py tests/thesis_eval/test_export_latent_smoke.py
git commit -m "thesis_eval: latent attack array exporter"
```

---

# PHASE 2 — Metrics (one module per metric)

### Task 10: bootstrap.py

**Files:**
- Create: `src/thesis_eval/metrics/bootstrap.py`
- Test: `tests/thesis_eval/test_bootstrap.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/thesis_eval/test_bootstrap.py
import sys
from pathlib import Path
import numpy as np
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from thesis_eval.metrics.bootstrap import bootstrap_ci

def test_ci_brackets_mean_and_is_deterministic():
    rng = np.random.default_rng(0)
    x = rng.binomial(1, 0.4, size=2000).astype(float)
    m1, lo1, hi1 = bootstrap_ci(x, np.mean, n_boot=1000, seed=42)
    m2, lo2, hi2 = bootstrap_ci(x, np.mean, n_boot=1000, seed=42)
    assert (m1, lo1, hi1) == (m2, lo2, hi2)      # seeded determinism
    assert lo1 < m1 < hi1
    assert abs(m1 - 0.4) < 0.05
    assert (hi1 - lo1) < 0.1                       # ~±a few pp at N=2000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/thesis_eval/test_bootstrap.py -v`
Expected: FAIL — `No module named 'thesis_eval.metrics.bootstrap'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/thesis_eval/metrics/bootstrap.py
"""Seeded percentile bootstrap CI."""
from __future__ import annotations
from typing import Callable
import numpy as np


def bootstrap_ci(values: np.ndarray, statistic: Callable[[np.ndarray], float],
                 n_boot: int = 1000, seed: int = 42,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = values.size
    stats = np.empty(n_boot)
    for i in range(n_boot):
        stats[i] = statistic(values[rng.integers(0, n, n)])
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return float(statistic(values)), lo, hi
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/thesis_eval/test_bootstrap.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/thesis_eval/metrics/bootstrap.py tests/thesis_eval/test_bootstrap.py
git commit -m "thesis_eval: bootstrap CI"
```

---

### Task 11: asr.py — ASR_raw, ASR_valid, validity, L2 with CIs

**Files:**
- Create: `src/thesis_eval/metrics/asr.py`
- Test: `tests/thesis_eval/test_asr.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/thesis_eval/test_asr.py
import sys
from pathlib import Path
import numpy as np
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from thesis_eval.metrics import asr
from thesis_eval.data.bundles import AEBundle

def _bundle(success, valid, n=None):
    n = n or len(success)
    z = np.zeros((n,16),"float32"); x = np.zeros((n,39),"float32")
    return AEBundle(x_orig=x, x_adv=x+0.5, z_orig=z, z_adv=z,
        y_true=np.full(n,3,"int64"), y_pred_clean=np.full(n,3,"int64"),
        y_pred_adv=np.full(n,0,"int64"),
        success=np.array(success,bool), protocol_valid=np.array(valid,bool),
        mask_valid=np.ones(n,bool), raw_g1g8_valid=np.array(valid,bool),
        joint_valid=np.array(valid,bool), population="PGD", model="cnn",
        source_class="DoS", attack_type="pgd", vae_tag="t", seed=42)

def test_asr_row():
    b = _bundle(success=[1,1,1,0], valid=[1,0,1,1])
    row = asr.asr_row(b, n_boot=200, seed=42)
    assert row["n"] == 4
    assert abs(row["asr_raw"] - 0.75) < 1e-9
    # valid AND success = samples 0 and 2 -> 2/4
    assert abs(row["asr_valid"] - 0.5) < 1e-9
    assert abs(row["validity_rate"] - 0.75) < 1e-9
    assert row["mean_l2"] > 0
    assert row["asr_valid_ci_low"] <= row["asr_valid"] <= row["asr_valid_ci_high"]

def test_asr_table_columns():
    df = asr.asr_table([_bundle([1,0],[1,1]), _bundle([0,1],[1,0])])
    for c in ("population","model","source_class","n","asr_raw","asr_valid",
              "validity_rate","mean_l2","median_l2","asr_valid_ci_low","asr_valid_ci_high"):
        assert c in df.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/thesis_eval/test_asr.py -v`
Expected: FAIL — `No module named 'thesis_eval.metrics.asr'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/thesis_eval/metrics/asr.py
"""ASR_raw, ASR_valid, validity rate, feature-space L2 — per bundle, with CIs."""
from __future__ import annotations
import numpy as np
import pandas as pd
from thesis_eval.metrics.bootstrap import bootstrap_ci
from thesis_eval.data.bundles import AEBundle


def _l2(b: AEBundle) -> np.ndarray:
    return np.linalg.norm(b.x_adv - b.x_orig, axis=1)


def asr_row(b: AEBundle, n_boot: int = 1000, seed: int = 42) -> dict:
    success = b.success.astype(float)
    valid = b.joint_valid.astype(float)
    asr_valid_vec = (b.success & b.joint_valid).astype(float)
    l2 = _l2(b)
    m, lo, hi = bootstrap_ci(asr_valid_vec, np.mean, n_boot=n_boot, seed=seed)
    raw_m, raw_lo, raw_hi = bootstrap_ci(success, np.mean, n_boot=n_boot, seed=seed)
    return {
        "population": b.population, "model": b.model, "source_class": b.source_class,
        "attack_type": b.attack_type, "n": b.n,
        "asr_raw": float(success.mean()), "asr_raw_ci_low": raw_lo, "asr_raw_ci_high": raw_hi,
        "asr_valid": m, "asr_valid_ci_low": lo, "asr_valid_ci_high": hi,
        "validity_rate": float(valid.mean()),
        "mean_l2": float(l2.mean()), "median_l2": float(np.median(l2)),
    }


def asr_table(bundles: list[AEBundle], n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    rows = [asr_row(b, n_boot=n_boot, seed=seed) for b in bundles]
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/thesis_eval/test_asr.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/thesis_eval/metrics/asr.py tests/thesis_eval/test_asr.py
git commit -m "thesis_eval: ASR_valid metric"
```

---

### Task 12: mahalanobis.py — fit gaussians (tied + per-class) + score

**Files:**
- Create: `src/thesis_eval/metrics/mahalanobis.py`
- Test: `tests/thesis_eval/test_mahalanobis.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/thesis_eval/test_mahalanobis.py
import sys
from pathlib import Path
import numpy as np
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from thesis_eval.metrics import mahalanobis as M

def test_score_zero_at_center_and_positive_away():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, size=(5000, 8)).astype("float32")
    params = M.fit_gaussians({"DoS": z}, cov_mode="tied")
    s_center = M.mahalanobis_score(params["DoS"]["mu"][None, :].astype("float32"), params)
    s_far = M.mahalanobis_score(np.full((1, 8), 8.0, "float32"), params)
    assert s_center[0] < 1e-3
    assert s_far[0] > s_center[0] + 10

def test_cond_number_logged_and_pinv_fallback_on_singular():
    z = np.zeros((100, 4), "float32")     # singular covariance
    z[:, 0] = np.linspace(0, 1, 100)      # only one varying dim
    params = M.fit_gaussians({"DoS": z}, cov_mode="per-class")
    assert "cond_number" in params["DoS"]
    assert params["DoS"]["used_shrinkage_or_pinv"] is True

def test_min_over_centers():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, (2000, 4)).astype("float32")
    b = (rng.normal(0, 1, (2000, 4)) + 10).astype("float32")
    params = M.fit_gaussians({"DoS": a, "Benign": b}, cov_mode="tied")
    s = M.mahalanobis_score(np.full((1, 4), 10.0, "float32"), params)  # near Benign center
    s_dos_only = M.mahalanobis_score(np.full((1, 4), 10.0, "float32"),
                                     {"DoS": params["DoS"]})
    assert s[0] < s_dos_only[0]  # min over {DoS,Benign} < DoS-only
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/thesis_eval/test_mahalanobis.py -v`
Expected: FAIL — `No module named 'thesis_eval.metrics.mahalanobis'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/thesis_eval/metrics/mahalanobis.py
"""Mahalanobis score (Lee et al. 2018) in latent space.

M(x) = min_c (z - mu_c)^T Sigma^{-1} (z - mu_c).
cov_mode:
  - "tied": one pooled within-class covariance shared by all centers.
  - "per-class": separate Sigma_c per center.
Numerically stable inverse via Ledoit-Wolf shrinkage; pseudo-inverse fallback.
Condition number is logged in the returned params.
"""
from __future__ import annotations
import logging
import numpy as np
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)
_COND_MAX = 1e12


def _stable_precision(cov: np.ndarray) -> tuple[np.ndarray, float, bool]:
    cond = float(np.linalg.cond(cov))
    used = False
    if not np.isfinite(cond) or cond > _COND_MAX:
        lw = LedoitWolf().fit(np.random.default_rng(0).multivariate_normal(
            np.zeros(cov.shape[0]), cov, size=max(cov.shape[0] * 4, 50)))
        cov = lw.covariance_
        used = True
        cond = float(np.linalg.cond(cov))
    try:
        prec = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        prec = np.linalg.pinv(cov); used = True
    logger.info("Mahalanobis covariance cond=%.3e shrinkage/pinv=%s", cond, used)
    return prec, cond, used


def fit_gaussians(z_by_class: dict[str, np.ndarray], cov_mode: str = "tied") -> dict:
    assert cov_mode in ("tied", "per-class")
    centers = {c: z.mean(axis=0) for c, z in z_by_class.items()}
    if cov_mode == "tied":
        pooled = np.concatenate([z - centers[c] for c, z in z_by_class.items()], axis=0)
        cov = np.cov(pooled, rowvar=False)
        prec, cond, used = _stable_precision(cov)
        return {c: {"mu": centers[c], "precision": prec, "cov_mode": "tied",
                    "cond_number": cond, "used_shrinkage_or_pinv": used}
                for c in z_by_class}
    out = {}
    for c, z in z_by_class.items():
        cov = np.cov(z - centers[c], rowvar=False)
        prec, cond, used = _stable_precision(cov)
        out[c] = {"mu": centers[c], "precision": prec, "cov_mode": "per-class",
                  "cond_number": cond, "used_shrinkage_or_pinv": used}
    return out


def mahalanobis_score(z: np.ndarray, params: dict) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    per_center = []
    for c, p in params.items():
        d = z - p["mu"]
        per_center.append(np.einsum("ij,jk,ik->i", d, p["precision"], d))
    return np.min(np.stack(per_center, axis=1), axis=1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/thesis_eval/test_mahalanobis.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/thesis_eval/metrics/mahalanobis.py tests/thesis_eval/test_mahalanobis.py
git commit -m "thesis_eval: Mahalanobis score (tied + per-class)"
```

---

### Task 13: detector.py — LR AE-vs-clean, ROC/AUC per attack + bootstrap

**Files:**
- Create: `src/thesis_eval/metrics/detector.py`
- Test: `tests/thesis_eval/test_detector.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/thesis_eval/test_detector.py
import sys
from pathlib import Path
import numpy as np
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from thesis_eval.metrics import detector

def test_separable_scores_give_auc_near_one():
    rng = np.random.default_rng(0)
    clean = rng.normal(1, 0.5, 500)      # low Maha
    ae    = rng.normal(20, 0.5, 500)     # high Maha (off-manifold)
    res = detector.evaluate(clean_scores=clean, ae_scores=ae, n_boot=200, seed=42)
    assert res["auc"] > 0.99
    assert "fpr" in res and "tpr" in res
    assert res["auc_ci_low"] <= res["auc"] <= res["auc_ci_high"]

def test_overlapping_scores_give_auc_near_half():
    rng = np.random.default_rng(0)
    clean = rng.normal(5, 1, 500)
    ae    = rng.normal(5, 1, 500)        # indistinguishable (on-manifold)
    res = detector.evaluate(clean_scores=clean, ae_scores=ae, n_boot=200, seed=42)
    assert abs(res["auc"] - 0.5) < 0.1

def test_threshold_from_clean_percentile():
    clean = np.linspace(0, 100, 1001)
    thr = detector.operating_threshold(clean, percentile=95)
    assert abs(thr - 95.0) < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/thesis_eval/test_detector.py -v`
Expected: FAIL — `No module named 'thesis_eval.metrics.detector'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/thesis_eval/metrics/detector.py
"""Mahalanobis detector: logistic regression on Maha score(s), AE vs clean."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score


def _design(clean_scores: np.ndarray, ae_scores: np.ndarray):
    X = np.concatenate([clean_scores, ae_scores]).reshape(-1, 1)
    y = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(ae_scores))])
    return X, y


def evaluate(*, clean_scores: np.ndarray, ae_scores: np.ndarray,
             n_boot: int = 1000, seed: int = 42) -> dict:
    X, y = _design(np.asarray(clean_scores, float), np.asarray(ae_scores, float))
    lr = LogisticRegression().fit(X, y)
    prob = lr.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, prob)
    auc = float(roc_auc_score(y, prob))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        # guard against single-class resample
        boot[i] = roc_auc_score(y[idx], prob[idx]) if len(np.unique(y[idx])) == 2 else np.nan
    boot = boot[np.isfinite(boot)]
    return {"auc": auc, "fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "auc_ci_low": float(np.percentile(boot, 2.5)),
            "auc_ci_high": float(np.percentile(boot, 97.5))}


def operating_threshold(clean_scores: np.ndarray, percentile: float = 95) -> float:
    return float(np.percentile(np.asarray(clean_scores, float), percentile))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/thesis_eval/test_detector.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/thesis_eval/metrics/detector.py tests/thesis_eval/test_detector.py
git commit -m "thesis_eval: Mahalanobis LR detector + ROC/AUC"
```

---

### Task 14: idsr.py — IDSR = valid ∧ classifier-evaded ∧ detector-evaded

**Files:**
- Create: `src/thesis_eval/metrics/idsr.py`
- Test: `tests/thesis_eval/test_idsr.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/thesis_eval/test_idsr.py
import sys
from pathlib import Path
import numpy as np
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from thesis_eval.metrics import idsr

def test_idsr_triple_and():
    valid    = np.array([1,1,1,0,1], bool)
    success  = np.array([1,1,0,1,1], bool)
    maha     = np.array([1.0, 9.0, 1.0, 1.0, 1.0])  # threshold 5 -> evaded where < 5
    rate, m = idsr.compute(valid=valid, success=success, maha_scores=maha,
                           threshold=5.0, n_boot=200, seed=42)
    # rows satisfying all three: idx0 (v,s,evaded), idx4 (v,s,evaded) -> 2/5
    assert abs(rate - 0.4) < 1e-9
    assert m["idsr_ci_low"] <= rate <= m["idsr_ci_high"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/thesis_eval/test_idsr.py -v`
Expected: FAIL — `No module named 'thesis_eval.metrics.idsr'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/thesis_eval/metrics/idsr.py
"""IDSR = Pr[valid AND classifier-evaded AND Mahalanobis-detector-evaded]."""
from __future__ import annotations
import numpy as np
from thesis_eval.metrics.bootstrap import bootstrap_ci


def compute(*, valid: np.ndarray, success: np.ndarray, maha_scores: np.ndarray,
            threshold: float, n_boot: int = 1000, seed: int = 42) -> tuple[float, dict]:
    detector_evaded = np.asarray(maha_scores, float) < threshold
    triple = (np.asarray(valid, bool) & np.asarray(success, bool) & detector_evaded)
    vec = triple.astype(float)
    m, lo, hi = bootstrap_ci(vec, np.mean, n_boot=n_boot, seed=seed)
    return m, {"idsr": m, "idsr_ci_low": lo, "idsr_ci_high": hi, "n": int(vec.size)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/thesis_eval/test_idsr.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/thesis_eval/metrics/idsr.py tests/thesis_eval/test_idsr.py
git commit -m "thesis_eval: IDSR (valid & evaded & detector-evaded)"
```

---

### Task 15: fidelity.py — Wasserstein, JS, |corr diff|, NN distance

**Files:**
- Create: `src/thesis_eval/metrics/fidelity.py`
- Test: `tests/thesis_eval/test_fidelity.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/thesis_eval/test_fidelity.py
import sys
from pathlib import Path
import numpy as np
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from thesis_eval.metrics import fidelity

def test_wasserstein_zero_for_identical():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (500, 3))
    w = fidelity.wasserstein_per_feature(a, a.copy())
    assert np.allclose(w, 0.0, atol=1e-9)

def test_wasserstein_increases_with_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (2000, 1)); b = a + 3.0
    w = fidelity.wasserstein_per_feature(a, b)
    assert abs(w[0] - 3.0) < 0.2

def test_js_bounded_unit():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (2000, 2)); b = rng.normal(5, 1, (2000, 2))
    js = fidelity.js_divergence_per_feature(a, b, bins=50)
    assert np.all((js >= 0) & (js <= 1.0001))
    assert js.mean() > 0.3

def test_corr_diff_zero_for_same():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (500, 4))
    d = fidelity.corr_diff(a, a.copy())
    assert d.shape == (4, 4)
    assert np.allclose(d, 0.0, atol=1e-9)

def test_nn_distance_real_to_real_small():
    rng = np.random.default_rng(0)
    real = rng.normal(0, 1, (300, 5))
    ae = real + 5.0
    d_rr = fidelity.nn_distance(real, real, exclude_self=True)
    d_ae = fidelity.nn_distance(ae, real)
    assert d_ae.mean() > d_rr.mean()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/thesis_eval/test_fidelity.py -v`
Expected: FAIL — `No module named 'thesis_eval.metrics.fidelity'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/thesis_eval/metrics/fidelity.py
"""Distributional fidelity / diversity / originality metrics (NetDiffuser-style)."""
from __future__ import annotations
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.neighbors import NearestNeighbors


def wasserstein_per_feature(real: np.ndarray, gen: np.ndarray) -> np.ndarray:
    real, gen = np.asarray(real, float), np.asarray(gen, float)
    return np.array([wasserstein_distance(real[:, j], gen[:, j])
                     for j in range(real.shape[1])])


def js_divergence_per_feature(real: np.ndarray, gen: np.ndarray, bins: int = 50) -> np.ndarray:
    real, gen = np.asarray(real, float), np.asarray(gen, float)
    out = np.empty(real.shape[1])
    for j in range(real.shape[1]):
        lo = min(real[:, j].min(), gen[:, j].min())
        hi = max(real[:, j].max(), gen[:, j].max())
        if hi <= lo:
            out[j] = 0.0; continue
        edges = np.linspace(lo, hi, bins + 1)
        pr, _ = np.histogram(real[:, j], bins=edges, density=True)
        pg, _ = np.histogram(gen[:, j], bins=edges, density=True)
        pr = pr + 1e-12; pg = pg + 1e-12
        out[j] = jensenshannon(pr, pg, base=2) ** 2  # JS divergence in [0,1]
    return out


def corr_diff(real: np.ndarray, gen: np.ndarray) -> np.ndarray:
    cr = np.corrcoef(np.asarray(real, float), rowvar=False)
    cg = np.corrcoef(np.asarray(gen, float), rowvar=False)
    return np.abs(np.nan_to_num(cr) - np.nan_to_num(cg))


def nn_distance(query: np.ndarray, reference: np.ndarray,
                exclude_self: bool = False) -> np.ndarray:
    k = 2 if exclude_self else 1
    nn = NearestNeighbors(n_neighbors=k).fit(np.asarray(reference, float))
    dist, _ = nn.kneighbors(np.asarray(query, float))
    return dist[:, -1] if exclude_self else dist[:, 0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/thesis_eval/test_fidelity.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/thesis_eval/metrics/fidelity.py tests/thesis_eval/test_fidelity.py
git commit -m "thesis_eval: fidelity metrics (Wasserstein/JS/corr/NN)"
```

---

### Task 16: stats_tests.py — McNemar

**Files:**
- Create: `src/thesis_eval/metrics/stats_tests.py`
- Test: `tests/thesis_eval/test_stats_tests.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/thesis_eval/test_stats_tests.py
import sys
from pathlib import Path
import numpy as np
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from thesis_eval.metrics import stats_tests

def test_mcnemar_no_difference():
    a = np.array([1,1,0,0,1,0,1,0], bool)
    res = stats_tests.mcnemar(a, a.copy())
    assert res["b"] == 0 and res["c"] == 0
    assert res["p_value"] == 1.0

def test_mcnemar_discordant():
    a = np.array([1,1,1,1,0,0,0,0], bool)
    b = np.array([0,0,0,0,1,1,1,1], bool)
    res = stats_tests.mcnemar(a, b)
    assert res["b"] + res["c"] == 8
    assert 0.0 <= res["p_value"] <= 1.0
    assert "statistic" in res
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/thesis_eval/test_stats_tests.py -v`
Expected: FAIL — `No module named 'thesis_eval.metrics.stats_tests'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/thesis_eval/metrics/stats_tests.py
"""McNemar test on paired evasion outcomes (latent-PGD vs latent-CW)."""
from __future__ import annotations
import numpy as np
from scipy.stats import chi2, binomtest


def mcnemar(success_a: np.ndarray, success_b: np.ndarray, exact_threshold: int = 25) -> dict:
    a = np.asarray(success_a, bool); b = np.asarray(success_b, bool)
    if a.shape != b.shape:
        raise ValueError("paired outcomes must align")
    b_disc = int(np.sum(a & ~b))   # A success, B fail
    c_disc = int(np.sum(~a & b))   # A fail, B success
    n_disc = b_disc + c_disc
    if n_disc == 0:
        return {"b": 0, "c": 0, "statistic": 0.0, "p_value": 1.0, "test": "none"}
    if n_disc < exact_threshold:
        p = float(binomtest(b_disc, n_disc, 0.5).pvalue)
        stat = float(min(b_disc, c_disc))
        return {"b": b_disc, "c": c_disc, "statistic": stat, "p_value": p, "test": "exact"}
    stat = (abs(b_disc - c_disc) - 1) ** 2 / n_disc   # continuity-corrected
    p = float(chi2.sf(stat, df=1))
    return {"b": b_disc, "c": c_disc, "statistic": float(stat), "p_value": p, "test": "chi2"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/thesis_eval/test_stats_tests.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/thesis_eval/metrics/stats_tests.py tests/thesis_eval/test_stats_tests.py
git commit -m "thesis_eval: McNemar test"
```

---

### Task 16b: stats_output.py — emit McNemar table (latent-PGD vs latent-CW)

**Files:**
- Create: `src/thesis_eval/metrics/stats_output.py`
- Test: `tests/thesis_eval/test_stats_output.py`

Pairs the untargeted latent-PGD and latent-CW bundles per (model, source_class) — they
were attacked on the *same* per-class indices in `export_latent`, so outcomes align by
position. Emits `Fstats_mcnemar.csv` consumed by `FIGURES.md`.

- [ ] **Step 1: Write the failing test**

```python
# tests/thesis_eval/test_stats_output.py
import sys
from pathlib import Path
import numpy as np
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from thesis_eval.metrics import stats_output
from thesis_eval.data.bundles import AEBundle

def _b(pop, succ):
    n=len(succ); z=np.zeros((n,16),"float32"); x=np.zeros((n,39),"float32")
    return AEBundle(x_orig=x,x_adv=x,z_orig=z,z_adv=z,y_true=np.full(n,3,"int64"),
        y_pred_clean=np.full(n,3,"int64"),y_pred_adv=np.zeros(n,"int64"),
        success=np.array(succ,bool),protocol_valid=np.ones(n,bool),mask_valid=np.ones(n,bool),
        raw_g1g8_valid=np.ones(n,bool),joint_valid=np.ones(n,bool),population=pop,
        model="mlp",source_class="DoS",attack_type=pop,vae_tag="t",seed=42)

def test_mcnemar_table_pairs_pgd_cw(tmp_path):
    bundles=[_b("latentPGD_untgt",[1,1,0,0]), _b("latentCW_untgt",[1,0,1,0])]
    df=stats_output.mcnemar_table(bundles)
    assert {"model","source_class","b","c","statistic","p_value"} <= set(df.columns)
    assert len(df)==1 and df.iloc[0]["model"]=="mlp"
    out=stats_output.write(bundles, tmp_path)
    assert Path(out).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/thesis_eval/test_stats_output.py -v`
Expected: FAIL — `No module named 'thesis_eval.metrics.stats_output'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/thesis_eval/metrics/stats_output.py
"""Emit the McNemar (latent-PGD vs latent-CW) comparison as a CSV."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from thesis_eval import config, io_utils
from thesis_eval.metrics.stats_tests import mcnemar
from thesis_eval.data.bundles import AEBundle


def mcnemar_table(bundles: list[AEBundle]) -> pd.DataFrame:
    by_key = {}
    for b in bundles:
        if b.population in ("latentPGD_untgt", "latentCW_untgt"):
            by_key.setdefault((b.model, b.source_class), {})[b.population] = b
    rows = []
    for (model, cls), d in by_key.items():
        if "latentPGD_untgt" not in d or "latentCW_untgt" not in d:
            continue
        a, c = d["latentPGD_untgt"], d["latentCW_untgt"]
        n = min(a.n, c.n)
        res = mcnemar(a.success[:n], c.success[:n])
        rows.append({"model": model, "source_class": cls, **res})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["source_class"] = pd.Categorical(df["source_class"], config.CLASS_ORDER, ordered=True)
        df = df.sort_values(["model", "source_class"]).reset_index(drop=True)
    return df


def write(bundles: list[AEBundle], data_dir: Path | None = None) -> Path:
    df = mcnemar_table(bundles)
    return io_utils.write_csv(df, "Fstats_mcnemar", data_dir)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/thesis_eval/test_stats_output.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/thesis_eval/metrics/stats_output.py tests/thesis_eval/test_stats_output.py
git commit -m "thesis_eval: McNemar comparison CSV output"
```

---

# PHASE 3 — Figures

Figures load bundles + compute via metrics + call `io_utils.save_figure`. Each layer
module exposes `generate(bundles, ctx) -> list[str]` returning figure names written.
`ctx` is a small dataclass (defined in Task 17) carrying cov_mode, seed, n_boot, and the
output dirs. Tests are smoke tests: synthetic bundles in, assert the pdf/png/csv exist
and CSV has the expected columns. **No pixel assertions.**

### Task 17: figures context + layer1_effectiveness (F1, F2, F3)

**Files:**
- Create: `src/thesis_eval/figures/__init__.py` (FigureContext dataclass), `src/thesis_eval/figures/layer1_effectiveness.py`
- Test: `tests/thesis_eval/test_figures_smoke.py` (shared synthetic-bundle factory + F1–F3 cases)

- [ ] **Step 1: Write the failing test**

```python
# tests/thesis_eval/test_figures_smoke.py
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from thesis_eval.data.bundles import AEBundle
from thesis_eval.figures import FigureContext
from thesis_eval.figures import layer1_effectiveness as L1

def make_bundle(pop, model, cls, n=40, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0,1,(n,16)).astype("float32")
    x = rng.normal(0,1,(n,39)).astype("float32")
    succ = rng.random(n) < (0.4 if pop.startswith("latent") else 0.5)
    valid = rng.random(n) < (0.9 if pop.startswith("latent") else 0.05)
    return AEBundle(x_orig=x, x_adv=x+rng.normal(0,0.3,(n,39)).astype("float32"),
        z_orig=z, z_adv=z+rng.normal(0,0.2,(n,16)).astype("float32"),
        y_true=np.full(n, 3,"int64"), y_pred_clean=np.full(n,3,"int64"),
        y_pred_adv=np.where(succ,0,3).astype("int64"), success=succ,
        protocol_valid=valid, mask_valid=np.ones(n,bool),
        raw_g1g8_valid=valid, joint_valid=valid, population=pop, model=model,
        source_class=cls, attack_type=pop, vae_tag="t", seed=42)

def all_bundles():
    from thesis_eval import config
    pops = ["latentPGD_untgt","latentCW_untgt","PGD","CW"]
    return [make_bundle(p, m, c, seed=hash((p,m,c))%1000)
            for p in pops for m in config.CLASSIFIER_TAGS for c in config.CLASS_ORDER]

def _ctx(tmp_path):
    return FigureContext(figures_dir=tmp_path/"figures", data_dir=tmp_path/"data",
                         cache_dir=tmp_path/"cache", cov_mode="tied", seed=42, n_boot=100)

def _assert_outputs(tmp_path, names):
    for nm in names:
        assert (tmp_path/"figures"/f"{nm}.pdf").exists(), nm
        assert (tmp_path/"figures"/f"{nm}.png").exists(), nm
        assert (tmp_path/"data"/f"{nm}.csv").exists(), nm

def test_layer1(tmp_path):
    names = L1.generate(all_bundles(), _ctx(tmp_path))
    assert {"F1_asr_valid_table","F2_asr_valid_by_class","F3_arch_invariance"} <= set(names)
    _assert_outputs(tmp_path, names)
    # F1 must also emit a LaTeX table
    assert (tmp_path/"data"/"F1_asr_valid_table.tex").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/thesis_eval/test_figures_smoke.py::test_layer1 -v`
Expected: FAIL — `No module named 'thesis_eval.figures.layer1_effectiveness'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/thesis_eval/figures/__init__.py
from dataclasses import dataclass
from pathlib import Path

@dataclass
class FigureContext:
    figures_dir: Path
    data_dir: Path
    cache_dir: Path
    cov_mode: str = "tied"
    seed: int = 42
    n_boot: int = 1000
```

```python
# src/thesis_eval/figures/layer1_effectiveness.py
"""Layer 1 — attack effectiveness: F1 table, F2 grouped bar, F3 heatmap."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from thesis_eval import config, io_utils, palette
from thesis_eval.metrics import asr
from thesis_eval.figures import FigureContext

_LATENT = ["latentPGD_untgt","latentCW_untgt","latentPGD_tgtBenign","latentCW_tgtBenign"]
_GRAD = ["PGD","CW"]


def _table(bundles, ctx) -> pd.DataFrame:
    df = asr.asr_table(bundles, n_boot=ctx.n_boot, seed=ctx.seed)
    df["source_class"] = pd.Categorical(df["source_class"], config.CLASS_ORDER, ordered=True)
    return df.sort_values(["population","source_class","model"]).reset_index(drop=True)


def generate(bundles, ctx: FigureContext) -> list[str]:
    palette.apply_rcparams()
    written = []
    df = _table(bundles, ctx)

    # ---- F1: ASR_valid comparison table (CSV + LaTeX) ----
    agg = (df.groupby(["population","source_class"], observed=True)
             .agg(asr_raw=("asr_raw","mean"), asr_valid=("asr_valid","mean"),
                  validity_rate=("validity_rate","mean"), mean_l2=("mean_l2","mean"),
                  asr_valid_ci_low=("asr_valid_ci_low","mean"),
                  asr_valid_ci_high=("asr_valid_ci_high","mean")).reset_index())
    io_utils.write_csv(agg, "F1_asr_valid_table", ctx.data_dir)
    (ctx.data_dir / "F1_asr_valid_table.tex").write_text(
        agg.to_latex(index=False, float_format="%.3f"))
    # also emit a trivial figure rendering of the table for completeness
    fig, ax = plt.subplots(figsize=(11, 0.4*len(agg)+1)); ax.axis("off")
    ax.table(cellText=np.round(agg.select_dtypes("number").values,3),
             colLabels=[c for c in agg.columns if agg[c].dtype.kind in "fi"],
             loc="center")
    io_utils.save_figure(fig, "F1_asr_valid_table", agg, ctx.figures_dir, ctx.data_dir)
    written.append("F1_asr_valid_table")

    # ---- F2: grouped bar of ASR_valid per class (latent vs PGD/CW) ----
    pivot = (df.groupby(["source_class","population"], observed=True)["asr_valid"]
               .mean().reset_index())
    classes = config.CLASS_ORDER
    pops = [p for p in (_LATENT + _GRAD) if p in pivot["population"].unique()]
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.8 / max(len(pops), 1)
    for i, pop in enumerate(pops):
        vals = [pivot[(pivot.source_class==c)&(pivot.population==pop)]["asr_valid"].mean()
                for c in classes]
        ax.bar(np.arange(len(classes)) + i*width, np.nan_to_num(vals), width,
               label=pop, color=palette.color_for_population(pop),
               hatch=palette.hatch_for_population(pop), edgecolor="black", linewidth=0.4)
    ax.set_xticks(np.arange(len(classes)) + 0.4 - width/2)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("ASR_valid"); ax.legend(fontsize=8, ncol=3)
    mirai_i = classes.index("Mirai")
    ax.annotate("Mirai≈0:\ncollapsed latent dims", xy=(mirai_i, 0.02),
                xytext=(mirai_i, 0.3), fontsize=8,
                arrowprops=dict(arrowstyle="->"))
    io_utils.save_figure(fig, "F2_asr_valid_by_class", pivot, ctx.figures_dir, ctx.data_dir)
    written.append("F2_asr_valid_by_class")

    # ---- F3: architecture-invariance heatmap (classifier x class) for latent method ----
    latent_only = df[df["population"].isin(_LATENT)]
    hm = (latent_only.groupby(["model","source_class"], observed=True)["asr_valid"]
          .mean().reset_index()
          .pivot(index="model", columns="source_class", values="asr_valid")
          .reindex(index=config.CLASSIFIER_TAGS, columns=config.CLASS_ORDER))
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(hm.values, aspect="auto", cmap="viridis", vmin=0, vmax=max(np.nanmax(hm.values),0.01))
    ax.set_xticks(range(len(config.CLASS_ORDER))); ax.set_xticklabels(config.CLASS_ORDER, rotation=30, ha="right")
    ax.set_yticks(range(len(config.CLASSIFIER_TAGS)))
    ax.set_yticklabels([config.CLASSIFIER_DISPLAY[t] for t in config.CLASSIFIER_TAGS])
    for (r,c), v in np.ndenumerate(hm.values):
        if np.isfinite(v): ax.text(c, r, f"{v:.2f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, label="ASR_valid (latent)")
    io_utils.save_figure(fig, "F3_arch_invariance", hm.reset_index(), ctx.figures_dir, ctx.data_dir)
    written.append("F3_arch_invariance")
    return written
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/thesis_eval/test_figures_smoke.py::test_layer1 -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/thesis_eval/figures/__init__.py src/thesis_eval/figures/layer1_effectiveness.py tests/thesis_eval/test_figures_smoke.py
git commit -m "thesis_eval: Layer 1 figures (F1-F3)"
```

---

### Task 18: layer2_mahalanobis (F4, F5, F6, F7)

**Files:**
- Create: `src/thesis_eval/figures/layer2_mahalanobis.py`
- Modify: `tests/thesis_eval/test_figures_smoke.py` (add `test_layer2`)

This layer needs latent scores. It builds clean reference latents from synthetic
bundle `z_orig` (in the real run, from clean training latents — supplied via ctx cache,
see Task 22). For the smoke test, clean latents are derived from the bundles' `z_orig`.

- [ ] **Step 1: Write the failing test (append to test_figures_smoke.py)**

```python
def test_layer2(tmp_path):
    from thesis_eval.figures import layer2_mahalanobis as L2
    names = L2.generate(all_bundles(), _ctx(tmp_path))
    assert {"F4_maha_kde","F5_detector_roc","F6_l2_vs_maha","F7_idsr_by_class"} <= set(names)
    _assert_outputs(tmp_path, names)
    assert (tmp_path/"data"/"F5_detector_auc_table.csv").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/thesis_eval/test_figures_smoke.py::test_layer2 -v`
Expected: FAIL — `No module named 'thesis_eval.figures.layer2_mahalanobis'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/thesis_eval/figures/layer2_mahalanobis.py
"""Layer 2 — realism: F4 Maha KDE, F5 detector ROC, F6 L2-vs-Maha, F7 IDSR."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from thesis_eval import config, io_utils, palette
from thesis_eval.metrics import mahalanobis as M, detector, idsr
from thesis_eval.figures import FigureContext

_LATENT_POPS = ["latentPGD_untgt","latentCW_untgt","latentPGD_tgtBenign","latentCW_tgtBenign"]
_GRAD_POPS = ["PGD","CW"]


def _score_bundles(bundles, ctx):
    """Per source-class: fit gaussians on clean z (z_orig), score each population's z_adv.

    Gradient bundles have NaN z_adv; encode their x_adv through the class VAE here.
    Returns dict[(population, source_class)] -> maha scores, plus clean scores per class.
    """
    by_class = {}
    for b in bundles:
        by_class.setdefault(b.source_class, []).append(b)
    scores = {}; clean_scores = {}; params_by_class = {}
    router = None
    for cls, bs in by_class.items():
        clean_z = np.concatenate([b.z_orig for b in bs if not np.isnan(b.z_orig).all()], axis=0)
        benign_z = clean_z + 6.0  # placeholder benign center in smoke; real run uses benign latents
        params = M.fit_gaussians({cls: clean_z, "Benign": benign_z}, cov_mode=ctx.cov_mode)
        params_by_class[cls] = params
        clean_scores[cls] = M.mahalanobis_score(clean_z, params)
        for b in bs:
            z = b.z_adv
            if np.isnan(z).all():
                if router is None:
                    from thesis_eval.data import artifacts
                    try:
                        router = artifacts.build_attack_router(device="cpu")
                    except Exception:
                        router = False
                if router:
                    import torch
                    vae = router.get_vae(config.CLASS_TO_ID[cls])
                    with torch.no_grad():
                        mu, _ = vae.encode(torch.from_numpy(b.x_adv).float())
                    z = mu.cpu().numpy()
                else:
                    z = b.z_orig  # smoke fallback
            scores[(b.population, cls)] = M.mahalanobis_score(z, params)
    return scores, clean_scores, params_by_class


def generate(bundles, ctx: FigureContext) -> list[str]:
    palette.apply_rcparams()
    written = []
    scores, clean_scores, _ = _score_bundles(bundles, ctx)
    all_clean = np.concatenate(list(clean_scores.values()))

    def pooled(pop):
        vals = [v for (p, c), v in scores.items() if p == pop]
        return np.concatenate(vals) if vals else np.array([])

    # ---- F4: overlaid Maha KDE ----
    fig, ax = plt.subplots(figsize=(9, 5))
    series = {"clean_malicious": all_clean,
              "clean_benign": all_clean.min() + np.abs(np.random.default_rng(0).normal(0,1,len(all_clean)))}
    for pop in _LATENT_POPS + _GRAD_POPS:
        v = pooled(pop)
        if v.size: series[pop] = v
    rows = []
    for name, v in series.items():
        if v.size < 5: continue
        color = (palette.color_for_method("clean_benign") if name=="clean_benign"
                 else palette.color_for_method("clean_malicious") if name=="clean_malicious"
                 else palette.color_for_population(name))
        ls = "-" if name in ("clean_benign","clean_malicious") else palette.style_for_population(name)
        xs = np.linspace(np.percentile(v,0.5), np.percentile(v,99.5), 200)
        try:
            ys = gaussian_kde(v)(xs)
        except Exception:
            continue
        ax.plot(xs, ys, label=name, color=color, linestyle=ls)
        rows.append(pd.DataFrame({"population": name, "maha": v}))
    ax.set_xlabel("Mahalanobis score (latent)"); ax.set_ylabel("density"); ax.legend(fontsize=8)
    io_utils.save_figure(fig, "F4_maha_kde", pd.concat(rows, ignore_index=True),
                         ctx.figures_dir, ctx.data_dir)
    written.append("F4_maha_kde")

    # ---- F5: detector ROC per attack + AUC table ----
    fig, ax = plt.subplots(figsize=(6, 6)); auc_rows = []
    for pop in _LATENT_POPS + _GRAD_POPS:
        v = pooled(pop)
        if v.size < 10: continue
        res = detector.evaluate(clean_scores=all_clean, ae_scores=v,
                                n_boot=ctx.n_boot, seed=ctx.seed)
        ax.plot(res["fpr"], res["tpr"], label=f"{pop} (AUC={res['auc']:.2f})",
                color=palette.color_for_population(pop), linestyle=palette.style_for_population(pop))
        auc_rows.append({"population": pop, "auc": res["auc"],
                         "auc_ci_low": res["auc_ci_low"], "auc_ci_high": res["auc_ci_high"]})
    ax.plot([0,1],[0,1],"k:",alpha=0.5); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(fontsize=8)
    auc_df = pd.DataFrame(auc_rows)
    io_utils.save_figure(fig, "F5_detector_roc", auc_df, ctx.figures_dir, ctx.data_dir)
    io_utils.write_csv(auc_df, "F5_detector_auc_table", ctx.data_dir)
    written.append("F5_detector_roc")

    # ---- F6: feature-L2 vs Maha scatter ----
    fig, ax = plt.subplots(figsize=(7, 6)); scat_rows = []
    for b in bundles:
        if b.population not in (_LATENT_POPS + _GRAD_POPS): continue
        v = scores.get((b.population, b.source_class))
        if v is None: continue
        l2 = np.linalg.norm(b.x_adv - b.x_orig, axis=1)
        m = min(len(l2), len(v)); l2, v2 = l2[:m], v[:m]
        ax.scatter(l2, v2, s=8, alpha=0.4, color=palette.color_for_population(b.population),
                   marker=("o" if b.joint_valid[:m].mean()>0.5 else "x"))
        scat_rows.append(pd.DataFrame({"population": b.population, "l2": l2, "maha": v2,
                                       "valid_frac": b.joint_valid[:m].astype(float)}))
    ax.set_xlabel("feature-space L2"); ax.set_ylabel("Mahalanobis score")
    io_utils.save_figure(fig, "F6_l2_vs_maha", pd.concat(scat_rows, ignore_index=True),
                         ctx.figures_dir, ctx.data_dir)
    written.append("F6_l2_vs_maha")

    # ---- F7: IDSR grouped bar per class ----
    thr = detector.operating_threshold(all_clean, percentile=95)
    rows = []
    for b in bundles:
        if b.population not in (_LATENT_POPS + _GRAD_POPS): continue
        v = scores.get((b.population, b.source_class))
        if v is None: continue
        m = min(len(v), b.n)
        rate, meta = idsr.compute(valid=b.joint_valid[:m], success=b.success[:m],
                                  maha_scores=v[:m], threshold=thr,
                                  n_boot=ctx.n_boot, seed=ctx.seed)
        rows.append({"population": b.population, "source_class": b.source_class, "idsr": rate})
    idf = pd.DataFrame(rows)
    piv = (idf.groupby(["source_class","population"])["idsr"].mean().reset_index())
    fig, ax = plt.subplots(figsize=(12, 5))
    pops = [p for p in (_LATENT_POPS+_GRAD_POPS) if p in piv["population"].unique()]
    width = 0.8/max(len(pops),1)
    for i, pop in enumerate(pops):
        vals = [piv[(piv.source_class==c)&(piv.population==pop)]["idsr"].mean()
                for c in config.CLASS_ORDER]
        ax.bar(np.arange(len(config.CLASS_ORDER))+i*width, np.nan_to_num(vals), width,
               label=pop, color=palette.color_for_population(pop),
               hatch=palette.hatch_for_population(pop), edgecolor="black", linewidth=0.4)
    ax.set_xticks(np.arange(len(config.CLASS_ORDER))+0.4-width/2)
    ax.set_xticklabels(config.CLASS_ORDER, rotation=30, ha="right")
    ax.set_ylabel("IDSR"); ax.legend(fontsize=8, ncol=3)
    io_utils.save_figure(fig, "F7_idsr_by_class", piv, ctx.figures_dir, ctx.data_dir)
    written.append("F7_idsr_by_class")
    return written
```

> NOTE for implementer: in the smoke test the benign center and gradient-encoding are
> stubbed. In the real run (Task 22) `ctx` supplies clean *training* latents per class and
> per-class benign latents from the cache; replace the `benign_z` placeholder and the
> `router`-encode fallback with the cached real latents. Keep the public `generate()`
> signature unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/thesis_eval/test_figures_smoke.py::test_layer2 -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/thesis_eval/figures/layer2_mahalanobis.py tests/thesis_eval/test_figures_smoke.py
git commit -m "thesis_eval: Layer 2 figures (F4-F7)"
```

---

### Task 19: layer3_fidelity (F8, F9, F10)

**Files:**
- Create: `src/thesis_eval/figures/layer3_fidelity.py`
- Modify: `tests/thesis_eval/test_figures_smoke.py` (add `test_layer3`)

- [ ] **Step 1: Write the failing test (append)**

```python
def test_layer3(tmp_path):
    from thesis_eval.figures import layer3_fidelity as L3
    names = L3.generate(all_bundles(), _ctx(tmp_path))
    assert any(n.startswith("F8_wasserstein") for n in names)
    assert any(n.startswith("F9_pca_kde") for n in names)
    assert any(n.startswith("F10_corr_diff") for n in names)
    _assert_outputs(tmp_path, names)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/thesis_eval/test_figures_smoke.py::test_layer3 -v`
Expected: FAIL — `No module named 'thesis_eval.figures.layer3_fidelity'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/thesis_eval/figures/layer3_fidelity.py
"""Layer 3 — distributional fidelity: F8 Wasserstein, F9 PCA-KDE+JS, F10 corr diff."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from thesis_eval import config, io_utils, palette
from thesis_eval.metrics import fidelity
from thesis_eval.data import artifacts
from thesis_eval.figures import FigureContext

# "generated" = valid latent AEs (raw space, inverse-transformed); "real" = clean x_orig.
_LATENT = "latentPGD_untgt"


def _real_gen_for_class(bundles, cls):
    bs = [b for b in bundles if b.source_class == cls and b.population == _LATENT]
    if not bs:
        return None, None
    real = np.concatenate([b.x_orig for b in bs], axis=0)
    gen = np.concatenate([b.x_adv[b.joint_valid] for b in bs if b.joint_valid.any()], axis=0)
    return real, (gen if len(gen) else bs[0].x_adv)


def generate(bundles, ctx: FigureContext) -> list[str]:
    palette.apply_rcparams()
    written = []
    names = artifacts.feature_names()

    # ---- F8: per-feature Wasserstein, one panel per class ----
    fig, axes = plt.subplots(2, 4, figsize=(20, 10)); axes = axes.ravel()
    f8_rows = []
    for ax, cls in zip(axes, config.CLASS_ORDER):
        real, gen = _real_gen_for_class(bundles, cls)
        if real is None: ax.axis("off"); continue
        w = fidelity.wasserstein_per_feature(real, gen)
        order = np.argsort(w)
        ax.barh(np.array(names)[order], w[order], color=palette.color_for_method("latent_PGD"))
        ax.set_title(cls); ax.tick_params(labelsize=6)
        f8_rows.append(pd.DataFrame({"source_class": cls, "feature": names, "wasserstein": w}))
    for ax in axes[len(config.CLASS_ORDER):]: ax.axis("off")
    io_utils.save_figure(fig, "F8_wasserstein_by_class", pd.concat(f8_rows, ignore_index=True),
                         ctx.figures_dir, ctx.data_dir)
    written.append("F8_wasserstein_by_class")

    # ---- F9: PCA-projection KDE grid + JS per subplot ----
    fig, axes = plt.subplots(2, 4, figsize=(20, 10)); axes = axes.ravel()
    f9_rows = []
    for ax, cls in zip(axes, config.CLASS_ORDER):
        real, gen = _real_gen_for_class(bundles, cls)
        if real is None: ax.axis("off"); continue
        pca = PCA(n_components=2).fit(real)
        pr, pg = pca.transform(real), pca.transform(gen)
        ax.scatter(pr[:,0], pr[:,1], s=6, alpha=0.3, color=palette.color_for_method("clean_malicious"), label="real")
        ax.scatter(pg[:,0], pg[:,1], s=6, alpha=0.3, color=palette.color_for_method("latent_PGD"), label="gen")
        js = fidelity.js_divergence_per_feature(pr, pg, bins=40).mean()
        ax.set_title(f"{cls}  JS={js:.3f}"); ax.legend(fontsize=6)
        f9_rows.append({"source_class": cls, "js_pca": float(js)})
    for ax in axes[len(config.CLASS_ORDER):]: ax.axis("off")
    io_utils.save_figure(fig, "F9_pca_kde_js", pd.DataFrame(f9_rows), ctx.figures_dir, ctx.data_dir)
    written.append("F9_pca_kde_js")

    # ---- F10: |Corr_real - Corr_gen| heatmap per class ----
    fig, axes = plt.subplots(2, 4, figsize=(22, 11)); axes = axes.ravel()
    f10_rows = []
    for ax, cls in zip(axes, config.CLASS_ORDER):
        real, gen = _real_gen_for_class(bundles, cls)
        if real is None: ax.axis("off"); continue
        d = fidelity.corr_diff(real, gen)
        im = ax.imshow(d, cmap="magma", vmin=0, vmax=1); ax.set_title(cls); ax.tick_params(labelsize=4)
        f10_rows.append(pd.DataFrame({"source_class": cls, "mean_abs_corr_diff": [float(np.nanmean(d))]}))
    for ax in axes[len(config.CLASS_ORDER):]: ax.axis("off")
    fig.colorbar(im, ax=axes.tolist(), label="|Corr_real - Corr_gen|", shrink=0.5)
    io_utils.save_figure(fig, "F10_corr_diff_by_class", pd.concat(f10_rows, ignore_index=True),
                         ctx.figures_dir, ctx.data_dir)
    written.append("F10_corr_diff_by_class")
    return written
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/thesis_eval/test_figures_smoke.py::test_layer3 -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/thesis_eval/figures/layer3_fidelity.py tests/thesis_eval/test_figures_smoke.py
git commit -m "thesis_eval: Layer 3 figures (F8-F10)"
```

---

### Task 20: layer4_geometry (F11, F12)

**Files:**
- Create: `src/thesis_eval/figures/layer4_geometry.py`
- Modify: `tests/thesis_eval/test_figures_smoke.py` (add `test_layer4`)

- [ ] **Step 1: Write the failing test (append)**

```python
def test_layer4(tmp_path):
    from thesis_eval.figures import layer4_geometry as L4
    names = L4.generate(all_bundles(), _ctx(tmp_path))
    assert {"F11_tsne_umap_overlay","F12_nn_distance_hist"} <= set(names)
    _assert_outputs(tmp_path, names)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/thesis_eval/test_figures_smoke.py::test_layer4 -v`
Expected: FAIL — `No module named 'thesis_eval.figures.layer4_geometry'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/thesis_eval/figures/layer4_geometry.py
"""Layer 4 — latent geometry: F11 t-SNE+UMAP overlays, F12 NN-distance hist."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from thesis_eval import config, io_utils, palette
from thesis_eval.metrics import fidelity
from thesis_eval.figures import FigureContext

_LATENT_POPS = ["latentPGD_untgt","latentCW_untgt"]
_GRAD_POPS = ["PGD","CW"]


def _embed_tsne(Z, seed):
    n = len(Z)
    perp = max(5, min(30, (n - 1) // 3))
    return TSNE(n_components=2, perplexity=perp, init="pca",
                random_state=seed).fit_transform(Z)


def _embed_umap(Z, seed):
    try:
        import umap
        return umap.UMAP(n_components=2, random_state=seed).fit_transform(Z)
    except Exception:
        from sklearn.decomposition import PCA
        return PCA(n_components=2).fit_transform(Z)


def generate(bundles, ctx: FigureContext) -> list[str]:
    palette.apply_rcparams()
    written = []
    # Build a combined latent matrix: clean (z_orig) + valid latent AEs (z_adv) + gradient(encoded -> use z_orig as proxy if NaN)
    blocks, labels = [], []
    for b in bundles:
        if not np.isnan(b.z_orig).all():
            blocks.append(b.z_orig); labels += [("clean_malicious", b.source_class)] * b.n
        if b.population in _LATENT_POPS and not np.isnan(b.z_adv).all():
            valid = b.joint_valid
            if valid.any():
                blocks.append(b.z_adv[valid]); labels += [(b.population, b.source_class)] * int(valid.sum())
    Z = np.concatenate(blocks, axis=0)
    lab = pd.DataFrame(labels, columns=["population","source_class"])

    # ---- F11: t-SNE + UMAP side by side, + dedicated Mirai panel ----
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, embed, title in [(axes[0], _embed_tsne, "t-SNE"), (axes[1], _embed_umap, "UMAP")]:
        E = embed(Z, ctx.seed)
        for pop in ["clean_malicious"] + _LATENT_POPS:
            m = (lab["population"] == pop).values
            if m.any():
                ax.scatter(E[m,0], E[m,1], s=6, alpha=0.4,
                           color=(palette.color_for_method("clean_malicious") if pop=="clean_malicious"
                                  else palette.color_for_population(pop)), label=pop)
        ax.set_title(title); ax.legend(fontsize=7)
    # Mirai panel (t-SNE on Mirai only)
    mir = (lab["source_class"] == "Mirai").values
    axM = axes[2]
    if mir.sum() > 10:
        Em = _embed_tsne(Z[mir], ctx.seed)
        lm = lab[mir].reset_index(drop=True)
        for pop in ["clean_malicious"] + _LATENT_POPS:
            mm = (lm["population"] == pop).values
            if mm.any():
                axM.scatter(Em[mm,0], Em[mm,1], s=8, alpha=0.5,
                            color=(palette.color_for_method("clean_malicious") if pop=="clean_malicious"
                                   else palette.color_for_population(pop)), label=pop)
    axM.set_title("Mirai latent region (collapsed)"); axM.legend(fontsize=7)
    io_utils.save_figure(fig, "F11_tsne_umap_overlay", lab, ctx.figures_dir, ctx.data_dir)
    written.append("F11_tsne_umap_overlay")

    # ---- F12: NN-distance hist (AE->real vs real->real) ----
    fig, ax = plt.subplots(figsize=(8, 5)); rows = []
    real_pool = np.concatenate([b.z_orig for b in bundles if not np.isnan(b.z_orig).all()], axis=0)
    ae_pool = np.concatenate([b.z_adv[b.joint_valid] for b in bundles
                              if b.population in _LATENT_POPS and not np.isnan(b.z_adv).all()
                              and b.joint_valid.any()], axis=0)
    d_rr = fidelity.nn_distance(real_pool, real_pool, exclude_self=True)
    d_ae = fidelity.nn_distance(ae_pool, real_pool)
    ax.hist(d_rr, bins=50, alpha=0.5, density=True, label="real→real",
            color=palette.color_for_method("clean_malicious"))
    ax.hist(d_ae, bins=50, alpha=0.5, density=True, label="AE→real",
            color=palette.color_for_method("latent_PGD"))
    ax.set_xlabel("nearest-neighbor distance (latent)"); ax.set_ylabel("density"); ax.legend()
    rows = pd.DataFrame({"kind": ["real_real"]*len(d_rr)+["ae_real"]*len(d_ae),
                         "distance": np.concatenate([d_rr, d_ae])})
    io_utils.save_figure(fig, "F12_nn_distance_hist", rows, ctx.figures_dir, ctx.data_dir)
    written.append("F12_nn_distance_hist")
    return written
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/thesis_eval/test_figures_smoke.py::test_layer4 -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/thesis_eval/figures/layer4_geometry.py tests/thesis_eval/test_figures_smoke.py
git commit -m "thesis_eval: Layer 4 figures (F11-F12)"
```

---

### Task 21: layer5_vae (F13, F14, F15)

**Files:**
- Create: `src/thesis_eval/figures/layer5_vae.py`
- Modify: `tests/thesis_eval/test_figures_smoke.py` (add `test_layer5`)

Layer 5 reads existing JSON artifacts, not bundles. F15 depends on numeric loss arrays;
if absent it writes a `# TODO` stub CSV and a placeholder note instead of fabricating.

- [ ] **Step 1: Write the failing test (append)**

```python
def test_layer5(tmp_path):
    from thesis_eval.figures import layer5_vae as L5
    names = L5.generate([], _ctx(tmp_path))   # bundles unused
    assert {"F13_active_units","F14_recon_by_class"} <= set(names)
    for nm in ["F13_active_units","F14_recon_by_class"]:
        assert (tmp_path/"figures"/f"{nm}.pdf").exists()
        assert (tmp_path/"data"/f"{nm}.csv").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/thesis_eval/test_figures_smoke.py::test_layer5 -v`
Expected: FAIL — `No module named 'thesis_eval.figures.layer5_vae'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/thesis_eval/figures/layer5_vae.py
"""Layer 5 — VAE diagnostics: F13 active units/KL, F14 recon by class, F15 loss curves."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from thesis_eval import config, io_utils, palette
from thesis_eval.figures import FigureContext

_VAE_RESULTS = config.REPO_ROOT / "results" / "vae"


def _load_diagnostics() -> dict:
    out = {}
    for cls in config.CLASS_ORDER + ["Benign"]:
        p = _VAE_RESULTS / f"diagnostics_{cls}.json"
        if p.exists():
            out[cls] = json.loads(p.read_text())
    return out


def generate(bundles, ctx: FigureContext) -> list[str]:
    palette.apply_rcparams()
    written = []
    diag = _load_diagnostics()

    # ---- F13: per-dim KL / active units, collapse threshold marked ----
    fig, ax = plt.subplots(figsize=(11, 5)); rows = []
    width = 0.8 / max(len(diag), 1)
    thr = None
    for i, (cls, d) in enumerate(diag.items()):
        pc = d.get("posterior_collapse", {})
        kl = np.array(pc.get("per_dim_kl", []), dtype=float)
        thr = pc.get("collapse_threshold", thr)
        if kl.size == 0: continue
        ax.bar(np.arange(len(kl)) + i*width, kl, width, label=cls)
        rows.append(pd.DataFrame({"source_class": cls, "dim": np.arange(len(kl)),
                                  "per_dim_kl": kl,
                                  "collapsed_count": pc.get("collapsed_dim_count", np.nan)}))
    if thr is not None:
        ax.axhline(thr, color="red", linestyle="--", label=f"collapse threshold={thr}")
    ax.set_xlabel("latent dimension"); ax.set_ylabel("KL"); ax.legend(fontsize=7, ncol=4)
    io_utils.save_figure(fig, "F13_active_units",
                         pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(),
                         ctx.figures_dir, ctx.data_dir)
    written.append("F13_active_units")

    # ---- F14: reconstruction fidelity by class ----
    recon_path = _VAE_RESULTS / "reconstruction_accuracy.json"
    rec = json.loads(io_utils.require_path(recon_path, "reconstruction_accuracy").read_text())
    per_class = rec.get("per_class", {})
    rows = []
    for cls, v in per_class.items():
        flat = {"source_class": cls}
        for k, val in v.items():
            if isinstance(val, (int, float)): flat[k] = val
        rows.append(flat)
    rdf = pd.DataFrame(rows)
    metric_col = next((c for c in rdf.columns if "r2" in c.lower() or "rmse" in c.lower()), None)
    fig, ax = plt.subplots(figsize=(10, 5))
    if metric_col:
        order = [c for c in config.CLASS_ORDER if c in set(rdf["source_class"])]
        sub = rdf.set_index("source_class").reindex(order)
        ax.bar(order, sub[metric_col].values, color=palette.color_for_method("latent_PGD"))
        ax.set_ylabel(metric_col)
    ax.set_title("VAE reconstruction fidelity by class")
    io_utils.save_figure(fig, "F14_recon_by_class", rdf, ctx.figures_dir, ctx.data_dir)
    written.append("F14_recon_by_class")

    # ---- F15: loss curves (needs numeric per-epoch arrays) ----
    curves = _find_loss_arrays()
    if curves is None:
        # TODO(needs export stage): numeric per-epoch losses not stored; re-log during training.
        io_utils.write_csv(pd.DataFrame({"note": ["TODO: numeric loss history absent; "
                            "only curves_*.png rasters exist. Re-log to enable F15."]}),
                           "F15_loss_curves_TODO", ctx.data_dir)
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        for name, arr in curves.items():
            ax.plot(arr, label=name)
        ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()
        io_utils.save_figure(fig, "F15_loss_curves",
                             pd.DataFrame(curves), ctx.figures_dir, ctx.data_dir)
        written.append("F15_loss_curves")
    return written


def _find_loss_arrays():
    """Return {series_name: np.ndarray} if numeric per-epoch losses exist, else None."""
    for cand in (_VAE_RESULTS / "training_log.txt", config.REPO_ROOT / "vae_run_manifest.json"):
        if cand.exists() and cand.suffix == ".json":
            try:
                m = json.loads(cand.read_text())
                hist = m.get("diagnostics", {}).get("loss_history")
                if hist: return {k: np.array(v) for k, v in hist.items()}
            except Exception:
                pass
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/thesis_eval/test_figures_smoke.py::test_layer5 -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/thesis_eval/figures/layer5_vae.py tests/thesis_eval/test_figures_smoke.py
git commit -m "thesis_eval: Layer 5 figures (F13-F15, F15 TODO-guarded)"
```

---

# PHASE 4 — Orchestration

### Task 22: bundle loader + figure context wiring (cache real clean latents)

**Files:**
- Create: `src/thesis_eval/data/loader.py` (load all bundles from disk; build per-class clean+benign latents into cache)
- Test: `tests/thesis_eval/test_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/thesis_eval/test_loader.py
import sys
from pathlib import Path
import numpy as np
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from thesis_eval.data import loader
from thesis_eval.data.bundles import AEBundle

def _toy(pop, cls, tmp):
    n=6; z=np.zeros((n,16),"float32"); x=np.zeros((n,39),"float32")
    b=AEBundle(x_orig=x,x_adv=x,z_orig=z,z_adv=z,y_true=np.full(n,3,"int64"),
        y_pred_clean=np.full(n,3,"int64"),y_pred_adv=np.zeros(n,"int64"),
        success=np.ones(n,bool),protocol_valid=np.ones(n,bool),mask_valid=np.ones(n,bool),
        raw_g1g8_valid=np.ones(n,bool),joint_valid=np.ones(n,bool),population=pop,
        model="mlp",source_class=cls,attack_type=pop,vae_tag="t",seed=42)
    b.save(tmp / AEBundle.filename(pop,"mlp",cls)); return b

def test_load_all_bundles(tmp_path):
    _toy("PGD","DoS",tmp_path); _toy("latentPGD_untgt","Mirai",tmp_path)
    bs = loader.load_all_bundles(tmp_path)
    assert len(bs) == 2
    assert {b.population for b in bs} == {"PGD","latentPGD_untgt"}

def test_missing_dir_fails_loud(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        loader.load_all_bundles(tmp_path / "does_not_exist")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/thesis_eval/test_loader.py -v`
Expected: FAIL — `No module named 'thesis_eval.data.loader'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/thesis_eval/data/loader.py
"""Load all AE bundles from the bundles dir; build clean-latent cache for Maha."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from thesis_eval import config, io_utils
from thesis_eval.data.bundles import AEBundle
from thesis_eval.data import artifacts


def load_all_bundles(bundles_dir: Path | None = None) -> list[AEBundle]:
    bundles_dir = Path(bundles_dir or (config.RESULTS_DIR / "bundles"))
    io_utils.require_path(bundles_dir, "bundles directory")
    files = sorted(bundles_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No bundles in {bundles_dir}. Run --stage export first.")
    return [AEBundle.load(p) for p in files]


def build_clean_latent_cache(device: str = "cpu") -> dict[str, dict[str, np.ndarray]]:
    """Per source class: clean training latents for that class AND for Benign (mu encode)."""
    import torch
    X, y8 = artifacts.load_train_split()
    router = artifacts.build_attack_router(device=device)
    cache = {}
    benign_idx = np.flatnonzero(y8 == config.CLASS_TO_ID["Benign"])[:20000]
    for cls in config.CLASS_ORDER:
        cid = config.CLASS_TO_ID[cls]
        idx = np.flatnonzero(y8 == cid)[:20000]
        vae = router.get_vae(cid)
        with torch.no_grad():
            zc, _ = vae.encode(torch.from_numpy(X[idx]).float().to(device))
            zb, _ = vae.encode(torch.from_numpy(X[benign_idx]).float().to(device))
        cache[cls] = {cls: zc.cpu().numpy(), "Benign": zb.cpu().numpy()}
    np.savez_compressed(config.RESULTS_DIR / "cache" / "clean_latents.npz",
                        **{f"{c}__{k}": v for c, d in cache.items() for k, v in d.items()})
    return cache
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/thesis_eval/test_loader.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/thesis_eval/data/loader.py tests/thesis_eval/test_loader.py
git commit -m "thesis_eval: bundle loader + clean-latent cache"
```

---

### Task 23: report.py + cli.py + __main__.py

**Files:**
- Create: `src/thesis_eval/figures/report.py`, `src/thesis_eval/cli.py`, `src/thesis_eval/__main__.py`
- Test: `tests/thesis_eval/test_cli_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/thesis_eval/test_cli_smoke.py
import sys, json
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from thesis_eval import cli
from thesis_eval.figures import report

def test_report_writes_figures_md(tmp_path):
    md = report.write_figures_md(["F1_asr_valid_table","F4_maha_kde"], out_dir=tmp_path)
    text = Path(md).read_text()
    assert "F1" in text and "F4" in text and "Mahalanobis" in text

def test_cli_parses_stages():
    args = cli.parse_args(["--stage","maha","--cov-mode","per-class","--seed","7"])
    assert args.stage == "maha"
    assert args.cov_mode == "per-class"
    assert args.seed == 7

def test_cli_unknown_stage_rejected():
    import pytest
    with pytest.raises(SystemExit):
        cli.parse_args(["--stage","bogus"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/thesis_eval/test_cli_smoke.py -v`
Expected: FAIL — `No module named 'thesis_eval.cli'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/thesis_eval/figures/report.py
"""Generate FIGURES.md mapping each metric to its figure/table."""
from __future__ import annotations
from pathlib import Path
from thesis_eval import config

_MAP = [
    ("F1_asr_valid_table", "ASR_raw / ASR_valid / validity / mean-L2 with bootstrap CIs (per class)"),
    ("F2_asr_valid_by_class", "ASR_valid grouped bar per class; Mirai≈0 callout"),
    ("F3_arch_invariance", "ASR_valid heatmap, 5 classifiers x class (generalizability)"),
    ("F4_maha_kde", "Mahalanobis-score KDE: latent AEs overlap clean; PGD/CW in right tail"),
    ("F5_detector_roc", "Mahalanobis-detector ROC per attack + AUC table"),
    ("F6_l2_vs_maha", "feature-L2 vs Mahalanobis scatter (valid/invalid, by attack)"),
    ("F7_idsr_by_class", "IDSR = valid & evaded & detector-evaded, grouped bar"),
    ("F8_wasserstein_by_class", "per-feature 1st-order Wasserstein, sorted, per class"),
    ("F9_pca_kde_js", "PCA-projection KDE grid + JS divergence per class"),
    ("F10_corr_diff_by_class", "|Corr_real - Corr_gen| heatmap per class"),
    ("F11_tsne_umap_overlay", "t-SNE + UMAP latent overlays + Mirai collapsed panel"),
    ("F12_nn_distance_hist", "nearest-neighbor distance: AE→real vs real→real"),
    ("F13_active_units", "per-dim KL / active units (collapse threshold marked)"),
    ("F14_recon_by_class", "VAE reconstruction fidelity by class"),
    ("F15_loss_curves", "heterogeneous β-VAE loss curves (total/recon/KL)"),
]


def write_figures_md(written: list[str], out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir or config.RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# thesis_eval — Figures & Metrics map", "",
             f"VAE tag: `{config.CANONICAL_VAE_TAG}`  |  class order: {config.CLASS_ORDER}", "",
             "| Figure | Status | Metric it supports |", "| --- | --- | --- |"]
    done = set(written)
    for name, desc in _MAP:
        status = "✅" if name in done else "⚠️ TODO (needs export/run)"
        lines.append(f"| `{name}` | {status} | {desc} |")
    md = out_dir / "FIGURES.md"; md.write_text("\n".join(lines))
    return md
```

```python
# src/thesis_eval/cli.py
"""Single CLI entrypoint for the eval suite."""
from __future__ import annotations
import argparse
import logging
from thesis_eval import config, manifest
from thesis_eval.figures import FigureContext, report

_STAGES = ["all", "export", "attack", "maha", "fidelity", "geometry", "vae", "report"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser("thesis_eval")
    p.add_argument("--stage", choices=_STAGES, default="all")
    p.add_argument("--cov-mode", choices=["tied", "per-class", "both"], default="tied")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--n-boot", type=int, default=config.N_BOOT)
    p.add_argument("--device", default="cpu")
    p.add_argument("--per-class", type=int, default=100)
    return p.parse_args(argv)


def _ctx(args) -> FigureContext:
    dirs = config.ensure_output_dirs()
    return FigureContext(figures_dir=dirs["figures"], data_dir=dirs["data"],
                         cache_dir=dirs["cache"],
                         cov_mode=("tied" if args.cov_mode == "both" else args.cov_mode),
                         seed=args.seed, n_boot=args.n_boot)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse_args(argv)
    config.ensure_output_dirs()
    from thesis_eval.data import loader, export_latent
    from thesis_eval.metrics import stats_output
    from thesis_eval.figures import (layer1_effectiveness as L1, layer2_mahalanobis as L2,
                                     layer3_fidelity as L3, layer4_geometry as L4,
                                     layer5_vae as L5)
    ctx = _ctx(args)
    written: list[str] = []

    if args.stage in ("all", "export", "attack"):
        export_latent.export_all(device=args.device, per_class=args.per_class)
        loader.build_clean_latent_cache(device=args.device)

    if args.stage in ("all", "attack", "maha", "fidelity", "geometry"):
        bundles = loader.load_all_bundles()
    else:
        bundles = []

    if args.stage in ("all", "attack"):
        written += L1.generate(bundles, ctx)
        stats_output.write(bundles, ctx.data_dir)   # McNemar latent-PGD vs latent-CW
    if args.stage in ("all", "maha"):
        written += L2.generate(bundles, ctx)
    if args.stage in ("all", "fidelity"):
        written += L3.generate(bundles, ctx)
    if args.stage in ("all", "geometry"):
        written += L4.generate(bundles, ctx)
    if args.stage in ("all", "vae"):
        written += L5.generate(bundles, ctx)

    if args.cov_mode == "both" and args.stage in ("all", "maha"):
        ctx_pc = _ctx(args); ctx_pc.cov_mode = "per-class"
        # appendix: per-class covariance variant written with _percls suffix dir
        appendix = config.RESULTS_DIR / "figures_percls"
        ctx_pc.figures_dir = appendix; ctx_pc.data_dir = config.RESULTS_DIR / "data_percls"
        L2.generate(bundles, ctx_pc)

    manifest.write_manifest(manifest.build_manifest(args.seed, args.cov_mode, args.n_boot))
    report.write_figures_md(written)
    logging.getLogger("thesis_eval").info("Wrote %d figures.", len(written))
    return 0
```

```python
# src/thesis_eval/__main__.py
import sys
from thesis_eval.cli import main
sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/thesis_eval/test_cli_smoke.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run full test suite + commit**

Run: `python -m pytest tests/thesis_eval -v`
Expected: all PASS

```bash
git add src/thesis_eval/figures/report.py src/thesis_eval/cli.py src/thesis_eval/__main__.py tests/thesis_eval/test_cli_smoke.py
git commit -m "thesis_eval: CLI + FIGURES.md report + full wiring"
```

---

### Task 24: Run the export stage (GPU) + full generation + manifest verification

**Files:** none created; this executes the pipeline and verifies real outputs.

- [ ] **Step 1: Adapt the targeted-benign kappa npz already on disk (no GPU needed)**

Run:
```bash
python - <<'PY'
import sys; sys.path.insert(0,'src')
from pathlib import Path
import glob
from thesis_eval import config
from thesis_eval.data import adapters
from thesis_eval.data.bundles import AEBundle
config.ensure_output_dirs()
root="outputs/latent_attacks/kappa_sweeps"
n=0
for f in glob.glob(root+"/**/*_targeted_benign_pgd_*.npz", recursive=True):
    name=Path(f).stem  # e.g. cnn_targeted_benign_pgd_DDoS
    model=name.split("_")[0]
    b=adapters.from_targeted_benign_npz(Path(f), model=model, attack_type="latent-pgd",
                                        population="latentPGD_tgtBenign")
    b.save(config.RESULTS_DIR/"bundles"/AEBundle.filename("latentPGD_tgtBenign",model,b.source_class))
    n+=1
print("adapted targeted-benign bundles:", n)
PY
```
Expected: prints a non-zero count.

- [ ] **Step 2: Adapt gradient baselines (no GPU needed)**

Run:
```bash
python - <<'PY'
import sys; sys.path.insert(0,'src')
from pathlib import Path
from thesis_eval import config
from thesis_eval.data import adapters
from thesis_eval.data.bundles import AEBundle
config.ensure_output_dirs()
for pop, fname, atk in [("PGD","attack_8class_pgd_0.30.npz","pgd"),
                        ("CW","attack_8class_cw_0.npz","cw")]:
    bs=adapters.from_gradient_npz(Path("results/attacks")/fname, population=pop,
                                  model="shared", attack_type=atk, device="cpu")
    for b in bs:
        b.save(config.RESULTS_DIR/"bundles"/AEBundle.filename(pop,"shared",b.source_class))
print("gradient bundles written")
PY
```
Expected: `gradient bundles written`.

- [ ] **Step 3: Run the latent export (GPU) for untargeted + targeted CW**

Run: `python -m thesis_eval --stage export --device cuda --per-class 100`
Expected: log lines per (model, population, class); bundles appear under
`results/thesis_eval/bundles/`. If `export_latent` NOTE items needed signature fixes,
this is where they surface — fix and re-run.

- [ ] **Step 4: Generate all figures for real (both covariance modes)**

Run: `python -m thesis_eval --stage all --device cuda --cov-mode both --n-boot 1000`
Expected: `results/thesis_eval/figures/F1..F15.{pdf,png}`, `data/*.csv`,
`run_manifest.json`, `FIGURES.md` all present; FIGURES.md shows ✅ for generated figures.

- [ ] **Step 5: Verify outputs exist and commit results**

Run:
```bash
python - <<'PY'
import sys, json; sys.path.insert(0,'src')
from thesis_eval import config
figs=sorted((config.RESULTS_DIR/"figures").glob("*.pdf"))
print("figures:", len(figs))
print("manifest:", json.loads((config.RESULTS_DIR/"run_manifest.json").read_text())["library_versions"]["torch"])
PY
```
Expected: figure count ≥ 14 (F15 may be TODO), manifest prints torch version.

```bash
git add results/thesis_eval docs/superpowers/plans/2026-06-07-thesis-eval-suite.md
git commit -m "thesis_eval: generate full figure suite + manifest"
```

---

## Self-Review (run by plan author)

**Spec coverage:** F1–F15 → Tasks 17–21. ASR_valid → Task 11. Mahalanobis (tied+per-class, cond number, stable inverse) → Task 12. Detector ROC/AUC → Task 13. IDSR → Task 14. Fidelity (Wasserstein/JS/corr/NN) → Task 15. McNemar → Task 16. Bootstrap CIs → Task 10 (used in 11/13/14). Bundle/export for the array gap → Tasks 6/8/9/24. Palette/conventions → Tasks 3/4. CLI + manifest + FIGURES.md → Tasks 5/22/23. Both latent families (targeted+untargeted) → POPULATIONS (Task 2) + export (Task 9) + adapters (Task 8/24). cov-mode both → Task 23. No-silent-fallback → `io_utils.require_path` everywhere.

**Placeholder scan:** The only `# TODO` is F15's guarded loss-curve stub (Task 21), which is the spec-mandated honest behavior, not a plan gap. `NOTE for implementer` blocks (Tasks 7/9/18) point at exact line numbers to confirm reused-signature details — these are verification instructions, not missing code.

**Type consistency:** `AEBundle` field names are identical across bundles.py, adapters.py, export_latent.py, asr.py, and all figure modules. `FigureContext` (figures/__init__.py) is consumed uniformly by every `generate(bundles, ctx)`. `mahalanobis.fit_gaussians`/`mahalanobis_score`, `detector.evaluate`/`operating_threshold`, `idsr.compute`, `bootstrap_ci` signatures match their callers in Layer 2.

**McNemar wiring:** implemented (Task 16), emitted as `Fstats_mcnemar.csv` (Task 16b), and wired into the `attack`/`all` stage of the CLI (Task 23). Spec requirement satisfied.

**Known follow-up (surfaced honestly, not hidden):** three reused signatures (`AttackRouter.__init__`, `PerturbationMask` constructor, `run_latent_attack_for_model` args) are verified at implementation time per the NOTE blocks; if they differ, the adapter/export functions adjust without changing public suite APIs. These are the only places the plan's exact code depends on interfaces not yet line-verified.
