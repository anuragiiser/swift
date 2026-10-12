"""Helpers shared by the ToxiGen reproduction scripts: repo checkout, installing
the reconstructed data module, and importing the authors' code."""
import os
import re
import subprocess
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO_DIR = os.path.join(HERE, "unlearning-entanglement")
EXP_SUBDIR = os.path.join("experiments", "toxigen_roberta")

# Git URL of the public release of the prior work (Cheng et al., 2026), whose
# training code and Original model this folder reproduces. Supplied at run time
# rather than hard-coded: set UPSTREAM_REPO_URL or pass --repo-url, or clone it
# yourself into --repo-dir.
REPO_URL = os.environ.get("UPSTREAM_REPO_URL", "")
REPO_HINT = ("Clone the public release of the prior work (Cheng et al., 2026) into "
             "{repo_dir}, or re-run with UPSTREAM_REPO_URL=<git url> (or --repo-url).")


def add_repo_arg(parser):
    parser.add_argument("--repo-dir", default=DEFAULT_REPO_DIR,
                        help="checkout of the prior work's public release")
    parser.add_argument("--repo-url", default=REPO_URL,
                        help="git URL to clone it from (default: $UPSTREAM_REPO_URL)")


def ensure_repo(repo_dir, clone=True, repo_url=None):
    if os.path.isdir(os.path.join(repo_dir, "src")):
        return
    repo_url = repo_url or REPO_URL
    if not clone:
        raise SystemExit(f"repo not found at {repo_dir} (drop --no-clone or pass --repo-dir)")
    if not repo_url:
        raise SystemExit("[error] no repository URL. "
                         + REPO_HINT.format(repo_dir=repo_dir))
    print(f"[setup] cloning {repo_url} -> {repo_dir}")
    subprocess.run(["git", "clone", "--depth", "1", repo_url, repo_dir], check=True)


def data_module_path(repo_dir):
    return os.path.join(repo_dir, "src", "data", "toxigen.py")


def install_data_module(repo_dir, max_length, split_seed):
    """Copy toxigen.py into <repo>/src/data/ with the chosen defaults baked in.

    The repo's run.py calls ``load_and_tokenize(num_proc=4)`` with no other
    arguments, so the module defaults must match the split that was prepared.
    """
    with open(os.path.join(HERE, "toxigen.py")) as fh:
        src = fh.read()
    for name, value in (("MAX_LENGTH", max_length), ("SPLIT_SEED", split_seed)):
        src, n = re.subn(rf"^{name} = \d+", f"{name} = {int(value)}", src, flags=re.M)
        if n != 1:
            raise RuntimeError(f"could not set {name} in toxigen.py")
    data_dir = os.path.dirname(data_module_path(repo_dir))
    os.makedirs(data_dir, exist_ok=True)
    open(os.path.join(data_dir, "__init__.py"), "a").close()
    with open(data_module_path(repo_dir), "w") as fh:
        fh.write(src)
    print(f"[setup] installed data module -> {data_module_path(repo_dir)} "
          f"(MAX_LENGTH={max_length}, SPLIT_SEED={split_seed})")


def read_forget_group(repo_dir):
    """Read FORGET_GROUP from the repo's common.py without importing torch."""
    path = os.path.join(repo_dir, EXP_SUBDIR, "common.py")
    try:
        with open(path) as fh:
            m = re.search(r'^FORGET_GROUP\s*=\s*["\']([^"\']+)["\']', fh.read(), flags=re.M)
        if m:
            return m.group(1)
    except OSError:
        pass
    return "lgbtq"


def _add_paths(repo_dir):
    for p in (os.path.abspath(repo_dir), os.path.abspath(os.path.join(repo_dir, EXP_SUBDIR))):
        if p not in sys.path:
            sys.path.insert(0, p)


def import_data_module(repo_dir):
    if not os.path.exists(data_module_path(repo_dir)):
        raise SystemExit(f"{data_module_path(repo_dir)} not found; run prepare_split.py first")
    _add_paths(repo_dir)
    import src.data.toxigen as tg
    return tg


def check_torch_for_transformers():
    """Fail early if transformers has disabled its PyTorch backend (e.g. torch too old)."""
    try:
        import torch
    except ImportError:
        raise SystemExit("[error] PyTorch is not installed. Install torch>=2.4 "
                         "(see https://pytorch.org/get-started/locally/).")
    import transformers
    from transformers.utils import is_torch_available
    if not is_torch_available():
        raise SystemExit(
            f"[error] transformers {transformers.__version__} cannot use torch {torch.__version__} "
            "(recent transformers needs torch>=2.4). Upgrade torch, e.g.\n"
            "  pip install --upgrade 'torch>=2.4' --index-url https://download.pytorch.org/whl/cu121\n"
            "or pin an older transformers, e.g. pip install 'transformers>=4.40,<4.57'.")


def import_repo(repo_dir):
    """Import the data module plus the repo pieces used for training/evaluation."""
    check_torch_for_transformers()
    tg = import_data_module(repo_dir)
    from src.utils import set_seed
    from src.evaluation import evaluate_on_three_datasets
    from common import FORGET_GROUP, build_collate
    return SimpleNamespace(tg=tg, set_seed=set_seed, build_collate=build_collate,
                           evaluate_on_three_datasets=evaluate_on_three_datasets,
                           FORGET_GROUP=FORGET_GROUP)
