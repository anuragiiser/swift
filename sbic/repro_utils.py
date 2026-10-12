"""Helpers shared by the SBIC scripts: working folder, data module, torch check.

Same function names as the ToxiGen helpers, so the shared scripts run unchanged. There
is no external repository: --repo-dir is simply the working folder for the tokenized
data, sets, checkpoints and results (default ./sbic_work next to the scripts).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WORK_DIR = os.path.join(HERE, "sbic_work")


def add_repo_arg(parser):
    parser.add_argument("--repo-dir", default=DEFAULT_WORK_DIR,
                        help="working folder for data, sets, checkpoints and results")


def import_data_module(repo_dir):
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import sbic_data
    sbic_data.WORK_DIR = os.path.abspath(repo_dir)
    return sbic_data


def read_forget_group(repo_dir=None):
    """Default forget source; scripts override it with --forget-source."""
    import sbic_data
    return sbic_data.DEFAULT_FORGET_SOURCE


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
            "(recent transformers needs torch>=2.4). Upgrade torch, or pin an older transformers, "
            "e.g. pip install 'transformers>=4.40,<4.57' 'numpy<2'.")
