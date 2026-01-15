import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.env312 import activate_local_deps

activate_local_deps()

import matplotlib.pyplot as plt
import pandas as pd

from rdi.utils import RunPaths, ensure_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default=None, help="Path to summary.csv (default: outputs/baselines/summary.csv)")
    ap.add_argument(
        "--methods",
        nargs="*",
        default=["no_synth", "copy_paste", "poisson_like", "rdi_net"],
        help="Classifier methods to plot (e.g., no_synth copy_paste poisson_like rdi_net)",
    )
    args = ap.parse_args()

    rp = RunPaths(root=REPO_ROOT)
    in_csv = Path(args.in_csv) if args.in_csv else (rp.outputs_dir / "baselines" / "summary.csv")
    df = pd.read_csv(in_csv)

    df_clf = df[df["family"] == "clf"].copy()
    out_dir = ensure_dir(rp.outputs_dir / "figures")

    # mean over categories
    g = df_clf[df_clf["method"].isin(args.methods)].groupby(["method", "k"], as_index=False)["auroc"].mean()
    plt.figure(figsize=(6, 4))
    for method in args.methods:
        sub = g[g["method"] == method].sort_values("k")
        plt.plot(sub["k"], sub["auroc"], marker="o", label=method)
    plt.xlabel("K-shot defects")
    plt.ylabel("Image AUROC")
    plt.title("Mean AUROC over categories")
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / "auroc_vs_k_mean.png"
    plt.savefig(out_path, dpi=200)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()


