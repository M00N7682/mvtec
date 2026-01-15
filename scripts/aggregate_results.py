import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.env312 import activate_local_deps

activate_local_deps()

import pandas as pd

from rdi.utils import RunPaths, load_json, ensure_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--categories",
        nargs="*",
        default=["bottle", "cable", "capsule", "hazelnut", "metal_nut"],
    )
    ap.add_argument("--ks", nargs="*", type=int, default=[1, 5, 10])
    args = ap.parse_args()

    rp = RunPaths(root=Path(__file__).resolve().parents[1])
    rows = []

    for cat in args.categories:
        for k in args.ks:
            # classifier baselines
            for method in ["no_synth", "copy_paste", "poisson_like"]:
                mpath = rp.outputs_dir / "baselines" / method / cat / f"k{k}" / "metrics.json"
                if mpath.exists():
                    obj = load_json(mpath)
                    rows.append(
                        {
                            "family": "clf",
                            "method": method,
                            "category": cat,
                            "k": k,
                            "auroc": obj["auroc"],
                        }
                    )

            # ours (RDI-Net)
            mpath = rp.outputs_dir / "rdi_net" / cat / f"k{k}" / "metrics.json"
            if mpath.exists():
                obj = load_json(mpath)
                rows.append({"family": "clf", "method": "rdi_net", "category": cat, "k": k, "auroc": obj["auroc"]})

            # padim
            ppath = rp.outputs_dir / "padim_lite" / cat / f"k{k}" / "metrics.json"
            if ppath.exists():
                obj = load_json(ppath)
                rows.append(
                    {"family": "normal_only", "method": "padim_lite", "category": cat, "k": k, "auroc": obj["auroc"]}
                )

            # bias check
            for method in ["copy_paste", "poisson_like"]:
                bpath = rp.outputs_dir / "bias_check" / method / cat / f"k{k}" / "metrics.json"
                if bpath.exists():
                    obj = load_json(bpath)
                    rows.append(
                        {
                            "family": "bias_check",
                            "method": f"{method}_real_vs_synth",
                            "category": cat,
                            "k": k,
                            "auroc": obj["auroc"],
                            "n_real": obj.get("n_real"),
                            "n_synth": obj.get("n_synth"),
                        }
                    )

    df = pd.DataFrame(rows)
    out_dir = ensure_dir(rp.outputs_dir / "baselines")
    out_path = out_dir / "summary.csv"
    df.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path} rows={len(df)}")


if __name__ == "__main__":
    main()


