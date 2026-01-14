import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--categories",
        nargs="*",
        default=["bottle", "cable", "capsule", "hazelnut", "metal_nut"],
    )
    ap.add_argument("--ks", nargs="*", type=int, default=[1, 5, 10])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--train_normals", type=int, default=200)
    ap.add_argument("--synth_defects", type=int, default=200)
    ap.add_argument("--clf_epochs", type=int, default=10)
    args = ap.parse_args()

    py = "/opt/homebrew/bin/python3.12"

    for cat in args.categories:
        for k in args.ks:
            # Baselines (classifier)
            for method in ["no_synth", "copy_paste", "poisson_like"]:
                _run(
                    [
                        py,
                        str(REPO_ROOT / "scripts" / "run_baselines.py"),
                        "--category",
                        cat,
                        "--k",
                        str(k),
                        "--method",
                        method,
                        "--seed",
                        str(args.seed),
                        "--image_size",
                        str(args.image_size),
                        "--train_normals",
                        str(args.train_normals),
                        "--synth_defects",
                        str(args.synth_defects if method != "no_synth" else 0),
                        "--clf_epochs",
                        str(args.clf_epochs),
                    ]
                )

            # PaDiM-lite (normal-only)
            _run(
                [
                    py,
                    str(REPO_ROOT / "scripts" / "run_padim.py"),
                    "--category",
                    cat,
                    "--k",
                    str(k),
                    "--seed",
                    str(args.seed),
                    "--image_size",
                    str(args.image_size),
                    "--train_normals",
                    str(args.train_normals),
                ]
            )

            # bias check for synth methods
            for method in ["copy_paste", "poisson_like"]:
                _run(
                    [
                        py,
                        str(REPO_ROOT / "scripts" / "run_bias_check.py"),
                        "--category",
                        cat,
                        "--k",
                        str(k),
                        "--method",
                        method,
                        "--seed",
                        str(args.seed),
                        "--image_size",
                        str(args.image_size),
                        "--real_source",
                        "both",
                    ]
                )

    print("\n[OK] full run finished. Next: aggregation/figures.")


if __name__ == "__main__":
    main()


