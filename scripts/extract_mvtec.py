import argparse
import tarfile
from pathlib import Path


DEFAULT_CATEGORIES = ["bottle", "cable", "capsule", "hazelnut", "metal_nut"]


def _wanted(member_name: str, categories: list[str]) -> bool:
    # tar layout is usually: mvtec_anomaly_detection/<category>/...
    # We keep:
    # - top-level folder entry
    # - selected categories paths under mvtec_anomaly_detection/
    if member_name.rstrip("/") == "mvtec_anomaly_detection":
        return True
    prefix = "mvtec_anomaly_detection/"
    if not member_name.startswith(prefix):
        return False
    rest = member_name[len(prefix) :]
    if not rest:
        return True
    # rest begins with "<category>/..."
    cat = rest.split("/", 1)[0]
    return cat in categories


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tar",
        required=True,
        nargs="+",
        help=(
            "Path(s) to tar(.xz) file(s). Supports either the full dataset tarball or "
            "category-specific tarballs (e.g., bottle.tar.xz)."
        ),
    )
    ap.add_argument("--out", required=True, help="Output directory (e.g., data/processed/mvtec_ad)")
    ap.add_argument(
        "--categories",
        nargs="*",
        default=DEFAULT_CATEGORIES,
        help="Extract only these categories (default: 5 categories used in the paper).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    categories = list(args.categories or [])
    tar_paths = [Path(p).expanduser().resolve() for p in args.tar]
    for p in tar_paths:
        if not p.exists():
            raise SystemExit(f"tar not found: {p}")

    print(f"Categories: {categories} (only selected categories will be extracted)")
    for tar_path in tar_paths:
        print(f"Extracting: {tar_path} -> {out_dir}")
        with tarfile.open(tar_path, "r:*") as tf:
            members = [m for m in tf.getmembers() if _wanted(m.name, categories)]
            # For category-specific tarballs, members names may be "<category>/..." without
            # "mvtec_anomaly_detection/" prefix. In that case, we extract everything if the
            # top-level category matches our selection.
            if not members:
                top_levels = {m.name.split("/", 1)[0] for m in tf.getmembers() if m.name}
                if top_levels & set(categories):
                    tf.extractall(out_dir)
                else:
                    print(f"[SKIP] {tar_path} does not match selected categories.")
                    continue
            else:
                tf.extractall(out_dir, members=members)
    print("Done.")


if __name__ == "__main__":
    main()


