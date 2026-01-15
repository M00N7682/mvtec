import csv
import sys
from pathlib import Path


def mean(xs: list[float]) -> float:
    return sum(xs) / max(1, len(xs))


def fmt(x: float) -> str:
    return f"{x:.4f}"


def main():
    repo = Path(__file__).resolve().parents[1]
    summary = repo / "outputs" / "baselines" / "summary.csv"
    out_dir = repo / "paper" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(summary, newline="") as f:
        for r in csv.DictReader(f):
            r["k"] = int(r["k"])
            r["auroc"] = float(r["auroc"])
            rows.append(r)

    # Baseline AUROC mean over categories
    acc: dict[tuple[str, int], list[float]] = {}
    for r in rows:
        if r["family"] == "clf":
            acc.setdefault((r["method"], r["k"]), []).append(r["auroc"])
        if r["family"] == "normal_only" and r["method"] == "padim_lite":
            acc.setdefault(("padim_lite", r["k"]), []).append(r["auroc"])

    def m(method: str, k: int) -> float:
        return mean(acc.get((method, k), []))

    baseline_tex = out_dir / "baseline_mean_rows.tex"
    has_rdi = all((("rdi_net", k) in acc and len(acc[("rdi_net", k)]) > 0) for k in (1, 5, 10))
    baseline_tex.write_text(
        "\n".join(
            [
                f"No synth & {fmt(m('no_synth',1))} & {fmt(m('no_synth',5))} & {fmt(m('no_synth',10))} \\\\",
                f"Copy-Paste & {fmt(m('copy_paste',1))} & {fmt(m('copy_paste',5))} & {fmt(m('copy_paste',10))} \\\\",
                f"Poisson-like & {fmt(m('poisson_like',1))} & {fmt(m('poisson_like',5))} & {fmt(m('poisson_like',10))} \\\\",
                f"PaDiM-lite (normal-only) & {fmt(m('padim_lite',1))} & {fmt(m('padim_lite',5))} & {fmt(m('padim_lite',10))} \\\\",
                (
                    f"RDI-Net (ours) & {fmt(m('rdi_net',1))} & {fmt(m('rdi_net',5))} & {fmt(m('rdi_net',10))} \\\\"
                    if has_rdi
                    else r"RDI-Net (ours) & -- & -- & -- \\"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Bias check mean over categories
    bias_acc: dict[tuple[str, int], list[float]] = {}
    for r in rows:
        if r["family"] == "bias_check":
            bias_acc.setdefault((r["method"], r["k"]), []).append(r["auroc"])

    def bm(method: str, k: int) -> float:
        return mean(bias_acc.get((method, k), []))

    bias_tex = out_dir / "bias_mean_table.tex"
    bias_tex.write_text(
        "\n".join(
            [
                r"\begin{table}[t]",
                r"\centering",
                r"\caption{Real-vs-synthetic patch 구분 AUROC(5개 카테고리 평균, 높을수록 합성 티가 강함).}",
                r"\label{tab:bias}",
                r"\begin{tabular}{lccc}",
                r"\toprule",
                r"Method & $K=1$ & $K=5$ & $K=10$ \\",
                r"\midrule",
                f"Copy-Paste(real vs synth) & {fmt(bm('copy_paste_real_vs_synth',1))} & {fmt(bm('copy_paste_real_vs_synth',5))} & {fmt(bm('copy_paste_real_vs_synth',10))} \\\\",
                f"Poisson-like(real vs synth) & {fmt(bm('poisson_like_real_vs_synth',1))} & {fmt(bm('poisson_like_real_vs_synth',5))} & {fmt(bm('poisson_like_real_vs_synth',10))} \\\\",
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[OK] wrote {baseline_tex}")
    print(f"[OK] wrote {bias_tex}")


if __name__ == "__main__":
    sys.exit(main())


