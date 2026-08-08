"""Plot TTFT and inter-token latency against prompt length.

    uv run python scripts/plot.py --out results/naive

Every engine under results/ is drawn as its own series, so the same command
produces a comparison chart once a second engine exists.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Measured by scripts/measure_overhead.py: a forward pass over a single token,
# synced only after all repeats. 18.54 ms of it is CPU, so this is the host
# floor a forward pass cannot go below. Drawn as a reference line because the
# three shortest prompts sit on it.
OVERHEAD_FLOOR_MS = 19.32


def load(include_warmup: bool = False) -> dict[str, list[dict]]:
    paths = sorted(RESULTS_DIR.glob("*/runs.csv"))
    if not paths:
        raise SystemExit(f"no results under {RESULTS_DIR}/*/runs.csv")

    rows: list[dict] = []
    for path in paths:
        with path.open() as f:
            rows.extend(csv.DictReader(f))
    if not include_warmup:
        rows = [r for r in rows if r["is_warmup"] != "True"]

    by_engine: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_engine[r["engine"]].append(r)
    return by_engine


def series(rows: list[dict], field: str) -> tuple[list[int], list[float]]:
    """Median of `field` per prompt, ordered by prompt length."""
    groups: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if r[field]:
            groups[int(r["prompt_tokens"])].append(float(r[field]))
    xs = sorted(groups)
    return xs, [statistics.median(groups[x]) for x in xs]


def annotate(ax, xs, ys, fmt: str) -> None:
    for x, y in zip(xs, ys):
        ax.annotate(fmt.format(y), (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)


def _overhead_line(ax) -> None:
    """Reference line for the measured per-pass host floor."""
    ax.axhline(OVERHEAD_FLOOR_MS, ls="--", lw=1, color="grey")
    ax.text(4096, OVERHEAD_FLOOR_MS * 1.02,
            f"{OVERHEAD_FLOOR_MS:.1f}ms measured host floor",
            fontsize=8, color="grey", ha="right", va="bottom")


def plot_ttft(by_engine: dict[str, list[dict]], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))

    for engine, rows in sorted(by_engine.items()):
        xs, ys = series(rows, "ttft_s")
        ys_ms = [y * 1000 for y in ys]
        ax.plot(xs, ys_ms, marker="o", label=engine)
        annotate(ax, xs, ys_ms, "{:.0f}ms")

    _overhead_line(ax)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([16, 64, 256, 1024, 4096])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("prompt tokens")
    ax.set_ylabel("time to first token (ms)")
    ax.set_title("TTFT vs prompt length — prefill cost")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    path = out / "ttft.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_itl(by_engine: dict[str, list[dict]], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))

    for engine, rows in sorted(by_engine.items()):
        xs, p50 = series(rows, "itl_p50_ms")
        _, p95 = series(rows, "itl_p95_ms")
        line, = ax.plot(xs, p50, marker="o", label=f"{engine} p50")
        ax.plot(xs, p95, marker="^", ls=":", color=line.get_color(),
                label=f"{engine} p95")
        ax.fill_between(xs, p50, p95, color=line.get_color(), alpha=0.12)
        annotate(ax, xs, p50, "{:.0f}ms")

    _overhead_line(ax)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([16, 64, 256, 1024, 4096])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("prompt tokens")
    ax.set_ylabel("inter-token latency (ms)")
    ax.set_title("Inter-token latency vs prompt length — decode cost")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    path = out / "itl.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=str(RESULTS_DIR),
                   help="directory to write ttft.png and itl.png into")
    p.add_argument("--include-warmup", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    by_engine = load(include_warmup=args.include_warmup)
    for path in (plot_ttft(by_engine, out), plot_itl(by_engine, out)):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
