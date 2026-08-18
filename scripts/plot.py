"""Figures for the engine writeups.

    uv run python scripts/plot.py --out results/cached
    uv run python scripts/plot.py --out results/naive --engines naive

Every engine under results/ is drawn as its own series, so the same command
produces a comparison chart once a second engine exists. `--engines` restricts
that set, which is how an earlier part keeps the figure it was written with.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import roofline  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Fixed hue per engine, assigned in part order and never cycled: an engine keeps
# its colour whichever chart it appears in, and adding a later engine cannot
# repaint an earlier one. Validated as a categorical palette (all pairs, light
# surface): worst CVD dE 9.2, worst normal-vision dE 24.0.
ENGINE_COLORS = {
    "naive": "#2a78d6",
    "cached": "#eb6834",
    "batched": "#1baf7a",
}
FALLBACK_COLORS = ["#eda100", "#e87ba4", "#008300"]

INK = "#3a3a38"
MUTED = "#8a8a85"

# Measured by scripts/measure_overhead.py: a forward pass over a single token,
# synced only after all repeats. 18.54 ms of it is CPU, so this is the host
# floor a forward pass cannot go below. Drawn because the three shortest naive
# prompts sit on it.
OVERHEAD_FLOOR_MS = 19.32

# 3.09 GB of weights at 600 GB/s. The memory traffic a decode step needs, and
# therefore the latency it would run at if bandwidth were the binding cost.
# Drawn as the other end of the same gap. See docs/arithmetic.md.
BANDWIDTH_FLOOR_MS = 5.1

# Published in results/naive/analysis.md, recorded before the cached engine
# existed: 19 to 20 ms per step, ~52 tok/s, flat at every prompt length.
NAIVE_PREDICTION_TPS = 52.0


def color_for(engine: str, index: int) -> str:
    if engine in ENGINE_COLORS:
        return ENGINE_COLORS[engine]
    return FALLBACK_COLORS[index % len(FALLBACK_COLORS)]


def ordered(by_engine: dict[str, list[dict]]) -> list[tuple[str, list[dict]]]:
    """Known engines in part order, then anything else alphabetically."""
    known = list(ENGINE_COLORS)
    return sorted(
        by_engine.items(),
        key=lambda kv: (known.index(kv[0]) if kv[0] in known else len(known), kv[0]),
    )


def load(include_warmup: bool = False, engines: list[str] | None = None) -> dict[str, list[dict]]:
    paths = sorted(RESULTS_DIR.glob("*/runs.csv"))
    if not paths:
        raise SystemExit(f"no results under {RESULTS_DIR}/*/runs.csv")

    rows: list[dict] = []
    for path in paths:
        with path.open() as f:
            rows.extend(csv.DictReader(f))
    if not include_warmup:
        rows = [r for r in rows if r["is_warmup"] != "True"]
    if engines:
        rows = [r for r in rows if r["engine"] in engines]
        if not rows:
            raise SystemExit(f"no recorded runs for engines {engines}")

    by_engine: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_engine[r["engine"]].append(r)
    return by_engine


def series(rows: list[dict], field: str) -> tuple[list[int], list[float]]:
    """Median of `field` per prompt, ordered by prompt length.

    Batch 1 only. These charts put prompt length on the x axis, so pooling
    several batch sizes into one point would average configurations that were
    never run together.
    """
    groups: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if r[field] and int(r.get("batch_size") or 1) == 1:
            groups[int(r["prompt_tokens"])].append(float(r[field]))
    xs = sorted(groups)
    return xs, [statistics.median(groups[x]) for x in xs]


def throughput_series(rows: list[dict]) -> tuple[list[int], list[float]]:
    """Decode tokens per second per prompt, aggregated the way the tables are.

    255 tokens over the mean decode time, not the mean of per-run rates, so a
    number here is the same number the writeups publish.
    """
    groups: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        if int(r.get("batch_size") or 1) == 1:
            groups[int(r["prompt_tokens"])].append(r)
    xs = sorted(groups)
    ys = []
    for x in xs:
        g = groups[x]
        gen = max(int(r["completion_tokens"]) for r in g)
        ys.append((gen - 1) / statistics.mean(float(r["decode_s"]) for r in g))
    return xs, ys


def batch_series(rows: list[dict], prompt_tokens: int) -> tuple[list[int], list[float]]:
    """Aggregate decode throughput per batch size, at one prompt length.

    Aggregate, not per sequence: rows in a batch share one wall clock, so the
    denominator counts each batch once. Summing decode_s per row instead
    reports mean per-sequence throughput, which is roughly flat in batch size
    and hides the entire result.
    """
    groups: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        if int(r["prompt_tokens"]) == prompt_tokens:
            groups[int(r.get("batch_size") or 1)].append(r)

    xs = sorted(groups)
    ys = []
    for batch_size in xs:
        group = groups[batch_size]
        per_batch = {
            (r.get("batch_id") or f"_row{i}"): float(r["decode_s"])
            for i, r in enumerate(group)
        }
        tokens = sum(int(r["completion_tokens"]) - 1 for r in group)
        seconds = sum(per_batch.values())
        ys.append(tokens / seconds if seconds else float("nan"))
    return xs, ys


def label_ends(ax, xs, ys, fmt: str, color: str, slot: int = 0) -> None:
    """First and last point only, one row per series.

    A number on every point is noise once there is more than one series; the
    ends carry the range, which is what these charts are read for. `slot`
    pushes each series' labels clear of the one before it, since two engines
    that agree at a point would otherwise print on top of each other.
    """
    dy = 10 + 15 * slot
    for i in (0, len(xs) - 1):
        ax.annotate(fmt.format(ys[i]), (xs[i], ys[i]), textcoords="offset points",
                    xytext=(0, dy), ha="center", fontsize=8.5, color=color)


def _reference(ax, y: float, text: str, xat: int = 256) -> None:
    """A horizontal control line, labelled mid axis and below, clear of the ends
    where the series labels sit."""
    ax.axhline(y, ls="--", lw=1, color=MUTED, zorder=0)
    ax.text(xat, y * 0.94, text, fontsize=8, color=MUTED, ha="center", va="top")


def _prompt_axis(ax) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks([16, 64, 256, 1024, 4096])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("prompt tokens")


def _finish(ax, title: str, ylabel: str, legend: bool = True) -> None:
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=INK)
    ax.grid(True, which="both", alpha=0.25, lw=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if legend:
        ax.legend(frameon=False, labelcolor=INK)


def plot_ttft(by_engine: dict[str, list[dict]], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (engine, rows) in enumerate(ordered(by_engine)):
        color = color_for(engine, i)
        xs, ys = series(rows, "ttft_s")
        ys_ms = [y * 1000 for y in ys]
        ax.plot(xs, ys_ms, marker="o", ms=6, lw=2, color=color, label=engine)
        label_ends(ax, xs, ys_ms, "{:.0f}ms", color, i)

    _reference(ax, OVERHEAD_FLOOR_MS, "19.3ms measured host floor")
    _prompt_axis(ax)
    ax.set_yscale("log")
    _finish(ax, "Time to first token against prompt length: prefill cost",
            "time to first token (ms)", legend=len(by_engine) > 1)

    path = out / "ttft.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_itl(by_engine: dict[str, list[dict]], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (engine, rows) in enumerate(ordered(by_engine)):
        color = color_for(engine, i)
        xs, p50 = series(rows, "itl_p50_ms")
        _, p95 = series(rows, "itl_p95_ms")
        ax.plot(xs, p50, marker="o", ms=6, lw=2, color=color, label=f"{engine} p50")
        ax.plot(xs, p95, marker="^", ms=6, ls=":", lw=1.6, color=color,
                label=f"{engine} p95")
        ax.fill_between(xs, p50, p95, color=color, alpha=0.10)
        label_ends(ax, xs, p50, "{:.0f}ms", color, i)

    _reference(ax, OVERHEAD_FLOOR_MS, "19.3ms measured host floor")
    _reference(ax, BANDWIDTH_FLOOR_MS,
               "5.1ms the memory traffic a step actually needs")
    _prompt_axis(ax)
    ax.set_yscale("log")
    _finish(ax, "Inter token latency against prompt length: decode cost",
            "inter token latency (ms)")

    path = out / "itl.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_prediction(by_engine: dict[str, list[dict]], out: Path) -> Path | None:
    """Measured decode throughput against the prediction recorded for it.

    Returns None unless the engine the prediction was about has runs, since the
    chart is the scoring of one specific claim and not a general view.
    """
    if "cached" not in by_engine:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (engine, rows) in enumerate(ordered(by_engine)):
        color = color_for(engine, i)
        xs, ys = throughput_series(rows)
        ax.plot(xs, ys, marker="o", ms=6, lw=2, color=color, label=f"{engine} measured")
        label_ends(ax, xs, ys, "{:.1f}", color, i)

    xs, _ = throughput_series(by_engine["cached"])
    ax.plot(xs, [NAIVE_PREDICTION_TPS] * len(xs), ls="--", lw=1.6,
            color=ENGINE_COLORS["cached"], alpha=0.6,
            label="cached predicted (~52 tok/s)")
    ax.annotate("52", (16, NAIVE_PREDICTION_TPS), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8.5,
                color=ENGINE_COLORS["cached"])

    _prompt_axis(ax)
    ax.set_ylim(0, NAIVE_PREDICTION_TPS * 1.25)
    _finish(ax, "What the cache was predicted to do, and what it did",
            "decode throughput (tok/s)")

    path = out / "prediction.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_step_budget(out: Path, engines: list[str] | None = None) -> Path | None:
    """How much of a decode step is memory traffic and how much is not.

    Only engines that hold a cache: for an engine that recomputes instead, the
    remainder is arithmetic rather than idle time, and one stacked bar cannot
    say both things at once.
    """
    data = roofline.load()
    rows = [
        (engine, prompt, m)
        for (engine, prompt, batch), m in sorted(data.items())
        if m["cached"] and batch == 1 and isinstance(prompt, int)
        and (not engines or engine in engines)
    ]
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels, traffic, rest = [], [], []
    for engine, prompt, m in rows:
        step_ms = 1000 * m["decode_s"] / (m["gen"] - 1)
        traffic_ms = 1000 * roofline.decode_bytes_per_step(m["seqs"], True) / roofline.BANDWIDTH
        labels.append(f"p{prompt}")
        traffic.append(traffic_ms)
        rest.append(step_ms - traffic_ms)

    x = range(len(labels))
    # 2px surface gap between the two segments, so the boundary reads as a
    # boundary rather than as a colour change.
    ax.bar(x, traffic, color=ENGINE_COLORS["cached"], label="memory traffic")
    ax.bar(x, rest, bottom=[t + 0.25 for t in traffic], color="#f0cdbd",
           label="everything else")
    for i, (t, r) in enumerate(zip(traffic, rest)):
        ax.text(i, t / 2, f"{t:.1f}", ha="center", va="center", fontsize=8.5, color="white")
        ax.text(i, t + r / 2, f"{100 * r / (t + r):.0f}%", ha="center", va="center",
                fontsize=8.5, color=INK)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("prompt tokens")
    # Headroom so the legend never sits on top of the tallest bar.
    ax.set_ylim(0, max(t + r for t, r in zip(traffic, rest)) * 1.3)
    _finish(ax, "Where a decode step goes, cached engine", "milliseconds per token")

    path = out / "step_budget.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_batch(by_engine: dict[str, list[dict]], out: Path, prompt_tokens: int) -> Path | None:
    """Aggregate throughput against batch size: this engine's headline result.

    Returns None when no engine has run above batch 1, since a single point is
    not a curve.
    """
    drawn = False
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (engine, rows) in enumerate(ordered(by_engine)):
        xs, ys = batch_series(rows, prompt_tokens)
        if len(xs) < 2:
            continue
        drawn = True
        color = color_for(engine, i)
        ax.plot(xs, ys, marker="o", ms=6, lw=2, color=color, label=engine)
        label_ends(ax, xs, ys, "{:.0f}", color, i)

        # Perfect scaling from the batch-1 point. Host cost is charged per
        # forward pass rather than per sequence, so the prediction is that the
        # measured curve tracks this until capacity or bandwidth binds.
        ax.plot(xs, [ys[0] * (x / xs[0]) for x in xs], ls="--", lw=1.4, alpha=0.5,
                color=color, label=f"{engine} ideal B x {ys[0]:.0f}")

    if not drawn:
        plt.close(fig)
        return None

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("batch size")
    _finish(ax, f"Throughput against batch size at {prompt_tokens} prompt tokens",
            "aggregate decode throughput (tok/s)")

    path = out / "batch.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=str(RESULTS_DIR),
                   help="directory to write the figures into")
    p.add_argument("--engines", default="",
                   help="comma separated engines to draw; default is all of them")
    p.add_argument("--include-warmup", action="store_true")
    p.add_argument("--batch-prompt", type=int, default=256,
                   help="prompt length for the throughput against batch chart")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    engines = [s.strip() for s in args.engines.split(",") if s.strip()] or None

    by_engine = load(include_warmup=args.include_warmup, engines=engines)
    written = (
        ("ttft.png", plot_ttft(by_engine, out)),
        ("itl.png", plot_itl(by_engine, out)),
        ("prediction.png", plot_prediction(by_engine, out)),
        ("step_budget.png", plot_step_budget(out, engines)),
        ("batch.png", plot_batch(by_engine, out, args.batch_prompt)),
    )
    for name, path in written:
        print(f"wrote {path}" if path else f"skipped {name}: nothing to draw")


if __name__ == "__main__":
    main()
