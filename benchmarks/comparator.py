"""Compare benchmark results and detect regressions."""
from typing import Dict


def _classify_change(baseline_mean: float, target_mean: float, threshold: float) -> str:
    """Classify a performance change as regression, improvement, or unchanged."""
    if baseline_mean == 0:
        if target_mean > 0:
            return "regression"
        return "unchanged"
    change = (target_mean - baseline_mean) / baseline_mean
    if change > threshold:
        return "regression"
    if change < -threshold:
        return "improvement"
    return "unchanged"


def _format_change(baseline_mean: float, target_mean: float) -> str:
    """Format a change as a percentage string."""
    if baseline_mean == 0:
        return "N/A"
    change = (target_mean - baseline_mean) / baseline_mean * 100
    sign = "+" if change > 0 else ""
    return f"{sign}{change:.0f}%"


def _icon(classification: str) -> str:
    """Return a unicode icon for a change classification."""
    return {"regression": "🔴", "improvement": "🟢", "unchanged": "⚪"}.get(classification, "?")


def compare_results(
    baseline: dict,
    target: dict,
    time_threshold: float = 0.2,
    memory_threshold: float = 0.3,
) -> str:
    """Compare two benchmark result sets and produce a report.

    Args:
        baseline: Baseline result dict (from JSON).
        target: Target result dict (from JSON).
        time_threshold: Fractional change to flag as time regression (>threshold).
        memory_threshold: Fractional change to flag as memory regression (>threshold).

    Returns:
        A human-readable report string.
    """
    baseline_results = {r["generator"]: r for r in baseline.get("results", [])}
    target_results = {r["generator"]: r for r in target.get("results", [])}

    baseline_branch = baseline.get("meta", {}).get("git_branch", "baseline")
    target_branch = target.get("meta", {}).get("git_branch", "target")

    lines = [
        f"Baseline: {baseline_branch}  →  Target: {target_branch}",
        "",
        f"{'Generator':<45} {'Scenario':<20} {'Time':<25} {'Memory':<25}",
        "-" * 115,
    ]

    regressions = 0
    improvements = 0
    unchanged = 0
    missing = 0
    new = 0

    all_generators = sorted(set(baseline_results.keys()) | set(target_results.keys()))

    for gen in all_generators:
        if gen not in target_results:
            lines.append(f"{gen:<45} {'(missing in target)':<20}")
            missing += 1
            continue
        if gen not in baseline_results:
            lines.append(f"{gen:<45} {'(new generator)':<20}")
            new += 1
            continue

        br = baseline_results[gen]
        tr = target_results[gen]

        if not br.get("performance") or not tr.get("performance"):
            continue

        b_time = br["performance"]["time_ms"]["mean"]
        t_time = tr["performance"]["time_ms"]["mean"]
        b_mem = br["performance"]["memory_mb"]["mean"]
        t_mem = tr["performance"]["memory_mb"]["mean"]

        time_class = _classify_change(b_time, t_time, time_threshold)
        mem_class = _classify_change(b_mem, t_mem, memory_threshold)

        time_str = f"{b_time:.1f}ms → {t_time:.1f}ms {_icon(time_class)} {_format_change(b_time, t_time)}"
        mem_str = f"{b_mem:.1f}MB → {t_mem:.1f}MB {_icon(mem_class)} {_format_change(b_mem, t_mem)}"

        lines.append(f"{gen:<45} {br['scenario']:<20} {time_str:<25} {mem_str:<25}")

        for cls in [time_class, mem_class]:
            if cls == "regression":
                regressions += 1
            elif cls == "improvement":
                improvements += 1
            else:
                unchanged += 1

    lines.append("")
    lines.append(f"Summary: {regressions} regressions, {improvements} improvements, "
                 f"{unchanged} unchanged, {missing} missing, {new} new")

    return "\n".join(lines)