from __future__ import annotations

from invllava.report.tables import ResultCell, ResultRow, validate_comparison


def _escape(value: str) -> str:
    for source, destination in (("&", r"\&"), ("%", r"\%"), ("_", r"\_")):
        value = value.replace(source, destination)
    return value


def format_cell(cell: ResultCell, precision: int = 1) -> str:
    if cell.low is None or cell.high is None:
        return f"{cell.value:.{precision}f}"
    return f"{cell.value:.{precision}f} [{cell.low:.{precision}f}, {cell.high:.{precision}f}]"


def render_tabular(rows: list[ResultRow], metrics: list[str], *, precision: int = 1) -> str:
    validate_comparison(rows)
    columns = "l" + "r" * len(metrics)
    lines = [f"\\begin{{tabular}}{{{columns}}}", "\\toprule"]
    lines.append("Model & " + " & ".join(_escape(metric) for metric in metrics) + r" \\")
    lines.append("\\midrule")
    for row in rows:
        values = [format_cell(row.cells[metric], precision) for metric in metrics]
        lines.append(_escape(row.model) + " & " + " & ".join(values) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"
