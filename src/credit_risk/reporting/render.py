"""Report rendering: Jinja2 HTML → PDF via WeasyPrint.

Usage:
    from credit_risk.reporting.render import render_report
    render_report(metrics, figures_dir, output_path)
"""

from __future__ import annotations

import base64
import json
import logging
from datetime import date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_DEFAULT_OUTPUT = Path("reports") / "model_risk_report.pdf"


def _img_to_data_uri(path: Path) -> str:
    """Convert image file to base64 data URI for embedding in HTML."""
    if not path.exists():
        return ""
    suffix = path.suffix.lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "svg": "image/svg+xml"}.get(suffix, "image/png")
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{data}"


def _collect_figures(fig_dir: Path) -> dict[str, str]:
    """Collect figure paths from the figures directory as data URIs."""
    name_map = {
        "roc_curve": ["validation/roc_curve.png", "roc_curve.png"],
        "roc_oot_overlay": ["validation/roc_oot_overlay.png", "roc_oot_overlay.png"],
        "calibration_test": ["validation/calibration_test.png", "calibration_test.png"],
        "calibration_oot": ["validation/calibration_oot.png", "calibration_oot.png"],
        "ks_chart": ["validation/ks_chart.png", "ks_chart.png"],
        "gains_chart": ["validation/gains_chart.png", "gains_chart.png"],
        "psi_distribution": ["validation/psi_distribution.png", "psi_distribution.png"],
        "vintage_default_curves": ["vintage_default_curves.png"],
        "default_rate_by_grade": ["default_rate_by_grade.png"],
        "default_rate_by_term": ["default_rate_by_term.png"],
        "default_rate_by_purpose": ["default_rate_by_purpose.png"],
        "target_distribution": ["target_distribution.png"],
        "numeric_distributions": ["numeric_distributions.png"],
        "missingness": ["missingness.png"],
    }
    result: dict[str, str] = {}
    for key, candidates in name_map.items():
        for cand in candidates:
            p = fig_dir / cand
            if p.exists():
                result[key] = _img_to_data_uri(p)
                break
    return result


def render_html(
    metrics: dict,
    fig_dir: Path,
    model_version: str = "1.0.0",
) -> str:
    """Render the Jinja2 template to HTML string."""
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report.html.j2")

    figures = _collect_figures(fig_dir)

    toc = [
        {"title": "Executive Summary", "page": 3},
        {"title": "Data & Methodology", "page": 4},
        {"title": "PD Scorecard & Reject Inference", "page": 5},
        {"title": "Loss Given Default (LGD) Model", "page": 6},
        {"title": "Exposure at Default (EAD)", "page": 7},
        {"title": "Basel IRB Capital & Stress Testing", "page": 8},
        {"title": "IFRS 9 Expected Credit Loss", "page": 9},
        {"title": "Model Validation", "page": 10},
        {"title": "Limitations & Caveats", "page": 11},
        {"title": "Appendices", "page": 12},
    ]

    html = template.render(
        title="Credit Risk Model Risk Report",
        report_date=date.today().strftime("%d %B %Y"),
        model_version=model_version,
        metrics=metrics,
        figures=figures,
        toc=toc,
    )
    return html


def render_report(
    metrics: dict | None = None,
    fig_dir: Path | None = None,
    output_path: Path | None = None,
    metrics_json: Path | None = None,
) -> Path:
    """Render full PDF report.

    Parameters
    ----------
    metrics:
        Metrics dict. If None, loaded from metrics_json or outputs/metrics.json.
    fig_dir:
        Directory containing figure PNGs.
    output_path:
        Output PDF path.
    metrics_json:
        Path to metrics.json (used if metrics is None).

    Returns
    -------
    Path to generated PDF.
    """
    # Load metrics
    if metrics is None:
        p = metrics_json or Path("outputs") / "metrics.json"
        if p.exists():
            with open(p) as f:
                metrics = json.load(f)
            logger.info("Metrics loaded from %s", p)
        else:
            logger.warning("metrics.json not found at %s; report will show N/A values.", p)
            metrics = {}

    if fig_dir is None:
        fig_dir = Path("reports") / "figures"

    if output_path is None:
        output_path = _DEFAULT_OUTPUT

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Rendering HTML report...")
    html = render_html(metrics, Path(fig_dir))

    # Save HTML intermediate (useful for debugging)
    html_path = output_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")
    logger.info("HTML report written to %s", html_path)

    # Convert to PDF
    try:
        from weasyprint import HTML  # noqa: PLC0415

        HTML(string=html, base_url=str(output_path.parent)).write_pdf(str(output_path))
        logger.info("PDF report written to %s", output_path)
    except ImportError:
        logger.warning(
            "WeasyPrint not installed. HTML report available at %s. "
            "Install with: pip install weasyprint",
            html_path,
        )
        return html_path
    except Exception as exc:
        logger.error("PDF generation failed: %s", exc)
        logger.info("HTML report available at %s", html_path)
        return html_path

    return output_path


if __name__ == "__main__":
    from credit_risk.utils.logging import setup_logging
    setup_logging()
    render_report()
