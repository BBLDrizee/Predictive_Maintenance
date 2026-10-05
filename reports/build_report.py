"""Build the project progress report PDF.

Figures are taken from the executed EDA notebook, and dataset statistics are
recomputed from data/raw, so rerun the notebook first if the analysis changes.

    python reports/build_report.py
"""

from __future__ import annotations

import base64
import io
from datetime import date
from pathlib import Path

import matplotlib
import nbformat
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, KeepTogether, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from turborul.config import REPO_ROOT, load_config
from turborul.data import load_all

OUT = REPO_ROOT / "reports" / "TurboRUL_Report.pdf"
NOTEBOOK = REPO_ROOT / "notebooks" / "01_eda.ipynb"

# Notebook figure order (see notebooks/01_eda.ipynb).
FIG_LIFETIMES, FIG_OP_SETTINGS, FIG_CONDITION_NORM, FIG_SENSOR_GRID, FIG_FAULT_MODES, FIG_TREND, FIG_RUL_CORR, FIG_SENSOR_CORR = range(8)

# DejaVu ships with matplotlib and covers ≤, ρ, ±, ≈ (the PDF base fonts do not).
_FONT_DIR = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DejaVu", _FONT_DIR / "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", _FONT_DIR / "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DejaVu-Oblique", _FONT_DIR / "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Oblique")

NAVY = colors.HexColor("#1F3A5F")
LIGHT = colors.HexColor("#EEF2F7")
GRID = colors.HexColor("#C5CED9")
MUTED = colors.HexColor("#5B6573")

_base = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("title", parent=_base["Title"], fontName="DejaVu-Bold", fontSize=20, leading=24, textColor=NAVY, spaceAfter=2),
    "subtitle": ParagraphStyle("subtitle", fontName="DejaVu", fontSize=10.5, leading=14, textColor=MUTED, alignment=TA_CENTER),
    "h1": ParagraphStyle("h1", fontName="DejaVu-Bold", fontSize=12.5, leading=16, textColor=NAVY, spaceBefore=10, spaceAfter=4),
    "h2": ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=10, leading=13, textColor=NAVY, spaceBefore=6, spaceAfter=2),
    "body": ParagraphStyle("body", fontName="DejaVu", fontSize=8.8, leading=12.2, spaceAfter=4),
    "cell": ParagraphStyle("cell", fontName="DejaVu", fontSize=7.8, leading=9.8),
    "cellb": ParagraphStyle("cellb", fontName="DejaVu-Bold", fontSize=7.8, leading=9.8, textColor=colors.white),
    "caption": ParagraphStyle("caption", fontName="DejaVu-Oblique", fontSize=7.6, leading=9.8, textColor=MUTED, spaceAfter=6),
    "box": ParagraphStyle("box", fontName="DejaVu", fontSize=8.8, leading=12.2),
}
PAGE_W = A4[0] - 36 * mm


def p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def bullets(items: list[str]) -> ListFlowable:
    return ListFlowable(
        [ListItem(p(t), leftIndent=10, value="•") for t in items],
        bulletType="bullet", bulletFontName="DejaVu", bulletFontSize=8, leftIndent=10, spaceAfter=4,
    )


def table(rows: list[list[str]], widths: list[float], align_right_from: int | None = None) -> Table:
    data = [[p(c, "cellb") for c in rows[0]]] + [[p(c, "cell") for c in r] for r in rows[1:]]
    t = Table(data, colWidths=[w * PAGE_W for w in widths], repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]
    t.setStyle(TableStyle(style))
    return KeepTogether([t, Spacer(1, 6)])


def callout(items: list[str]) -> Table:
    t = Table([[bullets(items)]], colWidths=[PAGE_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("LINEBEFORE", (0, 0), (0, -1), 3, NAVY),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def notebook_figures() -> list[bytes]:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    return [
        base64.b64decode(o["data"]["image/png"])
        for c in nb.cells if c.cell_type == "code"
        for o in c.outputs if "image/png" in o.get("data", {})
    ]


def figure(png: bytes, caption: str, width: float = 1.0) -> KeepTogether:
    w, h = ImageReader(io.BytesIO(png)).getSize()
    img_w = PAGE_W * width
    return KeepTogether([Image(io.BytesIO(png), width=img_w, height=img_w * h / w), Spacer(1, 2), p(caption, "caption")])


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("DejaVu", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, "TurboRUL — AI predictive maintenance for aircraft engines")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def dataset_rows(cfg: dict) -> list[list[str]]:
    rows = [["Subset", "Train engines", "Test engines", "Operating conditions", "Fault modes", "Life (min / median / max)",
             "Test histories < 30 cycles", "Test RUL ≤ 30"]]
    meta = {"FD001": ("1", "HPC"), "FD002": ("6", "HPC"), "FD003": ("1", "HPC + fan"), "FD004": ("6", "HPC + fan")}
    for name, ds in load_all(cfg["paths"]["raw_dir"], cfg["labels"]["rul_cap"]).items():
        life = ds.train.groupby("unit")["cycle"].max()
        test_len = ds.test.groupby("unit")["cycle"].max()
        rows.append([
            f"<b>{name}</b>", str(ds.train["unit"].nunique()), str(ds.test["unit"].nunique()), *meta[name],
            f"{life.min()} / {life.median():.0f} / {life.max()}",
            str(int((test_len < cfg["windows"]["window_len"]).sum())),
            str(int((ds.test_rul <= cfg["tiers"]["critical"]).sum())),
        ])
    return rows


def build() -> Path:
    cfg = load_config()
    figs = notebook_figures()
    story: list = []

    story += [
        p("TurboRUL", "title"),
        p("AI predictive maintenance for aircraft engines — project report", "subtitle"),
        p(f"Status: Phase 1 of 6 complete (setup and exploratory analysis) · {date.today():%d %B %Y}", "subtitle"),
        Spacer(1, 10),
    ]

    # 1. Summary
    story += [p("1. Summary", "h1"), p(
        "TurboRUL predicts the <b>Remaining Useful Life (RUL)</b> of turbofan engines, in flight cycles, from sensor "
        "histories, and turns those predictions into a ranked maintenance work list. It is built on NASA's public "
        "C-MAPSS run-to-failure dataset. This report covers the first phase: project setup, a tested data pipeline, "
        "and an exploratory analysis that fixes how sensors are selected and normalised before any model is trained. "
        "No model results exist yet; they arrive in Phases 2–3."),
        callout([
            "<b>Data pipeline working and tested.</b> All four subsets (FD001–FD004) load with capped RUL labels; 14 unit tests pass.",
            "<b>7 of 21 sensors carry no signal in FD001</b> (s1, s5, s6, s10, s16, s18, s19), matching published practice.",
            "<b>Sensor selection must be per subset.</b> s10 is flat in FD001 but strongly informative in FD003.",
            "<b>Per-condition normalisation is essential for FD002/FD004</b>: it lifts sensor–RUL correlation from |ρ| ≈ 0.1 to ≈ 0.7.",
            "<b>Two fault modes push five sensors in opposite directions</b> (FD003/FD004), which hides them from simple correlation "
            "and gives a ready signal for fault classification.",
            "<b>Data correction:</b> FD004 has 249 training and 248 test engines; the dataset readme swaps these counts.",
        ]),
    ]

    # 2. Problem and objectives
    story += [p("2. Problem and objectives", "h1"), p(
        "Airlines maintain engines either on fixed schedules, which replace parts that still have life left, or after "
        "failures, which ground aircraft unexpectedly. Predicting cycles-to-failure lets planners schedule shop visits "
        "just in time, during planned downtime. Each prediction will carry an uncertainty range, a risk tier "
        f"(Critical ≤ {cfg['tiers']['critical']}, Watch ≤ {cfg['tiers']['watch']}, otherwise Healthy), and the sensors that drove it."),
        table([
            ["Success metric", "Target"],
            ["RMSE on FD001 test set", "≤ 15 cycles (stretch ≤ 13)"],
            ["NASA asymmetric score", "Reported; beats own baselines"],
            ["Improvement over linear baseline", "≥ 25% lower RMSE"],
            ["Critical-engine recall (true RUL ≤ 30 flagged Critical)", "≥ 90%"],
            ["Coverage", "RMSE reported on all four subsets"],
            ["Demo latency", "Prediction in < 2 s"],
        ], [0.6, 0.4]),
    ]

    # 3. Dataset
    story += [p("3. Dataset", "h1"), p(
        "C-MAPSS simulates a fleet of large commercial turbofans. Each engine starts with unknown manufacturing wear, "
        "develops a fault, and is run cycle by cycle until a health margin crosses a failure threshold. Every cycle "
        "records 3 operating settings (altitude, Mach number, throttle) and 21 noisy sensors. Training files hold full "
        "run-to-failure histories; test files are cut off before failure, with the true RUL at the cut-off given "
        "separately. Counts below are measured from the files."),
        table(dataset_rows(cfg), [0.09, 0.1, 0.1, 0.12, 0.11, 0.18, 0.16, 0.14]),
        p(f"<b>Labels.</b> Training rows get RUL = last cycle − current cycle, clipped at <b>{cfg['labels']['rul_cap']} cycles</b> "
          "(configurable). Early-life engines show no measurable degradation, so a piecewise-linear target is standard. "
          "Test rows are labelled backwards from the official RUL at their last cycle, which is the only point scored."),
    ]

    # 4. Work completed
    story += [p("4. Work completed", "h1"), table([
        ["Component", "What it does"],
        ["<b>configs/default.yaml</b>", "Single source for RUL cap, window length, seeds, tier thresholds and paths (nothing hard-coded)."],
        ["<b>src/turborul/data.py</b>", "Parses the raw space-separated files (trailing spaces, no header), names the 26 columns, "
                                       "validates engine counts against RUL files, and labels train and test rows (FR-1)."],
        ["<b>tests/test_data.py</b>", "14 tests: countdown and cap logic, test-label back-fill, file parsing edge cases, and "
                                     "real-data checks (engine counts, RUL = 0 at failure, labels match official truth, contiguous cycles)."],
        ["<b>notebooks/01_eda.ipynb</b>", "Exploratory analysis below; uses training engines only for every sensor decision."],
        ["<b>Environment</b>", "Python 3.11 virtual environment with pinned versions; git repository initialised."],
    ], [0.27, 0.73])]

    # 5. Findings
    story += [p("5. Exploratory findings", "h1"), p("5.1 Flat sensors", "h2"), p(
        "In FD001, six sensors are exactly constant and s6 takes only two values with no trend. The remaining 14 drift "
        "steadily towards failure, most sharply in the last ~100 cycles — exactly the region the RUL cap focuses the "
        "model on. s9 and s14 (core speed) are near-duplicates (r = 0.96) and rise in some engines while falling in others."),
        figure(figs[FIG_SENSOR_GRID], "Figure 1. FD001 sensors aligned at failure: ten random engines (colour) and the fleet mean (black)."),
        p("5.2 Operating conditions", "h2"), p(
        "FD002 and FD004 fly six distinct conditions, which k-means recovers with near-zero spread. Raw sensors jump "
        "between condition levels every cycle, swamping degradation. Standardising each sensor within its condition "
        "reveals the trend — this preprocessing step matters more than model choice on these subsets."),
        figure(figs[FIG_CONDITION_NORM], "Figure 2. FD002 engine 1, sensor s4. Left: raw values coloured by operating condition. "
                                         "Right: after per-condition normalisation, with a smoothed trend.", 0.92),
        p("5.3 Which sensors carry signal", "h2"), p(
        "Global correlation with RUL can be misleading, so each sensor is scored by <b>trend strength</b>: the mean, over "
        "engines, of |Spearman ρ| between the sensor and cycle within one engine. A second score, <b>direction agreement</b>, "
        "shows whether engines trend the same way."),
        figure(figs[FIG_TREND], "Figure 3. Trend strength (top) and direction agreement (bottom) per subset, after per-condition "
                                "normalisation. Grey = constant sensor."),
        p("<b>Selection rule</b> (to be implemented in Phase 2, threshold in config): drop a sensor if it is constant within "
          "every condition or its trend strength is below 0.2."),
        table([
            ["Subset", "Dropped sensors", "Kept", "Weak but kept (0.2–0.4)"],
            ["FD001", "s1, s5, s6, s10, s16, s18, s19", "14", "—"],
            ["FD002", "s1, s5, s10, s18, s19", "16", "s6"],
            ["FD003", "s1, s5, s16, s18, s19", "16", "s6"],
            ["FD004", "s1, s5, s18, s19", "17", "s6, s10, s16"],
        ], [0.12, 0.4, 0.1, 0.38]),
        p("5.4 Fault modes split sensor directions", "h2"), p(
        "In FD003 and FD004, s7, s12, s15, s20 and s21 rise towards failure in some engines and fall in others "
        "(56 vs 44 engines in FD003). The two groups correspond to the two fault modes. Their global correlation with "
        "RUL is near zero, yet per engine they are among the strongest signals. Models therefore need per-engine trend "
        "features (slopes) as well as levels, and these sensors make fault classification (FR-13) straightforward."),
        figure(figs[FIG_FAULT_MODES], "Figure 4. FD003 engines grouped by whether s12 rises (red) or falls (blue) over life."),
    ]

    # 6. Implications
    story += [p("6. Decisions carried into modelling", "h1"), bullets([
        "Select sensors per subset with the trend-strength rule; never with a hard-coded list.",
        "Cluster operating conditions with k-means and normalise within condition for FD002/FD004, fitting both on training engines only.",
        "Include slope features alongside mean, std, min, max and last value for classical models.",
        f"Pad short test histories: 6 FD002 and 11 FD004 test engines have fewer than {cfg['windows']['window_len']} cycles.",
        "Expect coarse critical-recall steps on FD001: only 25 test engines have true RUL ≤ 30, so each counts for 4 points.",
        "Expect correlated sensors (e.g. s9/s14) to share SHAP credit and to destabilise linear-model coefficients.",
    ])]

    # 7. Next steps
    story += [KeepTogether([p("7. Next steps", "h1"), table([
        ["Phase", "Weeks", "Scope", "Exit criterion"],
        ["2 · Baselines", "3–4", "Preprocessing, engine-level split, 30-cycle windows, features, linear + XGBoost, metrics, training CLI",
         "FD001 test RMSE for both models in results.csv"],
        ["3 · Deep models", "5–6", "LSTM and 1D-CNN (PyTorch), Optuna tuning on validation engines, 5-seed runs",
         "Deep model beats XGBoost or the gap is explained"],
        ["4 · Robustness", "7–8", "FD002–FD004, uncertainty ranges, risk tiers, SHAP explanations", "RMSE on all four subsets"],
        ["5 · Product", "9–10", "FastAPI prediction endpoint, three-screen Streamlit dashboard", "All screens working locally"],
        ["6 · Showcase", "11–12", "Deployment, README, course report, demo video", "Public demo live"],
    ], [0.16, 0.09, 0.45, 0.30])])]

    # 8. Limitations
    story += [p("8. Limitations", "h1"), bullets([
        "The data is simulated, not from a real fleet, and failure is a threshold crossing rather than a removal decision.",
        "There is no cost, schedule or parts data, so maintenance value can only be argued, not measured.",
        "Test engines with true RUL above the cap (11 of 100 in FD001) cannot be predicted exactly by a capped model.",
        "Findings so far are descriptive; no predictive claims are made until models are scored once on the held-out test set.",
    ])]

    OUT.parent.mkdir(exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=18 * mm, title="TurboRUL — Project Report", author="TurboRUL")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return OUT


if __name__ == "__main__":
    print("wrote", build())
