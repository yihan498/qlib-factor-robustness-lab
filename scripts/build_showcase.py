"""Build GitHub, browser, and printable showcase artifacts from verified v2 output."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "portfolio_v2"
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
PDF_DIR = ROOT / "output" / "pdf"
EVIDENCE = ROOT / "evidence"
FONT_REGULAR = Path("C:/Windows/Fonts/msyh.ttc")
FONT_BOLD = Path("C:/Windows/Fonts/msyhbd.ttc")
COLORS = {
    "fixed_equal_signed": "#64748b",
    "rolling_ic_60": "#f59e0b",
    "adaptive_shrinkage": "#2563eb",
}
LABELS = {
    "fixed_equal_signed": "固定方向等权",
    "rolling_ic_60": "60日 Rolling-IC",
    "adaptive_shrinkage": "自适应收缩",
}


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def load() -> tuple[dict, dict]:
    v2 = json.loads((ARTIFACTS / "results.json").read_text(encoding="utf-8"))
    official = json.loads((EVIDENCE / "official_baseline_metrics.json").read_text(encoding="utf-8"))
    return v2, official


def setup_dirs() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)


def plot_performance(v2: dict, font: FontProperties) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.65, 1]})
    for name, color in COLORS.items():
        report = pd.read_csv(ARTIFACTS / f"backtest_{name}.csv", index_col=0, parse_dates=True)
        curve = (report["return"] - report["bench"] - report["cost"]).cumsum()
        ax1.plot(curve.index, curve, label=LABELS[name], color=color, linewidth=2)
    ax1.axhline(0, color="#94a3b8", linewidth=0.8)
    ax1.set_title("测试期成本后累计超额", fontproperties=font, fontsize=13, loc="left")
    ax1.set_ylabel("累计超额收益", fontproperties=font)
    ax1.grid(alpha=0.2)
    ax1.legend(prop=font, frameon=False)

    names = list(COLORS)
    values = [
        v2["test_results"][n]["portfolio"]["annualized_excess_return_after_cost"] for n in names
    ]
    bars = ax2.bar([LABELS[n] for n in names], values, color=[COLORS[n] for n in names])
    ax2.axhline(0, color="#334155", linewidth=0.8)
    ax2.set_title("成本后年化超额", fontproperties=font, fontsize=13, loc="left")
    ax2.tick_params(axis="x", rotation=20)
    for label in ax2.get_xticklabels():
        label.set_fontproperties(font)
    for bar, value in zip(bars, values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            value - 0.008,
            pct(value),
            ha="center",
            va="top",
            fontsize=9,
        )
    ax2.grid(axis="y", alpha=0.2)
    fig.tight_layout(rect=(0.03, 0.03, 0.98, 0.98))
    fig.savefig(ASSETS / "v2_performance.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_pipeline(v2: dict, font: FontProperties) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.axis("off")
    boxes = [
        (0.02, "社区示例数据\n479,497 行"),
        (0.22, "Alpha158\n30 个训练期因子"),
        (0.42, "两期信息滞后\n边界 Purge"),
        (0.62, "2016 验证\n12 个克制候选"),
        (0.82, "Qlib 执行回测\n部署 Gate: BLOCK"),
    ]
    for x, text in boxes:
        color = "#fee2e2" if x > 0.8 else "#eff6ff"
        edge = "#dc2626" if x > 0.8 else "#2563eb"
        ax.add_patch(
            plt.Rectangle(
                (x, 0.3),
                0.16,
                0.42,
                facecolor=color,
                edgecolor=edge,
                linewidth=1.6,
                transform=ax.transAxes,
            )
        )
        ax.text(
            x + 0.08,
            0.51,
            text,
            ha="center",
            va="center",
            fontproperties=font,
            fontsize=10,
            transform=ax.transAxes,
        )
        if x < 0.8:
            ax.annotate(
                "",
                xy=(x + 0.20, 0.51),
                xytext=(x + 0.16, 0.51),
                xycoords=ax.transAxes,
                arrowprops={"arrowstyle": "->", "color": "#64748b", "lw": 1.5},
            )
    ax.set_title(
        "从数据到部署决策的可审计管线",
        fontproperties=font,
        fontsize=17,
        fontweight="bold",
        loc="left",
    )
    ax.text(
        0.02,
        0.13,
        f"训练截止 {v2['temporal_controls']['train_effective_ic_cutoff']}  |  验证截止 {v2['temporal_controls']['validation_effective_ic_cutoff']}  |  测试 2017-2020",
        fontproperties=font,
        fontsize=10,
        color="#475569",
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "v2_pipeline.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_html(v2: dict, official: dict) -> None:
    adaptive = v2["test_results"]["adaptive_shrinkage"]
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qlib Factor Robustness Lab</title><link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect width='100' height='100' rx='18' fill='%232563eb'/><text x='50' y='68' text-anchor='middle' font-size='56' fill='white'>Q</text></svg>">
<style>
body{{margin:0;font-family:Inter,'Microsoft YaHei',sans-serif;background:#f8fafc;color:#0f172a}}main{{max-width:1080px;margin:auto;padding:48px 24px}}h1{{font-size:42px;margin:0 0 12px}}h2{{margin-top:42px}}.lead{{font-size:19px;color:#475569;max-width:820px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:16px;margin:28px 0}}.card{{background:white;padding:22px;border:1px solid #e2e8f0;border-radius:14px}}.k{{font-size:13px;color:#64748b}}.v{{font-size:28px;font-weight:700;margin-top:8px}}.good{{color:#047857}}.bad{{color:#b91c1c}}img{{width:100%;background:white;border:1px solid #e2e8f0;border-radius:14px;margin:12px 0}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:12px;border-bottom:1px solid #e2e8f0;text-align:left}}code{{background:#e2e8f0;padding:2px 5px;border-radius:4px}}.gate{{border-left:5px solid #dc2626}}footer{{margin-top:48px;color:#64748b;font-size:13px}}@media print{{body{{background:white}}main{{padding:12mm}}}}
</style></head><body><main>
<div class="k">求职作品集 / Quant Engineering Portfolio</div><h1>Qlib Factor Robustness Lab</h1>
<p class="lead">一个面向实务的量化研究验证管线：复现官方 LightGBM 基线，修复标签可得时间，配置化生成个人因子信号，并用 Qlib 的真实交易约束决定策略是否允许进入下一阶段。</p>
<div class="grid"><div class="card"><div class="k">官方 LightGBM 成本后年化超额</div><div class="v good">{pct(official["portfolio"]["annualized_excess_return_with_cost"])}</div></div>
<div class="card"><div class="k">个人自适应信号 Rank IC</div><div class="v">{adaptive["signal"]["rank_ic_mean"]:.4f}</div></div>
<div class="card"><div class="k">个人策略成本后年化超额</div><div class="v bad">{pct(adaptive["portfolio"]["annualized_excess_return_after_cost"])}</div></div>
<div class="card gate"><div class="k">自动部署决策</div><div class="v bad">BLOCK</div></div></div>
<img src="assets/v2_pipeline.png" alt="pipeline"><img src="assets/v2_performance.png" alt="performance">
<h2>我具体做了什么</h2><table><tr><th>模块</th><th>个人贡献</th></tr>
<tr><td>时间安全</td><td>按 Alpha158 的 T+1 到 T+2 标签设置两交易日信息滞后，并在训练/验证边界自动 purge。</td></tr>
<tr><td>配置化研究</td><td>用 YAML 固定数据、时间区间、候选参数和交易成本；输出配置 SHA-256。</td></tr>
<tr><td>执行验证</td><td>三组信号统一接入 Qlib TopkDropoutStrategy，处理涨跌停、手续费、持仓和换手。</td></tr>
<tr><td>质量门禁</td><td>9 项测试、84% 核心覆盖率；策略未战胜基线时明确阻止部署。</td></tr></table>
<h2>30 秒结论</h2><p>个人自适应方法提高了 rolling-IC 的信号质量，但成本后组合表现更差，因此没有被包装成盈利策略。项目展示的是可复现、能发现错误、能阻止伪 alpha 上线的量化工程能力。</p>
<footer>数据为社区维护的 Qlib 示例数据，底层来源标注为 Yahoo Finance；只用于教育和求职展示，不构成投资建议。</footer>
</main></body></html>"""
    (DOCS / "index.html").write_text(html, encoding="utf-8")


def build_pdf(v2: dict, official: dict) -> Path:
    pdfmetrics.registerFont(TTFont("MSYH", str(FONT_REGULAR)))
    pdfmetrics.registerFont(TTFont("MSYH-Bold", str(FONT_BOLD)))
    output = PDF_DIR / "interview-brief-v2.pdf"
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=13 * mm,
        bottomMargin=12 * mm,
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCN",
        parent=styles["Title"],
        fontName="MSYH-Bold",
        fontSize=22,
        leading=28,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#0f172a"),
    )
    h2 = ParagraphStyle(
        "H2CN",
        parent=styles["Heading2"],
        fontName="MSYH-Bold",
        fontSize=13,
        leading=18,
        spaceBefore=8,
        textColor=colors.HexColor("#1e3a8a"),
    )
    body = ParagraphStyle(
        "BodyCN",
        parent=styles["BodyText"],
        fontName="MSYH",
        fontSize=9.2,
        leading=14,
        textColor=colors.HexColor("#334155"),
    )
    small = ParagraphStyle(
        "SmallCN", parent=body, fontSize=7.8, leading=11, textColor=colors.HexColor("#64748b")
    )
    metric = v2["test_results"]["adaptive_shrinkage"]
    story = [
        Paragraph("Qlib Factor Robustness Lab", title),
        Paragraph("量化研究验证与部署闸门 | 求职展示简报 v2", h2),
        Paragraph(
            "目标不是包装一条赚钱曲线，而是建立一条可复现、无明显前视、能在真实交易约束下阻止伪 alpha 上线的研究管线。",
            body,
        ),
        Spacer(1, 4 * mm),
    ]
    table_data = [
        ["关键指标", "结果", "解读"],
        [
            "官方 LightGBM 成本后年化超额",
            pct(official["portfolio"]["annualized_excess_return_with_cost"]),
            "上游基线复现成功",
        ],
        ["个人自适应 Rank IC", f"{metric['signal']['rank_ic_mean']:.4f}", "有弱排序能力"],
        [
            "个人自适应成本后年化超额",
            pct(metric["portfolio"]["annualized_excess_return_after_cost"]),
            "未通过交易验证",
        ],
        ["部署状态", "BLOCK", "不进入实盘/下一阶段"],
    ]
    table = Table(table_data, colWidths=[62 * mm, 38 * mm, 75 * mm])
    table.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), "MSYH", 8.5),
                ("FONT", (0, 0), (-1, 0), "MSYH-Bold", 8.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ]
        )
    )
    story += [
        table,
        Spacer(1, 4 * mm),
        Image(str(ASSETS / "v2_performance.png"), width=180 * mm, height=69 * mm),
        Paragraph(
            "面试表达：我复现了官方基线，但个人方法在严格执行成本下失败。系统自动给出 BLOCK，避免把统计相关性误当成可交易收益。",
            body,
        ),
        Spacer(1, 2 * mm),
        Paragraph(
            "边界：社区 Qlib 示例数据，底层来源标注为 Yahoo Finance；结果仅用于教育和求职展示。",
            small,
        ),
        PageBreak(),
    ]
    story += [
        Paragraph("实现逻辑与个人贡献", title),
        Image(str(ASSETS / "v2_pipeline.png"), width=180 * mm, height=48 * mm),
        Paragraph("四项可追问贡献", h2),
    ]
    bullets = [
        "时间安全：Alpha158 标签预测 T+1 到 T+2，权重只使用至少滞后两个交易日的已实现 IC；训练和验证边界自动 purge。",
        "配置化：YAML 固定时间、候选参数、成本和持仓规则，运行结果记录环境版本与配置哈希。",
        "真实执行：个人信号接入 Qlib TopkDropoutStrategy，统一处理涨跌停、持仓变更、买卖成本和换手。",
        "质量门禁：9 项测试、84% 核心覆盖率；以成本后结果为部署条件，不按好看的 IC 选择结论。",
    ]
    for item in bullets:
        story.append(Paragraph("• " + item, body))
        story.append(Spacer(1, 1.5 * mm))
    params = v2["selected_adaptive_parameters"]
    story += [
        Paragraph("已验证环境", h2),
        Paragraph(
            f"Python {v2['runtime']['python']} | pyqlib {v2['runtime']['pyqlib']} | LightGBM {v2['runtime']['lightgbm']} | 479,497 行 | 546 只历史成分证券 | 868 个测试标签日 | 自适应参数: window={params['window']}, penalty={params['uncertainty_penalty']}, dynamic share={params['shrinkage_to_dynamic']}",
            body,
        ),
        Paragraph("招聘者可继续查看", h2),
        Paragraph(
            "README: 30 秒概览 | docs/index.html: 浏览器报告 | evidence/: 机器可读指标 | scripts/run_portfolio_v2.py: 端到端实现 | tests/: 时间边界与权重测试",
            body,
        ),
        Spacer(1, 5 * mm),
        Paragraph(
            "结论：这是一个量化工程与研究 QA 作品，不是实盘收益承诺。它的专业性来自可复现、能发现失败、能解释为什么不上线。",
            ParagraphStyle(
                "Conclusion",
                parent=body,
                fontName="MSYH-Bold",
                fontSize=11,
                leading=17,
                borderColor=colors.HexColor("#2563eb"),
                borderWidth=1,
                borderPadding=8,
                backColor=colors.HexColor("#eff6ff"),
            ),
        ),
    ]

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("MSYH", 7.5)
        canvas.setFillColor(colors.HexColor("#94a3b8"))
        canvas.drawString(15 * mm, 7 * mm, "Qlib Factor Robustness Lab | 求职展示简报 v2")
        canvas.drawRightString(A4[0] - 15 * mm, 7 * mm, f"{document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output


def publish_evidence(v2: dict) -> None:
    concise = {
        key: v2[key]
        for key in [
            "status",
            "runtime",
            "data_manifest",
            "temporal_controls",
            "selected_adaptive_parameters",
            "test_results",
            "interpretation_rule",
        ]
    }
    (EVIDENCE / "v2_verified_results.json").write_text(
        json.dumps(concise, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    setup_dirs()
    v2, official = load()
    font = FontProperties(fname=str(FONT_REGULAR))
    plot_performance(v2, font)
    plot_pipeline(v2, font)
    build_html(v2, official)
    publish_evidence(v2)
    pdf = build_pdf(v2, official)
    shutil.copyfile(pdf, DOCS / "interview-brief-v2.pdf")
    print(pdf)


if __name__ == "__main__":
    from build_showcase_v4 import main as build_current
    from reframe_showcase_v5 import main as reframe_current

    build_current()
    reframe_current()
