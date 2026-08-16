"""Build the institutional research-note presentation (v4)."""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
EVIDENCE = ROOT / "evidence"
PDF_DIR = ROOT / "output" / "pdf"
FONT_REGULAR = Path("C:/Windows/Fonts/msyh.ttc")
FONT_BOLD = Path("C:/Windows/Fonts/msyhbd.ttc")

NAVY = "#18324c"
SLATE = "#53677a"
MID = "#8998a6"
LIGHT = "#eef1f4"
RULE = "#c8d0d8"
FAIL = "#8f2d2d"
PASS = "#446b59"
METHODS = ["fixed_equal_signed", "rolling_ic_60", "adaptive_shrinkage"]
LABELS = dict(zip(METHODS, ["静态方向等权", "60日 Rolling-IC", "稳定性收缩组合"]))
METHOD_COLORS = dict(zip(METHODS, ["#596875", "#9aa6b1", NAVY]))
METHOD_STYLES = dict(zip(METHODS, ["--", ":", "-"]))
GATE_LABELS = {
    "test_days": "测试标签日",
    "rank_ic": "Rank IC",
    "rank_icir": "年化 Rank ICIR",
    "after_cost_annualized_excess": "成本后年化超额",
    "after_cost_information_ratio": "成本后信息比率",
    "max_drawdown": "最大回撤下限",
    "return_advantage_vs_best_control": "相对最佳对照收益优势",
    "ir_advantage_vs_best_control": "相对最佳对照 IR 优势",
}


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def load() -> tuple[dict, dict]:
    experiment = json.loads((ARTIFACTS / "results.json").read_text(encoding="utf-8"))
    official = json.loads((EVIDENCE / "official_baseline_metrics.json").read_text(encoding="utf-8"))
    return experiment, official


def reports() -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_csv(
            ARTIFACTS / f"backtest_{name}.csv", index_col=0, parse_dates=True
        )
        for name in METHODS
    }


def annual_rows(backtests: dict[str, pd.DataFrame]) -> list[list[str]]:
    rows = [["年份", *[LABELS[name] for name in METHODS]]]
    for year in sorted({year for frame in backtests.values() for year in frame.index.year}):
        values = []
        for name in METHODS:
            frame = backtests[name]
            period = frame.loc[frame.index.year == year]
            values.append(pct((period["return"] - period["bench"] - period["cost"]).sum()))
        rows.append([f"{year}{'（截至7月）' if year == 2020 else ''}", *values])
    return rows


def plot_research_design(experiment: dict, font: FontProperties) -> None:
    fig, ax = plt.subplots(figsize=(13.2, 3.5))
    ax.axis("off")
    columns = [
        ("DATA", "社区示例数据\nCSI300 历史成分\nAlpha158"),
        ("ESTIMATION", "2014-2015\n因子筛选与方向"),
        ("SELECTION", "2016\n12 项候选一次选参"),
        ("EVALUATION", "2017-2020\n冻结样本外执行"),
        ("DECISION", "Research Gate\nBLOCK · 3/8"),
    ]
    for i, (tag, body) in enumerate(columns):
        x = 0.02 + i * 0.195
        color = FAIL if tag == "DECISION" else NAVY
        ax.plot([x, x + 0.16], [0.77, 0.77], color=color, lw=3, transform=ax.transAxes)
        ax.text(x, 0.67, tag, color=color, fontsize=8.5, fontweight="bold", transform=ax.transAxes)
        ax.text(
            x,
            0.43,
            body,
            color=NAVY,
            fontsize=10,
            fontproperties=font,
            linespacing=1.55,
            va="center",
            transform=ax.transAxes,
        )
        if i < len(columns) - 1:
            ax.annotate(
                "",
                xy=(x + 0.184, 0.48),
                xytext=(x + 0.164, 0.48),
                xycoords=ax.transAxes,
                arrowprops={"arrowstyle": "->", "color": MID, "lw": 1},
            )
    tc = experiment["temporal_controls"]
    ax.text(
        0.02,
        0.09,
        f"Information controls  |  label lag: 2 trading sessions  |  "
        f"train IC cutoff: {tc['train_effective_ic_cutoff']}  |  "
        f"validation IC cutoff: {tc['validation_effective_ic_cutoff']}",
        fontsize=8.5,
        color=SLATE,
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "v4_research_design.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_temporal_control(experiment: dict, font: FontProperties) -> None:
    fig, ax = plt.subplots(figsize=(13.2, 3.4))
    ax.axis("off")
    segments = [
        (0.04, 0.34, "TRAIN", "2014-01 - 2015-12", "#29445f"),
        (0.38, 0.18, "VALIDATION", "2016-01 - 2016-12", "#587087"),
        (0.56, 0.40, "LOCKED TEST", "2017-01 - 2020-07", "#8594a2"),
    ]
    for x, width, label, dates, color in segments:
        ax.add_patch(
            plt.Rectangle((x, 0.64), width, 0.17, facecolor=color, edgecolor="none", transform=ax.transAxes)
        )
        ax.text(x + 0.012, 0.73, label, color="white", fontsize=8, fontweight="bold", transform=ax.transAxes)
        ax.text(x + 0.012, 0.665, dates, color="white", fontsize=8, transform=ax.transAxes)
    positions = [0.14, 0.31, 0.48, 0.65, 0.82]
    labels = ["T-2", "T-1", "T", "T+1", "T+2"]
    notes = ["最近可用 IC", "标签未完成", "形成信号", "标签起点", "标签终点"]
    ax.plot([positions[0], positions[-1]], [0.34, 0.34], color=RULE, lw=1.5, transform=ax.transAxes)
    for x, label, note in zip(positions, labels, notes):
        color = PASS if label == "T-2" else (FAIL if "+" in label else NAVY)
        ax.scatter([x], [0.34], color=color, s=45, transform=ax.transAxes, zorder=3)
        ax.text(x, 0.245, label, ha="center", color=NAVY, fontsize=9, transform=ax.transAxes)
        ax.text(x, 0.16, note, ha="center", color=SLATE, fontsize=8, fontproperties=font, transform=ax.transAxes)
    ax.text(
        0.96,
        0.06,
        "Label: T+1 -> T+2 return | Decision at T uses IC through T-2",
        ha="right",
        color=SLATE,
        fontsize=8.5,
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "v4_temporal_control.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_validation_matrix(experiment: dict, font: FontProperties) -> None:
    candidates = pd.DataFrame(experiment["validation_candidates"])
    candidates["row"] = candidates.apply(
        lambda row: f"W={int(row['window'])}, λ={row['uncertainty_penalty']:.1f}", axis=1
    )
    rows = ["W=60, λ=0.5", "W=60, λ=1.0", "W=120, λ=0.5", "W=120, λ=1.0"]
    cols = sorted(candidates["shrinkage_to_dynamic"].unique())
    matrix = candidates.pivot(
        index="row", columns="shrinkage_to_dynamic", values="rank_icir_annualized"
    ).reindex(rows)[cols]
    fig, ax = plt.subplots(figsize=(9.8, 3.8))
    image_map = ax.imshow(matrix.to_numpy(), cmap="Greys", aspect="auto", vmin=4.67, vmax=4.77)
    ax.set_xticks(range(3), [f"Dynamic share {value:.0%}" for value in cols], fontsize=9)
    ax.set_yticks(range(4), rows, fontsize=9)
    for i in range(4):
        for j in range(3):
            ax.text(j, i, f"{matrix.iloc[i, j]:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if matrix.iloc[i, j] > 4.73 else NAVY)
    selected = experiment["selected_adaptive_parameters"]
    row = rows.index(f"W={int(selected['window'])}, λ={selected['uncertainty_penalty']:.1f}")
    col = cols.index(selected["shrinkage_to_dynamic"])
    ax.add_patch(plt.Rectangle((col - 0.49, row - 0.49), 0.98, 0.98, fill=False, edgecolor=NAVY, lw=2.5))
    ax.set_title("Validation Rank ICIR by pre-specified candidate", loc="left", fontsize=11, color=NAVY)
    ax.text(0, -0.22, "Selected configuration outlined. Test-period results were not used for selection.",
            transform=ax.transAxes, fontsize=8, color=SLATE)
    fig.colorbar(image_map, ax=ax, fraction=0.027, pad=0.03)
    fig.tight_layout()
    fig.savefig(ASSETS / "v4_validation_matrix.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_oos_evidence(experiment: dict, backtests: dict[str, pd.DataFrame], font: FontProperties) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.6))
    cumulative: dict[str, pd.Series] = {}
    for name in METHODS:
        frame = backtests[name]
        series = (frame["return"] - frame["bench"] - frame["cost"]).cumsum()
        cumulative[name] = series
        axes[0, 0].plot(series.index, series, color=METHOD_COLORS[name], ls=METHOD_STYLES[name],
                        lw=2, label=LABELS[name])
        drawdown = series - series.cummax()
        axes[1, 0].plot(drawdown.index, drawdown, color=METHOD_COLORS[name],
                        ls=METHOD_STYLES[name], lw=1.7)
    axes[0, 0].set_title("A  After-cost cumulative excess return", loc="left", fontsize=10.5, color=NAVY)
    axes[1, 0].set_title("C  Excess-return drawdown", loc="left", fontsize=10.5, color=NAVY)
    axes[0, 0].legend(prop=font, frameon=False, fontsize=7.5, ncol=3)

    x = np.arange(3)
    before = [experiment["test_results"][name]["portfolio"]["annualized_excess_return_before_cost"] for name in METHODS]
    after = [experiment["test_results"][name]["portfolio"]["annualized_excess_return_after_cost"] for name in METHODS]
    axes[0, 1].bar(x - 0.18, before, 0.36, color="#c5ccd3", label="Before cost")
    axes[0, 1].bar(x + 0.18, after, 0.36, color=[METHOD_COLORS[name] for name in METHODS], label="After cost")
    axes[0, 1].set_xticks(x, ["Static", "Rolling IC", "Shrinkage"], fontsize=8)
    axes[0, 1].set_title("B  Annualized excess return", loc="left", fontsize=10.5, color=NAVY)
    axes[0, 1].legend(frameon=False, fontsize=8)

    annual = np.array([
        [
            (frame.loc[frame.index.year == year, "return"]
             - frame.loc[frame.index.year == year, "bench"]
             - frame.loc[frame.index.year == year, "cost"]).sum()
            for year in [2017, 2018, 2019, 2020]
        ]
        for frame in backtests.values()
    ])
    image_map = axes[1, 1].imshow(annual, cmap="RdGy", aspect="auto", vmin=-0.32, vmax=0.32)
    axes[1, 1].set_xticks(range(4), ["2017", "2018", "2019", "2020*"], fontsize=8)
    axes[1, 1].set_yticks(range(3), ["Static", "Rolling IC", "Shrinkage"], fontsize=8)
    axes[1, 1].set_title("D  Calendar-period arithmetic excess", loc="left", fontsize=10.5, color=NAVY)
    for i in range(3):
        for j in range(4):
            axes[1, 1].text(j, i, pct(annual[i, j]), ha="center", va="center", fontsize=8,
                            color="white" if abs(annual[i, j]) > 0.16 else NAVY)
    fig.colorbar(image_map, ax=axes[1, 1], fraction=0.025, pad=0.03)
    for ax in [axes[0, 0], axes[1, 0], axes[0, 1]]:
        ax.axhline(0, color=RULE, lw=0.8)
        ax.grid(alpha=0.15)
    fig.text(0.73, 0.015, "* 2020 through July", fontsize=7.5, color=SLATE)
    fig.tight_layout(pad=2)
    fig.savefig(ASSETS / "v4_oos_evidence.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def gate_value(value: float, key: str) -> str:
    if key in {"after_cost_annualized_excess", "max_drawdown", "return_advantage_vs_best_control"}:
        return pct(value)
    if key == "test_days":
        return str(int(value))
    return f"{value:.3f}"


def build_html(experiment: dict, official: dict, backtests: dict[str, pd.DataFrame]) -> None:
    gate = experiment["research_promotion_gate"]
    gate_rows = "".join(
        f"<tr><td>{GATE_LABELS[key]}</td><td>{gate_value(item['observed'], key)}</td>"
        f"<td>{gate_value(item['threshold'], key)}</td>"
        f"<td class='{'ok' if item['passed'] else 'bad'}'>{'PASS' if item['passed'] else 'FAIL'}</td></tr>"
        for key, item in gate["checks"].items()
    )
    result_rows = "".join(
        f"<tr><td>{LABELS[name]}</td><td>{experiment['test_results'][name]['signal']['rank_ic_mean']:.4f}</td>"
        f"<td>{experiment['test_results'][name]['signal']['rank_icir_annualized']:.3f}</td>"
        f"<td>{pct(experiment['test_results'][name]['portfolio']['annualized_excess_return_before_cost'])}</td>"
        f"<td>{pct(experiment['test_results'][name]['portfolio']['annualized_excess_return_after_cost'])}</td>"
        f"<td>{experiment['test_results'][name]['portfolio']['information_ratio_after_cost']:.3f}</td>"
        f"<td>{pct(experiment['test_results'][name]['portfolio']['max_drawdown_after_cost'])}</td></tr>"
        for name in METHODS
    )
    annual_html = "".join(
        "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row) + "</tr>"
        for row in annual_rows(backtests)[1:]
    )
    candidate_values = [item["rank_icir_annualized"] for item in experiment["validation_candidates"]]
    selected = experiment["selected_adaptive_parameters"]["rank_icir_annualized"]
    test_icir = experiment["test_results"]["adaptive_shrinkage"]["signal"]["rank_icir_annualized"]
    decay = 1 - test_icir / selected
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qlib Factor Robustness Lab — Research Note QR-001</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect width='100' height='100' fill='%2318324c'/><text x='50' y='66' text-anchor='middle' font-size='48' fill='white'>QR</text></svg>">
<style>
:root{{--navy:#18324c;--slate:#53677a;--mid:#8998a6;--light:#eef1f4;--rule:#c8d0d8;--fail:#8f2d2d;--pass:#446b59}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:white;color:var(--navy);font-family:"Segoe UI","Microsoft YaHei",Arial,sans-serif;font-size:15px;line-height:1.68}}
.top-rule{{height:6px;background:var(--navy)}}header{{max-width:1080px;margin:0 auto;padding:34px 28px 27px;border-bottom:1px solid var(--rule)}}
.series{{font-size:11px;letter-spacing:.14em;color:var(--slate);text-transform:uppercase}}h1{{font-size:33px;line-height:1.2;margin:8px 0 17px;font-weight:650;letter-spacing:-.02em}}
.meta{{display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}}.meta div{{padding:9px 12px 8px 0;font-size:12px;color:var(--slate)}}.meta b{{display:block;color:var(--navy);font-size:12px;font-weight:600}}
.layout{{max-width:1080px;margin:0 auto;display:grid;grid-template-columns:210px minmax(0,1fr);gap:44px;padding:32px 28px 80px}}aside{{font-size:12px;color:var(--slate)}}aside .sticky{{position:sticky;top:24px}}aside h2{{font-size:11px;letter-spacing:.1em;text-transform:uppercase;margin:0 0 8px;color:var(--navy)}}aside a{{display:block;color:var(--slate);text-decoration:none;padding:5px 0;border-bottom:1px solid #e5e9ed}}aside dl{{margin-top:28px}}aside dt{{margin-top:10px;color:var(--mid)}}aside dd{{margin:1px 0;color:var(--navy)}}article{{min-width:0}}article>section{{padding:0 0 38px;margin:0 0 38px;border-bottom:1px solid var(--rule)}}
h2.section-title{{font-size:21px;margin:0 0 17px;font-weight:650}}h3{{font-size:15px;margin:25px 0 8px;font-weight:650}}p{{margin:0 0 13px}}.summary{{font-size:17px;line-height:1.7}}.decision-line{{border-top:2px solid var(--navy);border-bottom:1px solid var(--rule);padding:12px 0;margin:20px 0;display:grid;grid-template-columns:160px 1fr;gap:20px}}.decision-line .label{{font-size:11px;letter-spacing:.08em;color:var(--slate);text-transform:uppercase}}.decision-line b{{color:var(--fail)}}.findings{{margin:15px 0;padding-left:22px}}.findings li{{padding:4px 0 8px}}
figure{{margin:24px 0}}figure img{{display:block;width:100%}}figcaption{{margin-top:8px;font-size:11px;color:var(--slate);border-top:1px solid var(--rule);padding-top:6px}}
.table-wrap{{overflow-x:auto;margin:17px 0 23px}}table{{border-collapse:collapse;width:100%;font-size:12.5px}}th{{text-align:left;color:var(--slate);font-weight:600;border-top:1.5px solid var(--navy);border-bottom:1px solid var(--navy);padding:7px 8px}}td{{border-bottom:1px solid #dde3e8;padding:8px;vertical-align:top}}.numeric td:not(:first-child),.numeric th:not(:first-child){{text-align:right;white-space:nowrap}}td.ok{{color:var(--pass);font-weight:650}}td.bad{{color:var(--fail);font-weight:650}}
.equation{{font-family:"Cambria Math","Times New Roman",serif;font-size:17px;padding:13px 16px;background:#f6f7f8;border-left:2px solid var(--navy);margin:12px 0}}.note{{font-size:13px;color:var(--slate);padding:12px 14px;background:#f6f7f8}}.red{{color:var(--fail)}}.small{{font-size:12px;color:var(--slate)}}footer{{max-width:1080px;margin:auto;padding:20px 28px 40px;border-top:1px solid var(--rule);font-size:11px;color:var(--slate)}}
@media(max-width:780px){{header{{padding:26px 20px}}h1{{font-size:28px}}.meta{{grid-template-columns:1fr 1fr}}.layout{{display:block;padding:24px 20px 60px}}aside{{margin-bottom:35px}}aside .sticky{{position:static}}aside nav{{display:none}}aside dl{{display:grid;grid-template-columns:1fr 1fr;gap:6px 18px;margin-top:0}}aside dl h2{{grid-column:1/-1}}aside dt{{margin-top:0}}aside dd{{margin:0;text-align:right}}.decision-line{{grid-template-columns:1fr}}}}
@media print{{.top-rule{{height:3px}}aside{{display:none}}.layout{{display:block;padding-top:20px}}article>section{{break-inside:auto}}figure,.table-wrap{{break-inside:avoid}}}}
</style></head><body><div class="top-rule"></div>
<header><div class="series">Quantitative Research Note · QR-001</div><h1>Qlib Factor Robustness Lab</h1>
<div class="meta"><div>Prepared by<b>Tian Yihan</b></div><div>Research window<b>2014-01 — 2020-07</b></div><div>Version / date<b>v4 · 2026-08-16</b></div><div>Promotion decision<b class="red">BLOCK · 3/8</b></div></div></header>
<div class="layout"><aside><div class="sticky"><nav><h2>Contents</h2><a href="#summary">Executive summary</a><a href="#scope">Mandate and scope</a><a href="#design">Research design</a><a href="#method">Methodology</a><a href="#results">Out-of-sample results</a><a href="#gate">Promotion gate</a><a href="#limits">Limitations</a></nav>
<dl><h2>Report facts</h2><dt>Universe</dt><dd>CSI300 history</dd><dt>Panel rows</dt><dd>{experiment['data_manifest']['feature_rows']:,}</dd><dt>Instruments</dt><dd>{experiment['data_manifest']['instruments']}</dd><dt>Test labels</dt><dd>868 days</dd><dt>Runtime</dt><dd>pyqlib {experiment['runtime']['pyqlib']}</dd><dt>Config</dt><dd>{experiment['runtime']['config_sha256'][:10]}…</dd></dl></div></aside>
<article>
<section id="summary"><h2 class="section-title">Executive summary</h2><p class="summary">本研究检验基于历史信息系数的稳定性加权，能否改善 Alpha158 截面因子组合在交易约束下的样本外表现。扩展方案在测试期取得正 Rank IC，但未形成正的成本后超额收益，也未优于两个预设对照。</p>
<div class="decision-line"><div class="label">Research decision</div><div><b>BLOCK — 不进入更高保真研究阶段。</b> 通过项为测试长度、Rank IC 与 Rank ICIR；成本后收益、信息比率、回撤及相对对照优势未达到配置阈值。</div></div>
<h3>Key findings</h3><ol class="findings"><li>稳定性收缩组合的成本后年化超额为 <b>-17.50%</b>，低于静态方向等权对照的 -11.59%。</li><li>三组方案的成本前超额均为负；交易成本约增加 4.5–4.8 个百分点的年化损失，但不是主要失败来源。</li><li>验证期候选 ICIR 仅分布在 {min(candidate_values):.3f}–{max(candidate_values):.3f}，候选面较平；冻结测试 ICIR 相对验证选择值下降 {decay:.1%}。</li><li>结果支持继续检查尾部 Top50 单调性、行业/规模暴露和换手约束，不支持“稳定性加权改善策略”的结论。</li></ol></section>
<section id="scope"><h2 class="section-title">1. Mandate and scope</h2><p>项目包含两条独立证据链。上游复现验证本地 Qlib 工作流；扩展评估检验独立研究假设。两条结果不构成同一模型的前后比较。</p>
<div class="table-wrap"><table><thead><tr><th>证据链</th><th>目的</th><th>Rank IC</th><th>成本后年化超额</th><th>结论范围</th></tr></thead><tbody><tr><td>Alpha158 + LightGBM 上游复现</td><td>验证框架、数据与执行环境</td><td>{official['signal']['Rank IC']:.4f}</td><td>{pct(official['portfolio']['annualized_excess_return_with_cost'])}</td><td>环境可运行</td></tr><tr><td>稳定性加权扩展</td><td>检验权重方法与研究治理</td><td>{experiment['test_results']['adaptive_shrinkage']['signal']['rank_ic_mean']:.4f}</td><td>{pct(experiment['test_results']['adaptive_shrinkage']['portfolio']['annualized_excess_return_after_cost'])}</td><td>未晋级</td></tr></tbody></table></div>
<figure><img src="assets/v4_research_design.png" alt="Research design"><figcaption>Figure 1. Research workflow and information controls. The decision stage is separated from estimation and selection.</figcaption></figure></section>
<section id="design"><h2 class="section-title">2. Research design</h2><p>训练、验证和测试按时间顺序隔离。训练期筛选 30 个因子并确定静态方向；验证期比较 12 个预设候选；测试期冻结参数。</p><figure><img src="assets/v4_temporal_control.png" alt="Temporal controls"><figcaption>Figure 2. Alpha158 target availability. A decision at T uses factor IC observed through T-2.</figcaption></figure>
<div class="table-wrap"><table><thead><tr><th>阶段</th><th>区间</th><th>用途</th><th>有效 IC 截止</th></tr></thead><tbody><tr><td>Train</td><td>2014-01 — 2015-12</td><td>因子筛选与静态方向</td><td>{experiment['temporal_controls']['train_effective_ic_cutoff']}</td></tr><tr><td>Validation</td><td>2016-01 — 2016-12</td><td>12 项候选一次选参</td><td>{experiment['temporal_controls']['validation_effective_ic_cutoff']}</td></tr><tr><td>Locked test</td><td>2017-01 — 2020-07</td><td>最终样本外评估</td><td>不再调参</td></tr></tbody></table></div></section>
<section id="method"><h2 class="section-title">3. Methodology</h2><div class="equation">IC<sub>j,t</sub> = Spearman(F<sub>j,t</sub>, R<sub>t+1→t+2</sub>)</div><div class="equation">w̃<sub>j,t</sub> = sign(μ<sub>j,t</sub>) · max(|μ<sub>j,t</sub>| − λ · σ<sub>j,t</sub>/√n<sub>j,t</sub>, 0)</div><p>动态权重与训练期静态方向收缩后按绝对权重和归一化。候选空间为窗口 60/120、惩罚 0.5/1.0、动态占比 25%/50%/75%。</p>
<figure><img src="assets/v4_validation_matrix.png" alt="Validation candidate matrix"><figcaption>Figure 3. Validation Rank ICIR across all pre-specified candidates. The narrow range indicates weak parameter separation.</figcaption></figure><p class="note">Selected configuration: window=60, penalty=1.0, dynamic share=75%. Validation Rank ICIR {selected:.3f}; locked-test Rank ICIR {test_icir:.3f}.</p></section>
<section id="results"><h2 class="section-title">4. Out-of-sample results</h2><figure><img src="assets/v4_oos_evidence.png" alt="Out-of-sample evidence"><figcaption>Figure 4. After-cost paths, cost impact, drawdown and calendar-period arithmetic excess. 2020 observations end in July.</figcaption></figure>
<div class="table-wrap"><table class="numeric"><thead><tr><th>方案</th><th>Rank IC</th><th>ICIR</th><th>成本前超额</th><th>成本后超额</th><th>IR</th><th>最大回撤</th></tr></thead><tbody>{result_rows}</tbody></table></div>
<h3>Calendar-period arithmetic excess, after cost</h3><div class="table-wrap"><table class="numeric"><thead><tr><th>年份</th><th>静态方向等权</th><th>60日 Rolling-IC</th><th>稳定性收缩组合</th></tr></thead><tbody>{annual_html}</tbody></table></div>
<p>稳定性收缩方案在四个日历分段均未取得正超额。结果的主要问题发生在成本之前；更高换手进一步扩大损失。</p></section>
<section id="gate"><h2 class="section-title">5. Research-promotion gate</h2><p>Gate 的适用范围仅为研究晋级，不代表生产或实盘批准。阈值由 YAML 配置读取，页面不生成决策。</p><div class="table-wrap"><table class="numeric"><thead><tr><th>检查项</th><th>观测值</th><th>阈值</th><th>结果</th></tr></thead><tbody>{gate_rows}</tbody></table></div></section>
<section id="limits"><h2 class="section-title">6. Limitations and next tests</h2><ul class="findings"><li>数据为社区 Qlib 示例数据，不等同于机构级 point-in-time 行情与成分股主数据；结论不外推到实盘。</li><li>全截面 IC 不能直接代表 Top50 尾部组合质量；下一步需要分位数组合和 top-k 敏感性分析。</li><li>行业、规模和风格暴露尚未显式中性化；当前回撤不能被归因于单一因子机制。</li><li>IC 显著性尚未对序列相关和多重检验进行校正；后续应采用 block bootstrap 并控制候选搜索偏差。</li><li>Gate 阈值是本研究的晋级条件，不是行业统一标准。</li></ul><p class="small">Reproducibility: Python {experiment['runtime']['python']} · pyqlib {experiment['runtime']['pyqlib']} · LightGBM {experiment['runtime']['lightgbm']} · 11 tests · 86.27% core coverage.</p></section>
</article></div><footer>Qlib Factor Robustness Lab · Research Note QR-001 · Educational and portfolio use only · Not investment advice</footer></body></html>"""
    (DOCS / "index.html").write_text(page, encoding="utf-8")
    (DOCS / "v4").mkdir(parents=True, exist_ok=True)
    (DOCS / "v4" / "index.html").write_text(
        page.replace('src="assets/', 'src="../assets/'), encoding="utf-8"
    )


def styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title", parent=sample["Title"], fontName="MSYH-Bold", fontSize=20,
            leading=26, alignment=TA_LEFT, textColor=colors.HexColor(NAVY),
        ),
        "series": ParagraphStyle(
            "Series", parent=sample["BodyText"], fontName="MSYH", fontSize=7.5,
            leading=10, textColor=colors.HexColor(SLATE), spaceAfter=4,
        ),
        "h1": ParagraphStyle(
            "H1", parent=sample["Heading1"], fontName="MSYH-Bold", fontSize=15,
            leading=20, textColor=colors.HexColor(NAVY), spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "H2", parent=sample["Heading2"], fontName="MSYH-Bold", fontSize=10.5,
            leading=15, textColor=colors.HexColor(NAVY), spaceBefore=6, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body", parent=sample["BodyText"], fontName="MSYH", fontSize=8.3,
            leading=12.5, textColor=colors.HexColor("#30465a"),
        ),
        "small": ParagraphStyle(
            "Small", parent=sample["BodyText"], fontName="MSYH", fontSize=7,
            leading=10, textColor=colors.HexColor(SLATE),
        ),
    }


def report_table(data: list[list[object]], widths: list[float], size: float = 7.2) -> Table:
    header = ParagraphStyle("TH", fontName="MSYH-Bold", fontSize=size, leading=size + 2, textColor=colors.HexColor(NAVY))
    body = ParagraphStyle("TD", fontName="MSYH", fontSize=size, leading=size + 2.2, textColor=colors.HexColor(NAVY))
    good = ParagraphStyle("GOOD", parent=body, fontName="MSYH-Bold", textColor=colors.HexColor(PASS))
    bad = ParagraphStyle("BAD", parent=body, fontName="MSYH-Bold", textColor=colors.HexColor(FAIL))

    def cell(value: object, row: int) -> Paragraph:
        style = header if row == 0 else (good if str(value) == "PASS" else bad if str(value) in {"FAIL", "BLOCK"} else body)
        return Paragraph(html.escape(str(value)), style)

    wrapped = [[cell(value, row_index) for value in row] for row_index, row in enumerate(data)]
    table = Table(wrapped, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 1.2, colors.HexColor(NAVY)),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(NAVY)),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor(RULE)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def build_pdf(experiment: dict, official: dict, backtests: dict[str, pd.DataFrame]) -> Path:
    pdfmetrics.registerFont(TTFont("MSYH", str(FONT_REGULAR)))
    pdfmetrics.registerFont(TTFont("MSYH-Bold", str(FONT_BOLD)))
    output = PDF_DIR / "quant-research-note-v4.pdf"
    doc = SimpleDocTemplate(
        str(output), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
        topMargin=15 * mm, bottomMargin=13 * mm,
    )
    s = styles()
    gate = experiment["research_promotion_gate"]
    selected = experiment["selected_adaptive_parameters"]["rank_icir_annualized"]
    test_icir = experiment["test_results"]["adaptive_shrinkage"]["signal"]["rank_icir_annualized"]
    candidate_values = [item["rank_icir_annualized"] for item in experiment["validation_candidates"]]
    story: list[object] = []

    metadata = [
        ["Prepared by", "Research window", "Version", "Promotion decision"],
        ["Tian Yihan", "2014-01 - 2020-07", "v4 · 2026-08-16", "BLOCK · 3/8"],
    ]
    evidence = [
        ["证据链", "目的", "Rank IC", "成本后年化超额", "结论范围"],
        ["上游 Alpha158 + LightGBM", "验证框架与执行环境", f"{official['signal']['Rank IC']:.4f}",
         pct(official["portfolio"]["annualized_excess_return_with_cost"]), "环境可运行"],
        ["稳定性加权扩展", "检验研究假设与治理", f"{experiment['test_results']['adaptive_shrinkage']['signal']['rank_ic_mean']:.4f}",
         pct(experiment["test_results"]["adaptive_shrinkage"]["portfolio"]["annualized_excess_return_after_cost"]), "未晋级"],
    ]
    story += [
        Paragraph("QUANTITATIVE RESEARCH NOTE · QR-001", s["series"]),
        Paragraph("Qlib Factor Robustness Lab", s["title"]),
        report_table(metadata, [42 * mm, 51 * mm, 42 * mm, 43 * mm], 7.2),
        Spacer(1, 5 * mm),
        Paragraph("Executive summary", s["h1"]),
        Paragraph(
            "本研究检验基于历史信息系数的稳定性加权，能否改善 Alpha158 截面因子组合在交易约束下的样本外表现。扩展方案在测试期取得正 Rank IC，但未形成正的成本后超额收益，也未优于两个预设对照。",
            s["body"],
        ),
        Paragraph("Research decision", s["h2"]),
        Paragraph(
            "<font color='#8f2d2d'><b>BLOCK - 不进入更高保真研究阶段。</b></font> "
            "通过项为测试长度、Rank IC 与 Rank ICIR；成本后收益、信息比率、最大回撤及相对对照优势未达到配置阈值。",
            s["body"],
        ),
        Paragraph("Key findings", s["h2"]),
        Paragraph(
            "1. 稳定性收缩组合成本后年化超额 -17.50%，低于静态对照 -11.59%。<br/>"
            "2. 三组方案成本前超额均为负；成本增加约 4.5-4.8 个百分点的年化损失。<br/>"
            f"3. 验证候选 ICIR 范围 {min(candidate_values):.3f}-{max(candidate_values):.3f}，参数分离较弱；测试 ICIR 相对选择值下降 {1 - test_icir / selected:.1%}。<br/>"
            "4. 结果支持进一步检查尾部选股、风险暴露与换手，不支持策略改进结论。",
            s["body"],
        ),
        Paragraph("Evidence streams", s["h2"]),
        report_table(evidence, [41 * mm, 51 * mm, 23 * mm, 34 * mm, 29 * mm], 7),
        Spacer(1, 5 * mm),
        Image(str(ASSETS / "v4_research_design.png"), width=178 * mm, height=47 * mm),
        PageBreak(),
    ]

    controls = [
        ["风险", "控制", "证据"],
        ["标签信息提前使用", "IC 权重滞后 2 个交易日", "temporal.py / boundary tests"],
        ["训练/验证边界污染", "交易日 purge", "effective cutoff in results.json"],
        ["测试期反复调参", "12 项候选只在 2016 选择", "complete candidate matrix"],
        ["执行假设不一致", "统一 Top50 / Drop5 / 成本", "three Qlib backtest CSVs"],
        ["研究结论越级", "8 项机器 Gate", "gating.py / YAML thresholds"],
    ]
    story += [
        Paragraph("1 · Research design and controls", s["h1"]),
        Paragraph(
            "训练、验证和测试按时间顺序隔离。训练期筛选因子并确定静态方向；验证期一次选择参数；测试期冻结。",
            s["body"],
        ),
        Image(str(ASSETS / "v4_temporal_control.png"), width=178 * mm, height=46 * mm),
        Paragraph("Information availability", s["h2"]),
        Paragraph(
            "Alpha158 标签为 T+1 到 T+2 收益。T 日权重只使用截至 T-2 已完成标签所对应的 IC。训练有效 IC 截止 "
            f"{experiment['temporal_controls']['train_effective_ic_cutoff']}；验证有效 IC 截止 "
            f"{experiment['temporal_controls']['validation_effective_ic_cutoff']}。",
            s["body"],
        ),
        Paragraph("Control matrix", s["h2"]),
        report_table(controls, [48 * mm, 65 * mm, 65 * mm], 7.1),
        Paragraph("Execution assumptions", s["h2"]),
        report_table(
            [["Universe", "Benchmark", "Portfolio", "Costs", "Limit threshold"],
             ["CSI300 history", "SH000300", "Top50 / Drop5", "5 bp buy / 15 bp sell", "9.5%"]],
            [38 * mm, 32 * mm, 34 * mm, 45 * mm, 29 * mm], 7,
        ),
        Paragraph("Scope boundary", s["h2"]),
        Paragraph(
            "Microsoft Qlib 提供 Alpha158、LightGBM 工作流、TopkDropoutStrategy 与交易模拟器。项目独立模块负责 IC、时序控制、稳定性权重、参数治理、Gate 与证据生成；上游源代码未被修改。",
            s["body"],
        ),
        PageBreak(),
    ]

    story += [
        Paragraph("2 · Methodology and parameter selection", s["h1"]),
        Paragraph("Daily cross-sectional Rank IC", s["h2"]),
        Paragraph("IC[j,t] = Spearman(F[j,t], R[t+1→t+2])", s["body"]),
        Paragraph("Stability-adjusted weight", s["h2"]),
        Paragraph("w_adj[j,t] = sign(mu[j,t]) * max(|mu[j,t]| - lambda * sigma[j,t] / sqrt(n[j,t]), 0)", s["body"]),
        Paragraph(
            "动态权重与训练期静态方向按验证期比例收缩，并按绝对权重和归一化。候选空间为窗口 60/120、惩罚 0.5/1.0、动态占比 25%/50%/75%。",
            s["body"],
        ),
        Spacer(1, 4 * mm),
        Image(str(ASSETS / "v4_validation_matrix.png"), width=165 * mm, height=64 * mm),
        Paragraph("Selection assessment", s["h2"]),
        Paragraph(
            f"验证期选定 window=60、penalty=1.0、dynamic share=75%。12 项候选 ICIR 仅分布在 {min(candidate_values):.3f}-{max(candidate_values):.3f}，最优与次优差异较小，参数识别并不强。冻结测试 ICIR 为 {test_icir:.3f}，相对验证选择值 {selected:.3f} 下降 {1 - test_icir / selected:.1%}。",
            s["body"],
        ),
        Paragraph("Interpretation", s["h2"]),
        Paragraph(
            "验证期高 ICIR 未在测试期维持，可能反映时间状态变化、有限样本估计误差或候选选择偏差。由于测试参数冻结，该落差被作为结果披露，而非再次调参消除。",
            s["body"],
        ),
        PageBreak(),
    ]

    result_data = [["方案", "Rank IC", "ICIR", "成本前", "成本后", "IR", "最大回撤"]]
    for name in METHODS:
        item = experiment["test_results"][name]
        result_data.append([
            LABELS[name], f"{item['signal']['rank_ic_mean']:.4f}",
            f"{item['signal']['rank_icir_annualized']:.3f}",
            pct(item["portfolio"]["annualized_excess_return_before_cost"]),
            pct(item["portfolio"]["annualized_excess_return_after_cost"]),
            f"{item['portfolio']['information_ratio_after_cost']:.3f}",
            pct(item["portfolio"]["max_drawdown_after_cost"]),
        ])
    story += [
        Paragraph("3 · Locked out-of-sample results", s["h1"]),
        Image(str(ASSETS / "v4_oos_evidence.png"), width=178 * mm, height=106 * mm),
        report_table(result_data, [38 * mm, 23 * mm, 22 * mm, 25 * mm, 25 * mm, 20 * mm, 25 * mm], 6.8),
        Paragraph("Calendar-period arithmetic excess, after cost", s["h2"]),
        report_table(annual_rows(backtests), [39 * mm, 47 * mm, 47 * mm, 45 * mm], 7),
        Paragraph("Assessment", s["h2"]),
        Paragraph(
            "稳定性收缩方案在 2017、2018、2019 和 2020 年截至 7 月的分段中均未取得正超额。三组方案在成本前已为负，因此失败不能主要归因于手续费；更高换手只进一步扩大了损失。",
            s["body"],
        ),
        PageBreak(),
    ]

    gate_data = [["检查项", "观测值", "阈值", "结果"]]
    for key, item in gate["checks"].items():
        gate_data.append([
            GATE_LABELS[key], gate_value(item["observed"], key), gate_value(item["threshold"], key),
            "PASS" if item["passed"] else "FAIL",
        ])
    story += [
        Paragraph("4 · Promotion decision, limitations and next tests", s["h1"]),
        Paragraph("Research-promotion gate", s["h2"]),
        report_table(gate_data, [68 * mm, 37 * mm, 37 * mm, 36 * mm], 7.1),
        Paragraph("Decision", s["h2"]),
        Paragraph(
            "<font color='#8f2d2d'><b>BLOCK (3/8 passed).</b></font> Gate 只决定是否进入更高保真研究阶段，不代表生产或实盘批准。阈值由 YAML 配置读取，报告不生成或修改决策。",
            s["body"],
        ),
        Paragraph("Material limitations", s["h2"]),
        Paragraph(
            "1. 数据为社区 Qlib 示例数据，不等同于机构级 point-in-time 行情与成分股主数据。<br/>"
            "2. 全截面 IC 不能直接代表 Top50 尾部组合质量。<br/>"
            "3. 行业、规模和风格暴露尚未显式中性化。<br/>"
            "4. IC 显著性尚未对序列相关和多重检验进行校正。<br/>"
            "5. Gate 阈值属于本研究晋级条件，不是行业统一标准。<br/>"
            "6. 当前结果只覆盖 2017-2020 的指定历史区间。",
            s["body"],
        ),
        Paragraph("Next tests", s["h2"]),
        report_table(
            [["Priority", "Hypothesis", "Test", "Continuation criterion"],
             ["P1", "Top50 tail differs from full cross-section", "quantile and top-k sensitivity", "stable tail monotonicity"],
             ["P1", "industry/size exposures dominate", "neutralized portfolio backtest", "risk-adjusted improvement"],
             ["P1", "reported IC is overstated", "block bootstrap / multiple-test control", "stable adjusted significance"],
             ["P2", "dynamic weights increase noise", "turnover penalty / holding period", "after-cost IR above zero"],
             ["P2", "sample data limits inference", "rolling re-estimation on better data", "cross-period consistency"]],
            [23 * mm, 53 * mm, 62 * mm, 40 * mm], 6.8,
        ),
        Paragraph("Reproducibility record", s["h2"]),
        Paragraph(
            f"Python {experiment['runtime']['python']} · pyqlib {experiment['runtime']['pyqlib']} · "
            f"LightGBM {experiment['runtime']['lightgbm']} · 11 tests · 86.27% core coverage · "
            f"config SHA-256 {experiment['runtime']['config_sha256']}",
            s["small"],
        ),
        Spacer(1, 4 * mm),
        Paragraph(
            "Data and disclaimer: community-maintained Qlib example data; downloader notes Yahoo Finance as the underlying source. Educational, research-engineering and portfolio use only. Not investment advice.",
            s["small"],
        ),
    ]

    def footer(canvas, document):
        canvas.saveState()
        canvas.setTitle("Qlib Factor Robustness Lab - Research Note QR-001")
        canvas.setAuthor("Tian Yihan")
        canvas.setSubject("Quantitative research validation and model-risk case study")
        canvas.setStrokeColor(colors.HexColor(NAVY))
        canvas.setLineWidth(1)
        canvas.line(16 * mm, A4[1] - 8 * mm, A4[0] - 16 * mm, A4[1] - 8 * mm)
        canvas.setFont("MSYH", 7)
        canvas.setFillColor(colors.HexColor(SLATE))
        canvas.drawString(16 * mm, 7 * mm, "QR-001 · Qlib Factor Robustness Lab · v4")
        canvas.drawRightString(A4[0] - 16 * mm, 7 * mm, f"{document.page} / 5")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    experiment, official = load()
    backtests = reports()
    font = FontProperties(fname=str(FONT_REGULAR))
    plot_research_design(experiment, font)
    plot_temporal_control(experiment, font)
    plot_validation_matrix(experiment, font)
    plot_oos_evidence(experiment, backtests, font)
    build_html(experiment, official, backtests)
    pdf = build_pdf(experiment, official, backtests)
    shutil.copyfile(pdf, DOCS / "quant-research-note-v4.pdf")
    shutil.copyfile(pdf, DOCS / "v4" / "quant-research-note-v4.pdf")
    manifest = {
        "showcase_version": "v4",
        "format": "institutional research note",
        "gate_decision": experiment["research_promotion_gate"]["decision"],
        "outputs": [
            "docs/index.html", "docs/v4/index.html", "docs/quant-research-note-v4.pdf",
            "docs/assets/v4_research_design.png", "docs/assets/v4_temporal_control.png",
            "docs/assets/v4_validation_matrix.png", "docs/assets/v4_oos_evidence.png",
        ],
    }
    (EVIDENCE / "v4_showcase_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pdf)


if __name__ == "__main__":
    main()
